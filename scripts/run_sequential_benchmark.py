"""
Run Artemis ASR benchmark sequentially (one file at a time) using Whisper large-v3.
Outputs a JSON results file compatible with the Artemis benchmark format.

Usage:
    python scripts/run_sequential_benchmark.py \
        --artemis-dir /path/to/artemis-asr-benchmark \
        --audio-dir   /path/to/audio_files \
        --output      results_sequential.json \
        [--compute-type float16] \
        [--beam-size 5] \
        [--use-compile]
"""

import argparse
import json
import os
import time

import torch
import whisper


def load_artemis_manifest(artemis_dir: str):
    """Load test file list from Artemis benchmark manifest."""
    manifest_path = os.path.join(artemis_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        # fallback: list all .wav/.mp3/.flac files in audio_dir
        return None
    with open(manifest_path) as f:
        return json.load(f)


def rtf(audio_duration_s: float, inference_time_s: float) -> float:
    if audio_duration_s == 0:
        return float("inf")
    return inference_time_s / audio_duration_s


def get_audio_duration(audio_path: str) -> float:
    import soundfile as sf
    info = sf.info(audio_path)
    return info.duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artemis-dir", required=True, help="Root of artemis-asr-benchmark repo")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output", default="results_sequential.json")
    parser.add_argument("--compute-type", default="float16", choices=["float16", "float32"])
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--use-compile", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"Model      : large-v3")
    print(f"Compute    : {args.compute_type}")
    print(f"Beam size  : {args.beam_size}")
    print(f"Compile    : {args.use_compile}")
    print()

    model = whisper.load_model(
        "large-v3",
        compute_type=args.compute_type,
        use_compile=args.use_compile,
    )

    manifest = load_artemis_manifest(args.artemis_dir)
    if manifest is None:
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        files = [
            f for f in os.listdir(args.audio_dir)
            if os.path.splitext(f)[1].lower() in audio_exts
        ]
        manifest = [{"audio": f, "reference": ""} for f in sorted(files)]

    results = []
    total_audio_s = 0.0
    total_infer_s = 0.0

    for i, entry in enumerate(manifest):
        audio_path = os.path.join(args.audio_dir, entry["audio"])
        if not os.path.isfile(audio_path):
            print(f"[{i+1}/{len(manifest)}] MISSING: {audio_path}")
            continue

        try:
            dur = get_audio_duration(audio_path)
        except Exception:
            dur = 0.0

        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()

        result = model.transcribe(
            audio_path,
            beam_size=args.beam_size,
            fp16=(args.compute_type == "float16"),
        )

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - t0

        total_audio_s += dur
        total_infer_s += elapsed

        entry_result = {
            "audio": entry["audio"],
            "reference": entry.get("reference", ""),
            "hypothesis": result["text"].strip(),
            "duration_s": dur,
            "inference_s": elapsed,
            "rtf": rtf(dur, elapsed),
            "language": result.get("language", ""),
        }
        results.append(entry_result)
        print(f"[{i+1}/{len(manifest)}] {entry['audio']:40s}  RTF={entry_result['rtf']:.3f}  {elapsed:.2f}s")

    summary = {
        "config": {
            "model": "large-v3",
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "use_compile": args.use_compile,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        },
        "total_audio_s": total_audio_s,
        "total_inference_s": total_infer_s,
        "overall_rtf": rtf(total_audio_s, total_infer_s),
        "n_files": len(results),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOverall RTF : {summary['overall_rtf']:.4f}")
    print(f"Results     : {args.output}")


if __name__ == "__main__":
    main()
