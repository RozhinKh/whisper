"""
HuggingFace Open ASR Leaderboard evaluation for original OpenAI Whisper.
Methodology: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard

Runs sequentially (one file at a time) and outputs artemis_results.json.

Usage:
    python scripts/hf_asr_leaderboard_eval.py \
        --model large-v3 \
        [--dataset librispeech_asr] \
        [--split test.clean] \
        [--max-samples 100] \
        [--output artemis_results.json]

Datasets (--dataset / --split):
    librispeech_asr      test.clean | test.other
    mozilla-foundation/common_voice_15_0  en/test
    speechcolab/gigaspeech  test
    LIUM/tedlium         test
"""

import argparse
import json
import time
import warnings

import numpy as np
import torch
import whisper
from datasets import load_dataset
from jiwer import wer as compute_wer
from whisper.normalizers import EnglishTextNormalizer

DATASET_CONFIGS = {
    "librispeech_asr": {
        "path": "librispeech_asr",
        "name": "clean",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "streaming": True,
    },
    "librispeech_asr_other": {
        "path": "librispeech_asr",
        "name": "other",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "streaming": True,
    },
    "common_voice": {
        "path": "mozilla-foundation/common_voice_15_0",
        "name": "en",
        "split": "test",
        "audio_col": "audio",
        "text_col": "sentence",
        "streaming": True,
    },
    "tedlium": {
        "path": "LIUM/tedlium",
        "name": "release3",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "streaming": True,
    },
    "gigaspeech": {
        "path": "speechcolab/gigaspeech",
        "name": "xs",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "streaming": True,
    },
}


def rtf(audio_duration_s: float, inference_s: float) -> float:
    return inference_s / max(audio_duration_s, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--dataset", default="librispeech_asr",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of samples (None = full dataset)")
    parser.add_argument("--compute-type", default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", default="artemis_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Device    : {gpu_name}")
    print(f"Model     : {args.model}  ({args.compute_type}  beam={args.beam_size})")
    print(f"Dataset   : {args.dataset}")
    print()

    model = whisper.load_model(args.model)

    normalizer = EnglishTextNormalizer()
    cfg = DATASET_CONFIGS[args.dataset]

    dataset = load_dataset(
        cfg["path"],
        cfg.get("name"),
        split=cfg["split"],
        streaming=cfg["streaming"],
        trust_remote_code=True,
    )

    hypotheses, references = [], []
    total_audio_s, total_infer_s = 0.0, 0.0
    n = 0

    for sample in dataset:
        if args.max_samples and n >= args.max_samples:
            break

        audio_array = np.array(sample[cfg["audio_col"]]["array"], dtype=np.float32)
        sampling_rate = sample[cfg["audio_col"]]["sampling_rate"]
        reference = sample[cfg["text_col"]]
        duration_s = len(audio_array) / sampling_rate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            result = model.transcribe(
                audio_array,
                language=args.language,
                beam_size=args.beam_size,
                fp16=(args.compute_type == "float16"),
            )

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        hypothesis = result["text"].strip()
        hypotheses.append(normalizer(hypothesis))
        references.append(normalizer(reference))
        total_audio_s += duration_s
        total_infer_s += elapsed
        n += 1

        if n % 10 == 0 or n == 1:
            running_wer = 100 * compute_wer(references, hypotheses)
            print(f"[{n:4d}] WER={running_wer:.2f}%  RTF={rtf(duration_s, elapsed):.3f}")

    word_error_rate = 100 * compute_wer(references, hypotheses)
    overall_rtf = rtf(total_audio_s, total_infer_s)

    print(f"\n{'='*50}")
    print(f"Samples   : {n}")
    print(f"WER       : {word_error_rate:.3f}%")
    print(f"RTF       : {overall_rtf:.4f}")
    print(f"Audio     : {total_audio_s/3600:.2f}h  Inference: {total_infer_s/60:.1f}min")

    results = [
        {
            "dataset": args.dataset,
            "model": args.model,
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "device": gpu_name,
            "n_samples": n,
            "wer": round(word_error_rate, 4),
            "rtf": round(overall_rtf, 4),
            "total_audio_hours": round(total_audio_s / 3600, 4),
            "total_inference_min": round(total_infer_s / 60, 4),
        }
    ]

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
