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
    parser.add_argument("--stride", type=int, default=1,
                        help="Take every Nth sample (e.g. 13 gives ~200 representative samples from the full 2620)")
    parser.add_argument("--concat-duration", type=float, default=None,
                        help="Concatenate samples into one audio of this length (seconds) and transcribe as a single file")
    parser.add_argument("--compute-type", default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Decoding temperature. 0.0 = greedy with no fallback (default). "
                             "Whisper default uses [0.0,0.2,...,1.0] fallback ladder.")
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--draft-model", default=None,
                        help="Enable speculative decoding with this draft model (e.g. 'tiny').")
    parser.add_argument("--spec-window", type=int, default=5,
                        help="Number of tokens the draft model proposes per round (default 5). "
                             "Smaller values reduce float16 batch-vs-sequential divergence at "
                             "the cost of slightly less speedup.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", default="artemis_results.json")
    args = parser.parse_args()

    use_speculative = args.draft_model is not None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Device    : {gpu_name}")
    print(f"Model     : {args.model}  ({args.compute_type}  beam={args.beam_size}  temp={args.temperature})")
    if use_speculative:
        print(f"Draft     : {args.draft_model}  (speculative decoding, window={args.spec_window})")
    print(f"Dataset   : {args.dataset}")
    print()

    model = whisper.load_model(args.model)
    if use_speculative:
        from whisper.speculative import speculative_transcribe
        draft_model = whisper.load_model(args.draft_model)
        print(f"Draft model loaded: {args.draft_model}")

    if args.use_compile and torch.cuda.is_available():
        import os
        # Single thread avoids inductor subprocess pool OOM crashes.
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        print("Compiling with torch.compile …")

        # Encoder: cudagraphs — input is always [1, n_mels, 3000] (fixed shape).
        # Captures the CUDA graph once on first call, replays it every subsequent call.
        try:
            model.encoder = torch.compile(model.encoder, backend="cudagraphs")
            print("  Encoder compiled (cudagraphs — fixed shape).")
        except Exception as e:
            print(f"  Encoder compile failed: {e}")

        # Decoder: inductor with dynamic=True — handles growing KV-cache length.
        # cudagraphs would crash here because KV-cache shape changes every token step.
        try:
            model.decoder = torch.compile(
                model.decoder, mode="reduce-overhead", dynamic=True
            )
            print("  Decoder compiled (inductor — dynamic shapes).")
        except Exception as e:
            print(f"  Decoder compile failed: {e}")

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

    if args.concat_duration:
        # Collect samples until we have enough audio, then transcribe as one long file
        target_s = args.concat_duration
        chunks, refs, sr = [], [], None
        for sample in dataset:
            arr = np.array(sample[cfg["audio_col"]]["array"], dtype=np.float32)
            sr = sample[cfg["audio_col"]]["sampling_rate"]
            chunks.append(arr)
            refs.append(sample[cfg["text_col"]])
            if sum(len(c) for c in chunks) / sr >= target_s:
                break

        audio_array = np.concatenate(chunks)
        duration_s = len(audio_array) / sr
        reference = " ".join(refs)
        print(f"Concatenated audio: {duration_s:.1f}s from {len(chunks)} samples")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model.transcribe(
                audio_array,
                language=args.language,
                beam_size=args.beam_size,
                temperature=args.temperature,
                fp16=(args.compute_type == "float16"),
            )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        hypothesis = result["text"].strip()
        hypotheses.append(normalizer(hypothesis))
        references.append(normalizer(reference))
        total_audio_s = duration_s
        total_infer_s = elapsed
        n = 1
        print(f"RTF={rtf(duration_s, elapsed):.4f}  elapsed={elapsed:.1f}s")
    else:
        sample_idx = 0
        for sample in dataset:
            if args.max_samples and n >= args.max_samples:
                break
            if sample_idx % args.stride != 0:
                sample_idx += 1
                continue
            sample_idx += 1

            audio_array = np.array(sample[cfg["audio_col"]]["array"], dtype=np.float32)
            sampling_rate = sample[cfg["audio_col"]]["sampling_rate"]
            reference = sample[cfg["text_col"]]
            duration_s = len(audio_array) / sampling_rate

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                if use_speculative:
                    result = speculative_transcribe(
                        model, draft_model, audio_array,
                        language=args.language,
                        fp16=(args.compute_type == "float16"),
                        spec_window=args.spec_window,
                    )
                else:
                    result = model.transcribe(
                        audio_array,
                        language=args.language,
                        beam_size=args.beam_size,
                        temperature=args.temperature,
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
