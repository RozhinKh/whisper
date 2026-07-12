"""Profile a single Whisper transcription to find remaining CPU/GPU bottlenecks.

Run on a GPU machine. Does one warm-up transcription (to exclude one-time CUDA
context / cuDNN autotune costs), then profiles a second transcription and
prints the top operations by self CPU time and self CUDA time, plus a
breakdown of any remaining sync-inducing ops (.item()/_local_scalar_dense/
cudaStreamSynchronize/nonzero) -- the same class of issue already found and
fixed in the KV cache and beam search code.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/profile_transcribe.py \
        --beam-size 5 --top-n 30
"""

import argparse

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import whisper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--audio", default=None,
                         help="Path to a local audio file. If omitted, pulls one LibriSpeech sample.")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--trace-out", default="whisper_profile_trace.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")

    model = whisper.load_model(args.model, device=device)

    if args.audio:
        audio_input = args.audio
    else:
        from datasets import load_dataset
        ds = load_dataset("librispeech_asr", "clean", split="test",
                           streaming=True, trust_remote_code=True)
        sample = next(iter(ds))
        sr = sample["audio"]["sampling_rate"]
        audio_input = np.array(sample["audio"]["array"], dtype=np.float32)
        print(f"Using LibriSpeech sample: duration={len(audio_input) / sr:.1f}s")

    kwargs = dict(
        beam_size=args.beam_size,
        temperature=args.temperature,
        fp16=(device == "cuda"),
        language="en",
    )

    # warm-up run: excludes one-time CUDA context / cuDNN autotune cost from the profile
    _ = model.transcribe(audio_input, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        _ = model.transcribe(audio_input, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()

    print("\n" + "=" * 90)
    print("TOP OPS BY SELF CPU TIME")
    print("=" * 90)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top_n))

    if device == "cuda":
        print("\n" + "=" * 90)
        print("TOP OPS BY SELF CUDA TIME")
        print("=" * 90)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.top_n))

    sync_keywords = ["item", "_local_scalar_dense", "synchronize", "nonzero", "copy_"]
    print("\n" + "=" * 90)
    print("SYNC-INDUCING OPS (item / _local_scalar_dense / synchronize / nonzero / copy_)")
    print("=" * 90)
    total_sync_cpu_us = 0.0
    total_calls = 0
    for evt in prof.key_averages():
        name_lower = evt.key.lower()
        if any(kw in name_lower for kw in sync_keywords):
            print(f"{evt.key:50s}  count={evt.count:6d}  self_cpu_time_total={evt.self_cpu_time_total / 1000:.2f}ms")
            total_sync_cpu_us += evt.self_cpu_time_total
            total_calls += evt.count
    print(f"\nTotal sync-op self CPU time: {total_sync_cpu_us / 1000:.2f}ms across {total_calls} calls")

    prof.export_chrome_trace(args.trace_out)
    print(f"\nChrome trace exported to {args.trace_out} (open in chrome://tracing or perfetto.dev for a timeline view)")


if __name__ == "__main__":
    main()
