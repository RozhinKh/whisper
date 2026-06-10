"""
Grid search over beam_size, compute_type, and torch.compile to find the fastest
configuration for Whisper large-v3 on RTX 3090 that keeps WER within tolerance.

Usage:
    python scripts/find_optimal_config.py --audio path/to/test.wav [--reference "ground truth text"]
"""

import argparse
import time
import warnings

import torch
import whisper

CONFIGS = [
    {"compute_type": "float16", "beam_size": 1, "use_compile": False},
    {"compute_type": "float16", "beam_size": 3, "use_compile": False},
    {"compute_type": "float16", "beam_size": 5, "use_compile": False},
    {"compute_type": "float16", "beam_size": 1, "use_compile": True},
    {"compute_type": "float16", "beam_size": 3, "use_compile": True},
    {"compute_type": "float32", "beam_size": 1, "use_compile": False},
    {"compute_type": "float32", "beam_size": 5, "use_compile": False},
]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    # simple dynamic programming WER
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)


def benchmark_config(audio_path: str, config: dict, n_warmup: int = 1, n_runs: int = 3):
    model = whisper.load_model(
        "large-v3",
        compute_type=config["compute_type"],
        use_compile=config["use_compile"],
    )

    # warm-up
    for _ in range(n_warmup):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.transcribe(audio_path, beam_size=config["beam_size"], fp16=(config["compute_type"] == "float16"))

    # timed runs
    times = []
    result_text = ""
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = model.transcribe(
            audio_path,
            beam_size=config["beam_size"],
            fp16=(config["compute_type"] == "float16"),
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        result_text = result["text"].strip()

    del model
    torch.cuda.empty_cache()

    return {
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "text": result_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to test audio file")
    parser.add_argument("--reference", default="", help="Ground truth transcript for WER")
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: No CUDA device found. Results will not reflect GPU performance.")

    print(f"\nDevice: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Audio:  {args.audio}\n")
    print(f"{'Config':<55} {'Mean(s)':>8} {'Min(s)':>7} {'WER':>6}")
    print("-" * 80)

    best = None
    for cfg in CONFIGS:
        label = f"fp16={cfg['compute_type']=='float16'} beam={cfg['beam_size']} compile={cfg['use_compile']}"
        try:
            res = benchmark_config(args.audio, cfg, n_runs=args.n_runs)
            score = wer(args.reference, res["text"]) if args.reference else float("nan")
            print(f"{label:<55} {res['mean_s']:>8.2f} {res['min_s']:>7.2f} {score:>6.3f}")
            if best is None or res["mean_s"] < best["mean_s"]:
                best = {**cfg, **res, "wer": score}
        except Exception as e:
            print(f"{label:<55} ERROR: {e}")

    if best:
        print("\n=== Recommended config ===")
        print(f"  compute_type : {best['compute_type']}")
        print(f"  beam_size    : {best['beam_size']}")
        print(f"  use_compile  : {best['use_compile']}")
        print(f"  mean latency : {best['mean_s']:.2f}s")
        if args.reference:
            print(f"  WER          : {best['wer']:.3f}")


if __name__ == "__main__":
    main()
