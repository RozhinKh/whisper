"""Capture latency + transcript for a single audio file against one endpoint.
Used for the intake form's "Demo Audio Clips" section -- a real, live
measurement against the actual running server, not derived from any
artemisasrbench JSON.

Usage:
    python scripts/capture_demo_clip.py --endpoint http://localhost:8009 \
        --audio /path/to/file.flac --duration 300.0 --label stock
"""
import argparse
import time

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", required=True)
parser.add_argument("--audio", required=True)
parser.add_argument("--duration", type=float, required=True, help="Audio duration in seconds")
parser.add_argument("--label", default="")
args = parser.parse_args()

with open(args.audio, "rb") as f:
    audio_bytes = f.read()

t0 = time.perf_counter()
resp = requests.post(
    f"{args.endpoint}/v1/audio/transcriptions",
    files={"file": ("audio.flac", audio_bytes, "audio/flac")},
    data={"language": "en"},
    timeout=900,
)
latency_s = time.perf_counter() - t0

resp.raise_for_status()
text = resp.json()["text"]
rtf = latency_s / args.duration

print(f"\n=== {args.label} ===")
print(f"Endpoint: {args.endpoint}")
print(f"Latency: {latency_s * 1000:.0f} ms")
print(f"RTF: {rtf:.4f}")
print(f"Transcript:\n{text}")
