"""Throwaway correctness check: fire N truly concurrent requests for the SAME
audio file against a server and verify the transcripts are byte-identical.

beam_size=5, temperature=0.0 is deterministic -- if concurrent access to a
shared model instance corrupts state (e.g. KV cache), transcripts will
differ or come back garbled. Identical output across all N requests is
strong evidence the server is safe for concurrent access without a lock.

Usage:
    python scripts/check_concurrent_correctness.py \
        --endpoint http://localhost:8002 --audio path/to/file.flac --n 6
"""
import argparse
import concurrent.futures
import time

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", required=True)
parser.add_argument("--audio", required=True)
parser.add_argument("--n", type=int, default=6)
args = parser.parse_args()

with open(args.audio, "rb") as f:
    audio_bytes = f.read()


def send_one(i):
    t0 = time.perf_counter()
    resp = requests.post(
        f"{args.endpoint}/v1/audio/transcriptions",
        files={"file": ("audio.flac", audio_bytes, "audio/flac")},
        data={"model": "whisper-large-v3", "language": "en"},
        timeout=900,
    )
    elapsed = time.perf_counter() - t0
    if resp.status_code != 200:
        return i, f"<HTTP {resp.status_code}: {resp.text[:200]}>", elapsed
    return i, resp.json().get("text", "<no text field>"), elapsed


print(f"Firing {args.n} truly concurrent requests for the same audio file...")
with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as pool:
    t_start = time.perf_counter()
    futures = [pool.submit(send_one, i) for i in range(args.n)]
    results = [f.result() for f in futures]
    total_elapsed = time.perf_counter() - t_start

results.sort(key=lambda r: r[0])
texts = [r[1] for r in results]

print(f"\nTotal wall time for all {args.n} concurrent requests: {total_elapsed:.1f}s")
for i, text, elapsed in results:
    print(f"  [{i}] {elapsed:.1f}s  len={len(text)}  {text[:80]!r}...")

unique = set(texts)
print(f"\nUnique transcripts: {len(unique)} (expect 1 if correct)")
if len(unique) == 1:
    print("PASS -- all concurrent requests returned identical output.")
else:
    print("FAIL -- transcripts differ. Concurrent access is corrupting state.")
    for u in unique:
        print(f"  variant: {u[:150]!r}")
