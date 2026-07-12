"""Minimal OpenAI Whisper (original PyTorch, NOT faster-whisper) HTTP server.

Exposes:
  POST /v1/audio/transcriptions   — OpenAI-compatible transcription
  GET  /health                    — liveness check

Use this with the artemisasrbench CLI's `compare` / `validate` commands to
benchmark the Artemis optimization against the original Whisper baseline.

The optimization (hook-free KV cache + vectorized beam search bookkeeping)
applies unconditionally — there is no flag to toggle it. To compare
baseline vs optimized, run this exact same script and command from two
different git checkouts: `main` (unmodified Whisper) for baseline, and
this feature branch for optimized.

No Docker required — runs directly in the existing venv.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/server_whisper.py \
        --model large-v3 --beam-size 5 --port 8001
"""

import argparse
import os
import tempfile
import threading

from flask import Flask, request, jsonify

import whisper

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="large-v3")
parser.add_argument("--compute-type", default="float16", choices=["float16", "float32"])
parser.add_argument("--beam-size", type=int, default=5)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--device", default="cuda")
parser.add_argument("--language", default=None,
                     help="Force language (e.g. 'en'). Default: auto-detect.")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--no-inference-lock", action="store_true",
                     help="Disable the inference lock. Only safe for the hook-free KV "
                          "cache implementation (this branch) at temperature=0.0 -- the "
                          "original hook-based cache (main branch) corrupts under "
                          "concurrent access without this lock.")
args = parser.parse_args()

print(f"Loading model: {args.model}  compute={args.compute_type}  "
      f"beam={args.beam_size}  device={args.device}")

_model = whisper.load_model(args.model, device=args.device)

print(f"Model ready. Listening on port {args.port} ...")

app = Flask(__name__)

# Concurrent HTTP connections (multipart upload parsing, etc.) are safe and
# need real threading -- Flask's threaded=False serializes the whole request,
# which corrupted simultaneous large-file uploads under concurrent load.
#
# The inference lock below guards the actual model call. It's required for
# the original hook-based KV cache (hooks attach to shared model layers,
# so concurrent calls can write into each other's cache). The hook-free KV
# cache on this branch passes the cache as a per-call argument instead, with
# no shared mutable state on the model layers themselves -- so it may not
# need this lock at all. --no-inference-lock tests that hypothesis; verify
# output correctness before trusting any speed numbers collected with it.
_inference_lock = threading.Lock()
_use_lock = not args.no_inference_lock


def _transcribe(tmp_path: str, language):
    lang = language or args.language or "en"
    kwargs = dict(
        language=lang,
        beam_size=args.beam_size,
        temperature=args.temperature,
        fp16=(args.compute_type == "float16"),
    )
    if _use_lock:
        with _inference_lock:
            result = _model.transcribe(tmp_path, **kwargs)
    else:
        result = _model.transcribe(tmp_path, **kwargs)
    return result["text"].strip()


@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400

    audio_bytes = request.files["file"].read()
    language = request.form.get("language")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        text = _transcribe(tmp_path, language)
    finally:
        os.unlink(tmp_path)

    return jsonify({"text": text})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # threaded=True + _inference_lock (above): lets Flask handle concurrent
    # HTTP connections/uploads correctly (threaded=False corrupted
    # simultaneous large multipart uploads under concurrent benchmark load),
    # while the lock still serializes access to the shared model instance to
    # prevent KV cache corruption ("Key and Value must have the same sequence
    # length") that threaded=True caused on its own.
    app.run(host="0.0.0.0", port=args.port, threaded=True)
