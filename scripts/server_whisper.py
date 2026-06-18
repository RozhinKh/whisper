"""Minimal OpenAI Whisper (original PyTorch, NOT faster-whisper) HTTP server.

Exposes:
  POST /v1/audio/transcriptions   — OpenAI-compatible transcription
  GET  /health                    — liveness check

Use this with the artemisasrbench CLI's `compare` / `validate` commands to
benchmark the Artemis optimization (speculative decoding) against the
original Whisper baseline through an HTTP interface.

No Docker required — runs directly in the existing venv.

Usage:
    # Baseline (stock large-v3, beam search)
    CUDA_VISIBLE_DEVICES=0 python scripts/server_whisper.py \
        --model large-v3 --beam-size 1 --port 8001

    # Candidate (Artemis optimization: speculative decoding)
    CUDA_VISIBLE_DEVICES=0 python scripts/server_whisper.py \
        --model large-v3 --draft-model base --port 8002
"""

import argparse
import os
import tempfile

from flask import Flask, request, jsonify

import whisper

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="large-v3")
parser.add_argument("--draft-model", default=None,
                     help="Enable speculative decoding with this draft model (e.g. 'base'). "
                          "Omit for stock/baseline behavior.")
parser.add_argument("--spec-window", type=int, default=5)
parser.add_argument("--compute-type", default="float16", choices=["float16", "float32"])
parser.add_argument("--beam-size", type=int, default=5)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--device", default="cuda")
parser.add_argument("--language", default=None,
                     help="Force language (e.g. 'en'). Default: auto-detect.")
parser.add_argument("--port", type=int, default=8001)
args = parser.parse_args()

use_speculative = args.draft_model is not None

print(f"Loading model: {args.model}  compute={args.compute_type}  "
      f"beam={args.beam_size}  device={args.device}")
if use_speculative:
    print(f"Speculative decoding enabled, draft={args.draft_model}, window={args.spec_window}")

_model = whisper.load_model(args.model, device=args.device)
_draft_model = whisper.load_model(args.draft_model, device=args.device) if use_speculative else None
if use_speculative:
    from whisper.speculative import speculative_transcribe

print(f"Model ready. Listening on port {args.port} ...")

app = Flask(__name__)


def _transcribe(tmp_path: str, language):
    lang = language or args.language or "en"
    if use_speculative:
        audio_array = whisper.audio.load_audio(tmp_path)
        result = speculative_transcribe(
            _model, _draft_model, audio_array,
            language=lang,
            fp16=(args.compute_type == "float16"),
            temperature=args.temperature,
            spec_window=args.spec_window,
        )
    else:
        result = _model.transcribe(
            tmp_path,
            language=lang,
            beam_size=args.beam_size,
            temperature=args.temperature,
            fp16=(args.compute_type == "float16"),
        )
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
    app.run(host="0.0.0.0", port=args.port, threaded=True)
