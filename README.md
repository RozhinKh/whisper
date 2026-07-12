# Whisper — Inference Optimization

**+21.7% throughput · 0% accuracy loss · PyTorch · RTX 3090**

Three code-level optimizations on [OpenAI Whisper](https://github.com/openai/whisper) `large-v3`, validated on LibriSpeech test-clean (200 samples) with byte-identical transcripts before and after.

→ **[Full report: Optimization_Report.md](Optimization_Report.md)**

---

## Results

### LibriSpeech test-clean (200-sample sweep)

| Metric | Baseline | Optimized | Δ |
|--------|----------|-----------|---|
| RTF (mean) | 0.1677 | 0.1313 | **−21.7 %** |
| WER | 7.002 % | 7.002 % | **0.00 %** |

### artemisasrbench — Sequential

| Scenario | Stock RTF | Optimized RTF | Speedup |
|----------|-----------|---------------|---------|
| control_phrase_v1 | 0.1748 | 0.1688 | +3.4 % |
| clean_short_v1 | 0.2028 | 0.1511 | +25.5 % |
| clean_long_v1 | 0.3295 | 0.2442 | +25.9 % |
| long_form_v1 | 0.2022 | 0.1876 | +7.2 % |
| noisy_snr10_v1 | 0.2039 | 0.1472 | +27.8 % |
| noisy_snr20_v1 | 0.3046 | 0.1960 | +35.7 % |
| telephone_v1 | 0.1964 | 0.1644 | +16.3 % |

### artemisasrbench — Concurrent

| Scenario | Stock RTF | Optimized RTF | Speedup |
|----------|-----------|---------------|---------|
| clean_long_v1 | 0.6942 | 0.4243 | +38.9 % |
| noisy_snr10_v1 | 0.2041 | 0.1493 | +26.8 % |
| noisy_snr20_v1 | 0.3056 | 0.2019 | +33.9 % |
| telephone_v1 | 0.2302 | 0.1652 | +28.2 % |

---

## What changed

### 1 — Cross-attention SDPA kernel fix *(largest contributor)*

PyTorch's Flash Attention silently falls back to the slow unfused math kernel when cross-attention key/value tensors (batch=1, audio-level) are broadcast against beam-expanded query tensors (batch=beam\_size). The fix pre-expands k/v via `.expand()` — a zero-copy stride-0 view — so flash attention runs on every call.

```
math kernel (original):  174.65 µs/call
flash attention (fixed):  20.48 µs/call   →  8.5× per call
```

### 2 — Hook-free KV cache

Replaced 128 `register_forward_hook` callbacks per decode step (32 layers × 2 attention types × 2 projections) with inline integer-indexed dict accumulation inside `MultiHeadAttention.forward()`. Each layer writes directly to its own `{idx: [k, v]}` slot — no Python-level graph breaks, no external callbacks, safe for concurrent access without a serialization lock.

### 3 — Vectorized beam-search bookkeeping

Replaced `n_audio × beam_size × (beam_size+1) × 2` individual `.item()` CPU-GPU round trips per decode step (60 at `beam_size=5`) with a single batched `topk` + `tolist()` transfer. `rearrange_kv_cache` converts `source_indices` to a tensor once and reuses it across all 32 layers.

---

## Concurrent correctness

6 simultaneous requests against the same audio file → all responses byte-identical. The hook-free cache (per-call dict, no shared layer state) requires no `threading.Lock()` around inference. Stock Whisper does.

---

## Hardware & config

```
GPU:        NVIDIA RTX 3090 24 GB
CPU:        Intel Xeon Gold 6230 (80 logical CPUs)
Model:      whisper large-v3
Settings:   beam_size=5  temperature=0.0  fp16=True
Framework:  PyTorch 2.5.1+cu121  CUDA 12.1
```

---

## Benchmark data

Raw JSON results: [`benchmark_results/`](benchmark_results/sequential_concurrent_output/)

---

*Original Whisper README below.*

---

# Whisper (original)

[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://arxiv.org/abs/2212.04356)
[[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.

## Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

## Setup

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably [OpenAI's tiktoken](https://github.com/openai/tiktoken) for their fast tokenizer implementation. You can download and install (or update to) the latest release of Whisper with the following command:

    pip install -U openai-whisper

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

    pip install git+https://github.com/openai/whisper.git

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

### Required dependencies

`ffmpeg` is required to read audio files:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# MacOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

## Available models

|  Size  | Parameters | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:-------------:|:--------------:|
| tiny   |    39 M    |       `tiny`       |     ~1 GB     |      ~10x      |
| base   |    74 M    |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |      `turbo`       |     ~6 GB     |      ~8x       |

## Usage

```python
import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

Command-line:

    whisper audio.flac audio.mp3 audio.wav --model turbo

## License

MIT License.
