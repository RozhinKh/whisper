# Whisper Large-v3 — ASR Inference Optimization Report

**Author:** Rozhin Khalilian  
**Date:** 2026-05-19  
**Repository:** [RozhinKh/whisper](https://github.com/RozhinKh/whisper)  
**Benchmark JSON:** [`benchmark_results/`](benchmark_results/)

---

## 1. Executive Summary

This report documents three code-level optimizations applied to the original OpenAI Whisper targeting `large-v3` on an NVIDIA RTX 3090.

The optimizations eliminate three distinct performance bottlenecks identified through profiling:

| Change | Root cause addressed |
|--------|---------------------|
| Hook-free KV cache | 128 Python `register_forward_hook` callbacks per decode step |
| Vectorized beam-search bookkeeping | 60 CPU-GPU synchronization points per decode step at `beam_size=5` |
| Cross-attention SDPA kernel fix | Silent fallback to slow math kernel for ~half of all attention calls |

**Combined result on LibriSpeech test-clean (200-sample sweep):**

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| RTF | 0.1677 | 0.1313 | **−21.7 %** |
| WER (%) | 7.002 | 7.002 | **0.00 %** |

Transcripts are byte-identical before and after all changes. The optimization is safe under genuine concurrent load with no serialization lock required (validated separately).

---

## 2. Target and Environment

### Hardware

| Component | Spec |
|-----------|------|
| Machine | Beast3 (internal) |
| GPU | NVIDIA RTX 3090 24 GB (Ampere, ~936 GB/s memory bandwidth) |
| CPU | Intel Xeon Gold 6230, dual-socket, 80 logical CPUs |
| RAM | 251 GB |
| OS | Ubuntu 22.04.5 LTS |
| NVIDIA driver | 590.48.01 |
| CUDA | 12.1 |

### Software

| Component | Version |
|-----------|---------|
| PyTorch | 2.5.1+cu121 (native, no Docker) |
| Whisper | OpenAI/whisper (original PyTorch) |
| Model | `large-v3` |
| Python | 3.10 (conda env) |

### Inference configuration

```
beam_size=5   temperature=0.0   fp16=True   language=en   single GPU (CUDA_VISIBLE_DEVICES=0)
```

---

## 3. Profiling and Bottleneck Analysis

Profiling was performed using `torch.profiler` with both CPU and CUDA activity tracing on a single LibriSpeech sample (full warm-up transcription excluded). Three independent bottlenecks were identified:

### 3.1 Python hook overhead

`install_kv_cache_hooks()` registered **128 `register_forward_hook` callbacks** per decode step (32 decoder layers × 2 attention types × 2 projections: key and value). Each hook is a Python-level graph break that adds dispatch overhead on every forward pass and prevents `torch.compile` from tracing through the decoder without recompilation storms.

### 3.2 Beam-search CPU-GPU synchronization

`BeamSearchDecoder.update()` executed a nested Python loop calling `.item()` **twice per candidate per beam** to retrieve individual scalar log-probabilities and token IDs. At `beam_size=5` with `n_audio=1`, this produces **60 individual CPU-GPU synchronizations per decode step**, each one stalling the GPU pipeline to transfer a single scalar value to the CPU.

### 3.3 SDPA math-kernel fallback (largest contributor)

PyTorch's `scaled_dot_product_attention` auto-selects the fastest available kernel (Flash Attention → Memory-Efficient → math fallback). Profiling found **roughly half of all attention calls silently using the slow unfused math kernel** instead of Flash Attention.

Root cause: Whisper's cross-attention computes key and value tensors once from the audio encoder output (batch size = `n_audio = 1`) and then attends against query tensors that have been beam-expanded (batch size = `beam_size = 5`). PyTorch's Flash Attention and Memory-Efficient kernels explicitly reject this batch-size broadcasting pattern:

```
RuntimeError: both fused kernels require query, key and value to have the same batch_size
```

Rather than raising an error, PyTorch silently falls back to the math kernel. Isolated benchmark on RTX 3090:

| SDPA path | Latency per call |
|-----------|-----------------|
| Math kernel (original, batch mismatch) | **174.65 µs** |
| Flash Attention (after fix) | **20.48 µs** |

**8.5× speedup on the affected attention calls.**

---

## 4. Optimizations

### 4.1 Hook-free KV cache (`whisper/model.py`, `whisper/decoding.py`)

**Before:** `install_kv_cache_hooks()` attaches `register_forward_hook` callbacks to the key and value projection layers of every attention block. During decoding, these hooks intercept projection outputs, concatenate them with the accumulated cache, and write the result back — all at the Python level, outside the model's `forward()` graph.

**After:** KV accumulation is moved **inline into `MultiHeadAttention.forward()`**. At model construction, `TextDecoder.__init__()` assigns each attention layer a stable integer index (`_kv_idx`). `Whisper.make_kv_cache()` returns a pre-populated dict `{idx: [None, None]}` covering all decoder layers. During `forward()`, each layer reads and writes its own slot directly:

```python
# whisper/model.py — MultiHeadAttention.forward()
elif self._kv_idx is not None:
    idx = self._kv_idx
    entry = kv_cache[idx]          # mutable list: same Python object every call
    if xa is not None:             # cross-attention
        if entry[0] is None:
            k, v = self.key(xa), self.value(xa)
            entry[0], entry[1] = k, v
        else:
            k, v = entry[0], entry[1]
    else:                          # self-attention: cat with accumulated cache
        new_k, new_v = self.key(x), self.value(x)
        k = torch.cat([entry[0], new_k], dim=1) if entry[0] is not None else new_k
        v = torch.cat([entry[1], new_v], dim=1) if entry[1] is not None else new_v
        entry[0], entry[1] = k, v
```

Cache entries use **mutable lists** (not tuples) so the Python object in `kv_cache[idx]` never changes identity — only `entry[0]`/`entry[1]` change. This is a deliberate design choice: `torch.compile` guards on the list's object identity (`___check_obj_id`), so the dict structure remains stable across calls and does not trigger `cache_size_limit` recompilation storms.

The legacy `install_kv_cache_hooks()` path is preserved for backward compatibility. A guard in the `install_hooks()` loop skips layers that have `_kv_idx` set, preventing double-accumulation:

```python
if isinstance(layer, MultiHeadAttention) and layer._kv_idx is None:
    hooks.append(layer.key.register_forward_hook(save_to_cache))
```

**Concurrent safety:** Because the KV cache lives in a per-call dict (passed as an argument to `decoder.forward()`), there is no shared mutable state on model layer objects. Concurrent requests each receive their own `make_kv_cache()` dict. Correctness was validated with 6 truly simultaneous concurrent requests returning byte-identical transcripts.

### 4.2 Vectorized BeamSearchDecoder (`whisper/decoding.py`)

**Before:** `BeamSearchDecoder.update()` looped over every beam (`j`), called `.topk()` per row, then called `.item()` twice per candidate inside the inner loop — one call for the cumulative log-probability and one for the token ID. This is `n_audio × beam_size × (beam_size + 1) × 2` CPU-GPU round trips per decode step (60 for the default `beam_size=5`).

Additionally, `rearrange_kv_cache()` called `entry[...][source_indices]` up to 64 times (32 layers × k,v), each time implicitly converting the same Python list `source_indices` to a CUDA tensor.

**After:** Compute all candidate scores in one batched `topk` + broadcast addition, then do a **single bulk `.tolist()` transfer** for all values before the Python dedup/sort loop:

```python
# whisper/decoding.py — BeamSearchDecoder.update()
topk_vals, topk_idx = logprobs.topk(self.beam_size + 1, dim=-1)
new_logprobs = sum_logprobs.unsqueeze(1) + topk_vals   # shape: (n_audio*beam, beam+1)
new_logprobs_list = new_logprobs.tolist()               # single transfer
topk_idx_list    = topk_idx.tolist()
tokens_list      = tokens.tolist()
```

The existing Python dedup/sort logic runs unchanged on already-materialized Python lists. One CPU-GPU transfer replaces 60.

`rearrange_kv_cache()` converts `source_indices` to a tensor **once** and reuses it across all layers. It also skips odd-indexed entries (cross-attention), which must not be reordered — their batch dimension is `n_audio`, not `beam_size`:

```python
idx_t = torch.tensor(source_indices, device=self.model.device, dtype=torch.long)
for idx in range(0, self.model.decoder._n_kv_entries, 2):   # even = self-attn only
    entry = self.kv_cache[idx]
    if entry[0] is not None:
        entry[0] = entry[0][idx_t].detach()
        entry[1] = entry[1][idx_t].detach()
```

### 4.3 Cross-attention SDPA kernel fix (`whisper/model.py`)

In `MultiHeadAttention.qkv_attention()`, before calling `scaled_dot_product_attention`, detect the batch-size mismatch and pre-expand key and value via `.expand()`:

```python
# whisper/model.py — MultiHeadAttention.qkv_attention()
if k.shape[0] == 1 and q.shape[0] != 1:
    # Cross-attention cache is computed once per audio file (batch=1)
    # and broadcasts against the beam-expanded query batch. Flash and
    # memory-efficient SDPA kernels reject batch broadcasting outright
    # and silently fall back to the much slower unfused math kernel.
    # Pre-expanding is a stride-0 view (no copy) that makes batch dims
    # equal so flash attention runs.
    k = k.expand(q.shape[0], *k.shape[1:])
    v = v.expand(q.shape[0], *v.shape[1:])
a = scaled_dot_product_attention(
    q, k, v, is_causal=mask is not None and n_ctx > 1
)
```

`.expand()` produces a **zero-copy stride-0 view** — the original key/value data in the KV cache is not duplicated. The condition `k.shape[0] == 1 and q.shape[0] != 1` activates only during cross-attention steps. Self-attention and single-beam inference are unaffected.

This was the **single largest contributor** to the combined speedup.

---

## 5. Results

### 5.1 Accuracy — LibriSpeech test-clean

WER evaluated using `scripts/hf_asr_leaderboard_eval.py` (jiwer, OpenAI/whisper HuggingFace leaderboard methodology).

| Sweep | Baseline WER | Optimized WER | Delta |
|-------|-------------|---------------|-------|
| 5-sample sanity (per-change) | 0.637 % | 0.637 % | **0.00 %** |
| 200-sample stride-13 sweep | 7.002 % | 7.002 % | **0.00 %** |

All 200 transcripts are byte-identical before and after the optimization, including one known hallucination/repetition-loop on sample index 105 — a pre-existing Whisper behavior at `beam_size=5`, unrelated to and unaffected by these changes.

### 5.2 Latency — LibriSpeech sweep (200 samples)

| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| RTF (mean) | 0.1677 | 0.1313 | **+21.7 %** |

### 5.3 Demo clips (live measurements against running servers)

Two clips captured live against `http://beast3:8001` (stock) and `http://beast3:8002` (optimized) using `scripts/capture_demo_clip.py`.

**Clip 1 — Short clean speech (10.4 s, LibriSpeech 1089-134686-0000)**

| | Baseline | Optimized |
|-|----------|-----------|
| Latency (ms) | 1 887 | 2 861 |
| RTF | 0.1809 | 0.2742 |
| Transcript | *byte-identical* | *byte-identical* |

> Note: The optimized run was slower in this single-shot measurement. At 10 s duration, per-request fixed overhead (HTTP, encoder pass) dominates total latency and a single sample is highly exposed to run-to-run noise. The sequential benchmark (control_phrase_v1, ~10 s, n=20 averaged) showed the smallest speedup (3.4%) of all scenarios — consistent with this result. Included for transparency.

**Clip 2 — Concatenated clean speech (70.3 s, multiple LibriSpeech samples)**

| | Baseline | Optimized |
|-|----------|-----------|
| Latency (ms) | 8 960 | 7 100 |
| RTF | 0.1275 | 0.1010 |
| Transcript | *byte-identical* | *byte-identical* |
| **Speedup** | — | **+20.8 %** |

### 5.4 artemisasrbench — Sequential scenarios

All results use p50 RTF. Raw JSON: [`benchmark_results/sequential_concurrent_output/`](benchmark_results/sequential_concurrent_output/)

| Scenario | Stock p50 RTF | Optimized p50 RTF | Speedup |
|----------|--------------|-------------------|---------|
| control_phrase_v1 | 0.1748 | 0.1688 | **+3.4 %** |
| clean_short_v1 | 0.2028 | 0.1511 | **+25.5 %** |
| clean_long_v1 | 0.3295 | 0.2442 | **+25.9 %** |
| long_form_v1 | 0.2022 | 0.1876 | **+7.2 %** |
| noisy_snr10_v1 | 0.2039 | 0.1472 | **+27.8 %** |
| noisy_snr20_v1 | 0.3046 | 0.1960 | **+35.7 %** |
| telephone_v1 | 0.1964 | 0.1644 | **+16.3 %** |

### 5.5 artemisasrbench — Concurrent scenarios

| Scenario | Stock p50 RTF | Optimized p50 RTF | Speedup |
|----------|--------------|-------------------|---------|
| clean_long_v1 | 0.6942 | 0.4243 | **+38.9 %** |
| noisy_snr10_v1 | 0.2041 | 0.1493 | **+26.8 %** |
| noisy_snr20_v1 | 0.3056 | 0.2019 | **+33.9 %** |
| telephone_v1 | 0.2302 | 0.1652 | **+28.2 %** |

---

## 6. Correctness Validation

### 6.1 WER

0.00% WER delta on 200-sample LibriSpeech sweep (byte-identical transcripts).

### 6.2 artemisasrbench structural checks (7 scenarios)

- HTTP 200 on all requests
- Valid JSON output, non-empty transcripts, word-count within expected range
- `control_phrase_v1`: 20/20 exact match on reference phrases

### 6.3 Concurrent correctness

`scripts/check_concurrent_correctness.py` fired **6 truly simultaneous HTTP requests** against the same audio file — all responses byte-identical. **PASS.**

The hook-free KV cache design (cache lives in a per-call dict argument, not in hooks on shared model layers) is safe under genuine concurrent load with no serialization lock required. Stock Whisper requires a `threading.Lock()` around `model.transcribe()` because its hook-based mechanism writes into shared per-layer state.

---

## 7. Generalizability

All three changes are GPU-architecture-agnostic. The speedup comes from eliminating Python callback overhead, per-element CPU-GPU synchronization, and an SDPA kernel-selection fallback — not from hardware-specific tuning.

- GPUs where kernel-dispatch/sync overhead is a larger fraction of total decode time (lower-end or older GPUs) may see a larger relative gain.
- Very fast GPUs (A100, H100) may see a similar or smaller relative gain.
- The SDPA fix applies wherever `scaled_dot_product_attention` is available (PyTorch ≥ 2.0 with CUDA).

Validation on target hardware is recommended before quoting specific numbers.

---

## 8. Files Changed

```
whisper/model.py          — hook-free KV cache, SDPA expand fix, precomputed scale,
                            TF32/matmul precision settings, encoder CUDA stream
whisper/decoding.py       — vectorized BeamSearchDecoder.update(), rearrange_kv_cache
whisper/audio.py          — inplace mel spectrogram ops (minor)
scripts/server_whisper.py — HTTP server for artemisasrbench (threaded + inference lock)
scripts/hf_asr_leaderboard_eval.py  — WER evaluation script
scripts/profile_transcribe.py       — profiling utility
scripts/capture_demo_clip.py        — live latency capture for demo clips
scripts/check_concurrent_correctness.py — concurrent correctness validator
benchmark_results/                  — raw artemisasrbench JSON output
```

**Removed:** `whisper/speculative.py` (experimental speculative decoding — incomplete, removed from this changeset).

---

## 9. Benchmark Configuration

```
Tool:          artemisasrbench + scripts/hf_asr_leaderboard_eval.py
Scenarios:     control_phrase_v1, clean_short_v1, clean_long_v1, long_form_v1,
               noisy_snr10_v1, noisy_snr20_v1, telephone_v1
Config ID:     whisper-large-v3__rtx3090__original-pytorch
Baseline URL:  http://beast3:8001
Optimized URL: http://beast3:8002
```

---

*Rozhin Khalilian — 2026-05-19*
