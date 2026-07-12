"""Diagnostic: does PyTorch's flash/efficient SDPA backend support the batch
broadcasting pattern used by Whisper's cross-attention (k/v batch=n_audio,
q batch=n_audio*beam_size), or does it require falling back to the slow
'math' backend?

Not part of the optimization itself -- a throwaway diagnostic to confirm or
refute a hypothesis from profiling, before touching whisper/model.py.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/diag_sdpa_backend.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

device = "cuda"
dtype = torch.float16

n_audio = 1
beam = 5
n_head = 20
head_dim = 64
ctx_kv = 40  # cached audio context length

q = torch.randn(n_audio * beam, n_head, 1, head_dim, device=device, dtype=dtype)
k = torch.randn(n_audio, n_head, ctx_kv, head_dim, device=device, dtype=dtype)
v = torch.randn(n_audio, n_head, ctx_kv, head_dim, device=device, dtype=dtype)

print(f"q.shape={tuple(q.shape)}  k.shape={tuple(k.shape)}  v.shape={tuple(v.shape)}")

# 1) what does auto-select pick today (current behavior)?
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
    try:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        print(f"[auto-select, broadcast as-is] OK -> out.shape={tuple(out.shape)}")
    except Exception as e:
        print(f"[auto-select, broadcast as-is] FAILED: {e}")

# 2) force flash+efficient only, exclude math -- does broadcast (batch=1 vs batch=5) work?
for backends, label in [
    ([SDPBackend.FLASH_ATTENTION], "FLASH only"),
    ([SDPBackend.EFFICIENT_ATTENTION], "EFFICIENT only"),
    ([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], "FLASH+EFFICIENT"),
]:
    try:
        with sdpa_kernel(backends):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        print(f"[{label}, broadcast as-is] OK -> out.shape={tuple(out.shape)}")
    except Exception as e:
        print(f"[{label}, broadcast as-is] FAILED: {type(e).__name__}: {e}")

# 3) same test but with k/v pre-expanded to match q's batch (no broadcasting needed)
k_exp = k.expand(n_audio * beam, n_head, ctx_kv, head_dim)
v_exp = v.expand(n_audio * beam, n_head, ctx_kv, head_dim)
print(f"\nk_exp.is_contiguous()={k_exp.is_contiguous()}  (expand = stride-0 view, no copy)")

for backends, label in [
    ([SDPBackend.FLASH_ATTENTION], "FLASH only"),
    ([SDPBackend.EFFICIENT_ATTENTION], "EFFICIENT only"),
]:
    try:
        with sdpa_kernel(backends):
            out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        print(f"[{label}, pre-expanded] OK -> out.shape={tuple(out.shape)}")
    except Exception as e:
        print(f"[{label}, pre-expanded] FAILED: {type(e).__name__}: {e}")

# 4) timing comparison: auto-select (today) vs forced FLASH+EFFICIENT vs pre-expanded FLASH
import time

def bench(fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6  # us per call

print("\n--- timing (us/call, cross-attention shape, n=200) ---")
print(f"auto-select (today):          {bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False)):.2f} us")
try:
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        t = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False))
    print(f"EFFICIENT only, broadcast:     {t:.2f} us")
except Exception as e:
    print(f"EFFICIENT only, broadcast:     FAILED ({type(e).__name__})")
try:
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        t = bench(lambda: F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False))
    print(f"FLASH only, pre-expanded:      {t:.2f} us")
except Exception as e:
    print(f"FLASH only, pre-expanded:      FAILED ({type(e).__name__})")
