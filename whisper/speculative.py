"""
Greedy speculative decoding for OpenAI Whisper.

Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"
arXiv:2211.17192

Each verification round:
  1. Draft model proposes `spec_window` tokens autoregressively (cheap).
  2. Target model scores all of them in ONE forward pass (expensive but single call).
  3. Accept tokens greedily until first mismatch; take target token at mismatch.
  4. When all draft tokens accepted, claim a free bonus token from the target.

Net: each round accepts >= 1 token using exactly 1 target forward pass instead
of spec_window passes, reducing target decoder calls by ~(acceptance_rate * window).
Falls back to standard transcription for audio longer than 30 seconds.
"""

from typing import List, Optional
import numpy as np
import torch

from .audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES
from .model import MultiHeadAttention
from .tokenizer import get_tokenizer

_SPEC_WINDOW = 5


def _to_mel(audio: np.ndarray, n_mels: int, device, dtype) -> torch.Tensor:
    t = torch.from_numpy(audio).float()
    mel = log_mel_spectrogram(t, n_mels=n_mels, padding=N_SAMPLES)
    mel = pad_or_trim(mel, N_FRAMES, axis=-1)
    return mel.unsqueeze(0).to(device=device, dtype=dtype)


def _install_kv_hooks(model) -> tuple:
    """Simple KV cache using torch.cat.

    Unlike install_kv_cache_hooks (which uses a pre-allocated buffer with
    _pos/_buf tracking), this avoids all stateful position bookkeeping.
    Truncation via _truncate_kv works correctly: the next torch.cat uses the
    truncated slice as its left operand and creates a fresh correctly-shaped
    tensor.
    """
    cache: dict = {}
    hooks: list = []
    n_text_ctx = model.dims.n_text_ctx

    def save_to_cache(module, _, output):
        if module not in cache or output.shape[1] > n_text_ctx:
            cache[module] = output.detach()
        else:
            cache[module] = torch.cat([cache[module], output.detach()], dim=1)
        return cache[module]

    for layer in model.decoder.modules():
        if isinstance(layer, MultiHeadAttention):
            hooks.append(layer.key.register_forward_hook(save_to_cache))
            hooks.append(layer.value.register_forward_hook(save_to_cache))

    return cache, hooks


def _truncate_kv(cache: dict, n_audio_ctx: int, length: int) -> None:
    """Truncate self-attention KV caches to `length` timesteps in-place.
    Cross-attention entries (shape[1] == n_audio_ctx) are left untouched.
    """
    for mod in cache:
        t = cache[mod]
        if t.shape[1] == n_audio_ctx:
            continue
        if t.shape[1] > length:
            cache[mod] = t[:, :length].detach()


def speculative_transcribe(
    target,
    draft,
    audio: np.ndarray,
    language: str = "en",
    fp16: bool = True,
    spec_window: int = _SPEC_WINDOW,
    max_new_tokens: int = 448,
) -> dict:
    """
    Transcribe audio with speculative decoding.

    Parameters
    ----------
    target : Whisper
        Full target model (e.g. large-v3).
    draft : Whisper
        Smaller draft model (e.g. tiny).  Must share the same tokenizer vocabulary.
    audio : np.ndarray
        Raw waveform at 16 kHz, float32.
    language : str
        BCP-47 language code passed to the tokenizer.
    fp16 : bool
        Run inference in float16 when True.
    spec_window : int
        Number of tokens the draft proposes per verification round.
    max_new_tokens : int
        Hard cap on generated tokens (excluding initial prompt).

    Returns
    -------
    dict with key "text" — same shape as model.transcribe() output.
    """
    if len(audio) > N_SAMPLES:
        return target.transcribe(audio, language=language, fp16=fp16, beam_size=1)

    dtype = torch.float16 if fp16 else torch.float32
    t_dev = target.device
    d_dev = draft.device
    n_audio_ctx = target.dims.n_audio_ctx  # 1500 for all Whisper models

    t_mel = _to_mel(audio, target.dims.n_mels, t_dev, dtype)
    d_mel = _to_mel(audio, draft.dims.n_mels, d_dev, dtype)

    with torch.no_grad():
        t_feat = target.embed_audio(t_mel)
        d_feat = draft.embed_audio(d_mel)

    tokenizer = get_tokenizer(
        multilingual=target.is_multilingual,
        num_languages=target.num_languages,
        language=language,
        task="transcribe",
    )
    init = list(tokenizer.sot_sequence_including_notimestamps)
    eot = tokenizer.eot
    n_init = len(init)

    # Install our own simple KV hooks (torch.cat based — no _buf/_pos state).
    t_cache, t_hooks = _install_kv_hooks(target)
    d_cache, d_hooks = _install_kv_hooks(draft)

    with torch.no_grad():
        # Prime both KV caches with init[:-1].
        # Invariant maintained throughout the loop:
        #   caches are at position len(tokens)-1 at the start of every round.
        # tokens[-1] is therefore the "pending" token not yet in the cache.
        target.decoder(
            torch.tensor([init[:-1]], device=t_dev, dtype=torch.long),
            t_feat, kv_cache=t_cache,
        )
        draft.decoder(
            torch.tensor([init[:-1]], device=d_dev, dtype=torch.long),
            d_feat, kv_cache=d_cache,
        )

        tokens: List[int] = list(init)

        while len(tokens) - n_init < max_new_tokens:
            pos = len(tokens)  # = n_init + accepted so far

            # Cap proposals so v_inp never exceeds positional embedding range.
            effective_window = min(spec_window, target.dims.n_text_ctx - pos - 1)
            if effective_window <= 0:
                break

            # ── DRAFT PHASE ───────────────────────────────────────────────────
            # Cache starts at pos-1; feeding tokens[-1] advances it to pos,
            # then each proposal advances it one more step.
            # After the loop: draft cache at pos-1+k = pos+k-1
            # (proposals[-1] was generated but NOT fed back into draft).
            proposals: List[int] = []
            d_inp = torch.tensor([[tokens[-1]]], device=d_dev, dtype=torch.long)
            for _ in range(effective_window):
                d_log = draft.decoder(d_inp, d_feat, kv_cache=d_cache)
                tok = int(d_log[0, -1].argmax())
                proposals.append(tok)
                if tok == eot:
                    break
                d_inp = torch.tensor([[tok]], device=d_dev, dtype=torch.long)

            if not proposals:
                break
            k = len(proposals)

            # ── VERIFICATION PHASE ────────────────────────────────────────────
            # v_inp = [tokens[-1], p0, …, p_{k-1}]  (k+1 tokens)
            # Target cache starts at pos-1 → after this call: pos+k.
            # t_log[0, i] = target's prediction for absolute position (pos + i).
            v_inp = torch.tensor(
                [[tokens[-1]] + proposals], device=t_dev, dtype=torch.long
            )
            t_log = target.decoder(v_inp, t_feat, kv_cache=t_cache)

            # ── ACCEPT / REJECT ───────────────────────────────────────────────
            n_acc = 0
            correction: Optional[int] = None
            target_preds = [int(t_log[0, i].argmax()) for i in range(k)]
            for i, dp in enumerate(proposals):
                tp = target_preds[i]
                if tp == dp:
                    n_acc += 1
                    if dp == eot:
                        break
                else:
                    correction = tp
                    break

            tokens.extend(proposals[:n_acc])
            if tokens[-1] == eot:
                break

            if correction is None:
                # All k draft tokens accepted — take one free bonus token.
                # Target cache at pos+k = len(tokens)-1+1 (need one more step).
                bonus = int(t_log[0, n_acc].argmax())
                tokens.append(bonus)
                if bonus == eot:
                    break
                # Draft at pos+k-1; feed proposals[-1] (position pos+k-1) to sync.
                draft.decoder(
                    torch.tensor([[proposals[-1]]], device=d_dev, dtype=torch.long),
                    d_feat, kv_cache=d_cache,
                )
                # Both caches now at pos+k = len(tokens)-1. ✓
            else:
                tokens.append(correction)
                if correction == eot:
                    break

                want = len(tokens)  # = pos + n_acc + 1

                # Target at pos+k, draft at pos+k-1.
                # Truncate both to want-1 = pos+n_acc so correction = tokens[-1]
                # becomes the "pending" token fed as input next round.
                _truncate_kv(t_cache, n_audio_ctx, want - 1)
                _truncate_kv(d_cache, n_audio_ctx, want - 1)
                # Both caches now at want-1 = len(tokens)-1. ✓

    for h in t_hooks + d_hooks:
        h.remove()

    text = tokenizer.decode([t for t in tokens[n_init:] if t < eot]).strip()
    return {"text": text}
