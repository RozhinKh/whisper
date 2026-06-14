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

KV cache design
---------------
Uses the hook-free integer-indexed KV cache (model.make_kv_cache()).  Each
MultiHeadAttention layer accumulates K, V tensors inside its own forward()
rather than via register_forward_hook callbacks.  This eliminates the 128
Python graph-breaks that prevented torch.compile from tracing the decoder.
"""

from typing import List, Optional
import numpy as np
import torch

from .audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES
from .tokenizer import get_tokenizer

_SPEC_WINDOW = 5


def _to_mel(audio: np.ndarray, n_mels: int, device, dtype) -> torch.Tensor:
    t = torch.from_numpy(audio).float()
    mel = log_mel_spectrogram(t, n_mels=n_mels, padding=N_SAMPLES)
    mel = pad_or_trim(mel, N_FRAMES, axis=-1)
    return mel.unsqueeze(0).to(device=device, dtype=dtype)


def _truncate_kv(cache: dict, length: int) -> None:
    """Truncate self-attention KV entries to `length` tokens in-place.

    In the int-indexed cache, even indices hold self-attention [k, v] lists
    and odd indices hold cross-attention [k, v] lists.  Only self-attention
    entries are truncated; cross-attention (computed from fixed audio features)
    is left untouched.  List contents are updated in-place to preserve list
    object identity (required for stable dynamo guards).
    """
    for idx, entry in cache.items():
        if entry[0] is None:
            continue
        if idx % 2 != 0:
            # Odd index → cross-attention; leave alone.
            continue
        k = entry[0]
        if k.shape[1] > length:
            entry[0] = k[:, :length]
            entry[1] = entry[1][:, :length]


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

    t_mel = _to_mel(audio, target.dims.n_mels, t_dev, dtype)
    d_mel = _to_mel(audio, draft.dims.n_mels, d_dev, dtype)

    with torch.no_grad():
        t_feat = target.embed_audio(t_mel)
        d_feat = draft.embed_audio(d_mel)

    # Target tokenizer: governs the output token sequence and suppress masks.
    tokenizer = get_tokenizer(
        multilingual=target.is_multilingual,
        num_languages=target.num_languages,
        language=language,
        task="transcribe",
    )
    init = list(tokenizer.sot_sequence_including_notimestamps)
    eot = tokenizer.eot
    n_init = len(init)

    # Draft tokenizer: used ONLY to prime the draft KV cache with the correct
    # special token IDs. tiny/base (99 languages) have different IDs for
    # <|transcribe|> and <|notimestamps|> than large-v3 (100 languages), so
    # feeding target's init tokens directly would mis-condition the draft.
    d_tokenizer = get_tokenizer(
        multilingual=draft.is_multilingual,
        num_languages=draft.num_languages,
        language=language,
        task="transcribe",
    )
    d_init = list(d_tokenizer.sot_sequence_including_notimestamps)

    # Replicate Whisper's SuppressTokens + ApplyTimestampRules (no-timestamps mode).
    _non_speech = set(tokenizer.non_speech_tokens)
    _ts_begin = tokenizer.timestamp_begin
    _n_vocab_t = target.dims.n_vocab
    _n_vocab_d = draft.dims.n_vocab
    _t_mask = torch.zeros(_n_vocab_t, dtype=torch.bool, device=t_dev)
    _d_mask = torch.zeros(_n_vocab_d, dtype=torch.bool, device=d_dev)
    for _tid in _non_speech:
        if _tid < _n_vocab_t:
            _t_mask[_tid] = True
        if _tid < _n_vocab_d:
            _d_mask[_tid] = True
    _t_mask[_ts_begin:] = True
    _d_mask[_ts_begin:] = True

    def _argmax_t(logits: torch.Tensor) -> int:
        lg = logits.float()
        lg[_t_mask] = float("-inf")
        return int(lg.argmax())

    def _argmax_d(logits: torch.Tensor) -> int:
        lg = logits.float()
        lg[_d_mask] = float("-inf")
        return int(lg.argmax())

    # Create hook-free int-indexed KV caches.
    # Fixed structure ({0: None, ..., N: None} → values become (k,v) tuples after
    # first use) means torch.compile sees a stable dict and won't recompile on
    # every step.
    t_cache = target.make_kv_cache()
    d_cache = draft.make_kv_cache()

    with torch.no_grad():
        # Prime both KV caches with init[:-1].
        # Invariant: caches are at position len(tokens)-1 at the start of every round.
        target.decoder(
            torch.tensor([init[:-1]], device=t_dev, dtype=torch.long),
            t_feat, kv_cache=t_cache,
        )
        draft.decoder(
            torch.tensor([d_init[:-1]], device=d_dev, dtype=torch.long),
            d_feat, kv_cache=d_cache,
        )

        tokens: List[int] = list(init)
        d_pending: Optional[int] = d_init[-1]  # draft's no_timestamps token

        while len(tokens) - n_init < max_new_tokens:
            pos = len(tokens)

            effective_window = min(spec_window, target.dims.n_text_ctx - pos - 1)
            if effective_window <= 0:
                break

            # ── DRAFT PHASE ───────────────────────────────────────────────────
            proposals: List[int] = []
            _d_tok = d_pending if d_pending is not None else tokens[-1]
            d_pending = None
            d_inp = torch.tensor([[_d_tok]], device=d_dev, dtype=torch.long)
            for _ in range(effective_window):
                d_log = draft.decoder(d_inp, d_feat, kv_cache=d_cache)
                tok = _argmax_d(d_log[0, -1])
                proposals.append(tok)
                if tok == eot:
                    break
                d_inp = torch.tensor([[tok]], device=d_dev, dtype=torch.long)

            if not proposals:
                break
            k = len(proposals)

            # ── VERIFICATION PHASE ────────────────────────────────────────────
            v_inp = torch.tensor(
                [[tokens[-1]] + proposals], device=t_dev, dtype=torch.long
            )
            t_log = target.decoder(v_inp, t_feat, kv_cache=t_cache)

            # ── ACCEPT / REJECT ───────────────────────────────────────────────
            n_acc = 0
            correction: Optional[int] = None
            target_preds = [_argmax_t(t_log[0, i]) for i in range(k)]
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
                bonus = _argmax_t(t_log[0, n_acc])
                tokens.append(bonus)
                if bonus == eot:
                    break
                draft.decoder(
                    torch.tensor([[proposals[-1]]], device=d_dev, dtype=torch.long),
                    d_feat, kv_cache=d_cache,
                )
            else:
                tokens.append(correction)
                if correction == eot:
                    break

                want = len(tokens)
                _truncate_kv(t_cache, want - 1)
                _truncate_kv(d_cache, want - 1)

    text = tokenizer.decode([t for t in tokens[n_init:] if t < eot]).strip()
    return {"text": text}
