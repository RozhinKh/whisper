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

Long-form audio (> 30s) is split into fixed N_SAMPLES (30s) chunks and each
chunk is decoded independently with speculative decoding, then concatenated.
This is simpler than Whisper's native timestamp-based seeking (no VAD-based
boundaries, no cross-chunk text conditioning), so transcription quality at
chunk boundaries may be slightly worse than model.transcribe()'s long-form
algorithm — but every chunk gets the full speculative-decoding speedup.

KV cache design
---------------
Uses the hook-free integer-indexed KV cache (model.make_kv_cache()).  Each
MultiHeadAttention layer accumulates K, V tensors inside its own forward()
rather than via register_forward_hook callbacks.  This eliminates the 128
Python graph-breaks that prevented torch.compile from tracing the decoder.
"""

from typing import List, Optional, Tuple
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
    temperature: float = 0.0,
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
        Raw waveform at 16 kHz, float32.  Audio longer than 30s is split into
        fixed N_SAMPLES chunks, each decoded independently (see module docstring).
    language : str
        BCP-47 language code passed to the tokenizer.
    fp16 : bool
        Run inference in float16 when True.
    temperature : float
        Unused — speculative decoding is always greedy (argmax). Accepted for
        call-site compatibility with model.transcribe()'s signature.
    spec_window : int
        Number of tokens the draft proposes per verification round.
    max_new_tokens : int
        Hard cap on generated tokens per chunk (excluding initial prompt).

    Returns
    -------
    dict with key "text" — same shape as model.transcribe() output.
    """
    if len(audio) <= N_SAMPLES:
        return _speculative_transcribe_chunk(
            target, draft, audio, language, fp16, spec_window, max_new_tokens
        )

    texts: List[str] = []
    for start in range(0, len(audio), N_SAMPLES):
        chunk = audio[start : start + N_SAMPLES]
        if len(chunk) == 0:
            continue
        result = _speculative_transcribe_chunk(
            target, draft, chunk, language, fp16, spec_window, max_new_tokens
        )
        text = result["text"].strip()
        if text:
            texts.append(text)
    return {"text": " ".join(texts)}


def _speculative_transcribe_chunk(
    target,
    draft,
    audio: np.ndarray,
    language: str,
    fp16: bool,
    spec_window: int,
    max_new_tokens: int,
) -> dict:
    """Speculative decoding for a single chunk of audio (<= N_SAMPLES, i.e. <= 30s)."""
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


def _select_beams(indices: List[int], t_cache: dict, d_cache: dict, t_feat, d_feat, t_dev, d_dev):
    """Reorder/subset all batched beam state by `indices` (may repeat, e.g.
    when one beam spawns multiple children, or omit, e.g. dropping finished
    beams).  Mirrors decoding.py's rearrange_kv_cache but for arbitrary index
    lists rather than a pure permutation."""
    idx_t = torch.tensor(indices, device=t_dev, dtype=torch.long)
    idx_d = torch.tensor(indices, device=d_dev, dtype=torch.long)
    for entry in t_cache.values():
        if entry[0] is not None:
            entry[0] = entry[0][idx_t].detach()
            entry[1] = entry[1][idx_t].detach()
    for entry in d_cache.values():
        if entry[0] is not None:
            entry[0] = entry[0][idx_d].detach()
            entry[1] = entry[1][idx_d].detach()
    return t_feat[idx_t], d_feat[idx_d]


def speculative_transcribe_beam(
    target,
    draft,
    audio: np.ndarray,
    language: str = "en",
    fp16: bool = True,
    beam_size: int = 5,
    spec_window: int = _SPEC_WINDOW,
    max_new_tokens: int = 448,
) -> dict:
    """
    EXPERIMENTAL beam-search speculative decoding.

    Maintains up to `beam_size` candidate sequences simultaneously, batched
    across the beam dimension.  Each round: the draft model proposes
    `spec_window` tokens per beam (cheap); the target model batch-verifies
    all beams in ONE forward pass.  All beams are truncated to the common
    (minimum) accepted length across beams so the batch stays rectangular,
    then branched using the target's actual top-`beam_size` log-probabilities
    at that position (not just argmax) — this is what gives genuine beam
    diversity, since pure greedy speculative decoding with no randomness
    would otherwise make every beam identical.  Branching reuses logits
    already computed in the same verification pass (no extra target call).

    This is new, less-tested code relative to speculative_transcribe()'s
    greedy path — built specifically to compare against a beam_size=5
    baseline rather than as a validated production path.

    Returns
    -------
    dict with key "text" — same shape as model.transcribe() output.
    """
    if len(audio) > N_SAMPLES:
        texts: List[str] = []
        for start in range(0, len(audio), N_SAMPLES):
            chunk = audio[start : start + N_SAMPLES]
            if len(chunk) == 0:
                continue
            result = speculative_transcribe_beam(
                target, draft, chunk, language, fp16, beam_size, spec_window, max_new_tokens
            )
            text = result["text"].strip()
            if text:
                texts.append(text)
        return {"text": " ".join(texts)}

    dtype = torch.float16 if fp16 else torch.float32
    t_dev = target.device
    d_dev = draft.device

    tokenizer = get_tokenizer(
        multilingual=target.is_multilingual,
        num_languages=target.num_languages,
        language=language,
        task="transcribe",
    )
    init = list(tokenizer.sot_sequence_including_notimestamps)
    eot = tokenizer.eot
    n_init = len(init)

    d_tokenizer = get_tokenizer(
        multilingual=draft.is_multilingual,
        num_languages=draft.num_languages,
        language=language,
        task="transcribe",
    )
    d_init = list(d_tokenizer.sot_sequence_including_notimestamps)

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

    with torch.no_grad():
        t_mel = _to_mel(audio, target.dims.n_mels, t_dev, dtype)
        d_mel = _to_mel(audio, draft.dims.n_mels, d_dev, dtype)
        t_feat_1 = target.embed_audio(t_mel)
        d_feat_1 = draft.embed_audio(d_mel)

        beam_tokens: List[List[int]] = [list(init)]
        beam_scores: List[float] = [0.0]
        finished: List[Tuple[List[int], float]] = []

        b = 1
        t_cache = target.make_kv_cache()
        d_cache = draft.make_kv_cache()
        t_feat = t_feat_1.repeat(b, 1, 1)
        d_feat = d_feat_1.repeat(b, 1, 1)
        target.decoder(
            torch.tensor([init[:-1]] * b, device=t_dev, dtype=torch.long),
            t_feat, kv_cache=t_cache,
        )
        draft.decoder(
            torch.tensor([d_init[:-1]] * b, device=d_dev, dtype=torch.long),
            d_feat, kv_cache=d_cache,
        )
        # First draft step primes from draft's own no-timestamps token, since
        # tiny/base have different special-token ids than large-v3 (same
        # reasoning as the greedy path's d_pending handling).
        next_draft_input = [d_init[-1]] * b

        total_new = 0
        while total_new < max_new_tokens and beam_tokens:
            b = len(beam_tokens)
            cur_len = len(beam_tokens[0])
            import os
            if os.environ.get("SPEC_BEAM_DEBUG"):
                print(f"[round] b={b} cur_len={cur_len} lens={[len(s) for s in beam_tokens]} "
                      f"next_draft_input={next_draft_input} t_feat.shape={tuple(t_feat.shape)} "
                      f"d_feat.shape={tuple(d_feat.shape)}", flush=True)

            window = max(1, min(spec_window, target.dims.n_text_ctx - cur_len - 1))

            # ── DRAFT PHASE (batched across b beams) ──────────────────────
            d_inp = torch.tensor([[t] for t in next_draft_input], device=d_dev, dtype=torch.long)
            if os.environ.get("SPEC_BEAM_DEBUG"):
                print(f"[round] window={window} d_inp.shape={tuple(d_inp.shape)}", flush=True)
            proposals: List[List[int]] = [[] for _ in range(b)]
            active = [True] * b
            for _ in range(window):
                d_log = draft.decoder(d_inp, d_feat, kv_cache=d_cache)
                step_tok = []
                for i in range(b):
                    if not active[i]:
                        step_tok.append(eot)
                        continue
                    lg = d_log[i, -1].float()
                    lg[_d_mask] = float("-inf")
                    tok = int(lg.argmax())
                    proposals[i].append(tok)
                    if tok == eot:
                        active[i] = False
                    step_tok.append(tok)
                d_inp = torch.tensor([[t] for t in step_tok], device=d_dev, dtype=torch.long)

            k = window
            for i in range(b):
                while len(proposals[i]) < k:
                    proposals[i].append(eot)

            # ── VERIFY PHASE (batched, one target call for all b beams) ───
            v_inp = torch.tensor(
                [[beam_tokens[i][-1]] + proposals[i] for i in range(b)],
                device=t_dev, dtype=torch.long,
            )
            t_log = target.decoder(v_inp, t_feat, kv_cache=t_cache)  # [b, k+1, vocab]

            # ── PER-BEAM GREEDY ACCEPT WALK (vs target argmax) ─────────────
            accept_counts = [0] * b
            for i in range(b):
                for pos in range(k):
                    logits = t_log[i, pos].float()
                    logits[_t_mask] = float("-inf")
                    if int(logits.argmax()) == proposals[i][pos]:
                        accept_counts[i] += 1
                        if proposals[i][pos] == eot:
                            break
                    else:
                        break

            common = min(accept_counts)

            # If every beam fully accepted its window, draft's cache is one
            # token short: the draft loop never feeds its own last proposal
            # back in (same reasoning as the greedy path's "bonus" catch-up
            # call), so it needs an explicit advance here before truncation.
            if common == k:
                last_prop = torch.tensor(
                    [[proposals[i][common - 1]] for i in range(b)], device=d_dev, dtype=torch.long
                )
                draft.decoder(last_prop, d_feat, kv_cache=d_cache)

            # Truncate caches to the common accepted length so the batch
            # stays rectangular for the next round.  cur_len already counts
            # the old pending token (about to be confirmed by this verify
            # call), so the post-accept cache length is cur_len + common —
            # NOT cur_len - 1 + common, which would drop the last accepted
            # proposal's real KV entry instead of just the new pending token.
            common_cache_len = cur_len + common
            _truncate_kv(t_cache, common_cache_len)
            _truncate_kv(d_cache, common_cache_len)

            common_logprobs = [0.0] * b
            for i in range(b):
                lp = 0.0
                for pos in range(common):
                    logits = t_log[i, pos].float()
                    logits[_t_mask] = float("-inf")
                    lp += float(torch.log_softmax(logits, dim=-1)[proposals[i][pos]])
                common_logprobs[i] = lp

            for i in range(b):
                beam_tokens[i] = beam_tokens[i] + proposals[i][:common]
                beam_scores[i] = beam_scores[i] + common_logprobs[i]
            total_new += common

            # Split off any beams that ended in EOT within the common-accepted
            # prefix before branching (branching from a finished beam is
            # meaningless).
            keep_idx = [i for i in range(b) if beam_tokens[i][-1] != eot]
            for i in range(b):
                if beam_tokens[i][-1] == eot:
                    finished.append((beam_tokens[i], beam_scores[i]))
            if not keep_idx:
                beam_tokens = []
                break
            if len(keep_idx) != b:
                t_feat, d_feat = _select_beams(keep_idx, t_cache, d_cache, t_feat, d_feat, t_dev, d_dev)
                beam_tokens = [beam_tokens[i] for i in keep_idx]
                beam_scores = [beam_scores[i] for i in keep_idx]
                t_log = t_log[keep_idx]
                b = len(keep_idx)

            # ── BRANCH: expand each surviving beam using the target's actual
            # top-`beam_size` log-probs at position `common` (already computed
            # in this same verify pass — no extra target call needed), then
            # prune the global candidate pool to top `beam_size`. ───────────
            branch_logits = t_log[:, common].float()
            branch_logits[:, _t_mask] = float("-inf")
            branch_logprobs = torch.log_softmax(branch_logits, dim=-1)
            topk = min(beam_size, branch_logprobs.shape[-1])
            top_vals, top_idx = branch_logprobs.topk(topk, dim=-1)

            candidates = []
            for i in range(b):
                for j in range(topk):
                    tok = int(top_idx[i, j])
                    sc = beam_scores[i] + float(top_vals[i, j])
                    candidates.append((i, tok, sc))
            candidates.sort(key=lambda c: c[2], reverse=True)
            candidates = candidates[:beam_size]

            parent_idx = [c[0] for c in candidates]
            new_tokens = [c[1] for c in candidates]
            new_scores = [c[2] for c in candidates]
            new_beam_tokens = [beam_tokens[p] + [tok] for p, tok in zip(parent_idx, new_tokens)]

            # Reorder/duplicate cache rows to match the new beam composition.
            # No decoder call here: after truncation, cache length already
            # equals (post-accept beam length), i.e. (new length - 1) once
            # the branch token below is appended — the branch token is a
            # brand-new choice from logits, not yet run through a real
            # forward pass, so it stays the "pending" token for next round
            # (same invariant as the greedy path's un-advanced correction).
            t_feat, d_feat = _select_beams(parent_idx, t_cache, d_cache, t_feat, d_feat, t_dev, d_dev)
            total_new += 1

            # Split off newly-EOT'd beams again before the next round.
            keep_idx2 = [i for i in range(len(new_beam_tokens)) if new_beam_tokens[i][-1] != eot]
            for i in range(len(new_beam_tokens)):
                if new_beam_tokens[i][-1] == eot:
                    finished.append((new_beam_tokens[i], new_scores[i]))
            if not keep_idx2:
                beam_tokens = []
                break
            if len(keep_idx2) != len(new_beam_tokens):
                t_feat, d_feat = _select_beams(keep_idx2, t_cache, d_cache, t_feat, d_feat, t_dev, d_dev)

            beam_tokens = [new_beam_tokens[i] for i in keep_idx2]
            beam_scores = [new_scores[i] for i in keep_idx2]
            next_draft_input = [seq[-1] for seq in beam_tokens]

        for seq, sc in zip(beam_tokens, beam_scores):
            finished.append((seq, sc))

        if not finished:
            return {"text": ""}

        best_seq, _ = max(
            finished, key=lambda x: x[1] / max(len(x[0]) - n_init, 1)
        )

    text = tokenizer.decode([t for t in best_seq[n_init:] if t < eot]).strip()
    return {"text": text}
