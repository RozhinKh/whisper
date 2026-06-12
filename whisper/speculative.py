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
from .tokenizer import get_tokenizer

_SPEC_WINDOW = 5


def _to_mel(audio: np.ndarray, n_mels: int, device, dtype) -> torch.Tensor:
    t = torch.from_numpy(audio).float()
    mel = log_mel_spectrogram(t, n_mels=n_mels, padding=N_SAMPLES)
    mel = pad_or_trim(mel, N_FRAMES, axis=-1)  # trim to exactly 3000 mel frames
    return mel.unsqueeze(0).to(device=device, dtype=dtype)


def _truncate_kv(cache: dict, n_audio_ctx: int, length: int) -> None:
    """Truncate self-attention KV caches to `length` timesteps in-place.
    Cross-attention entries (shape[1] == n_audio_ctx) are left untouched.
    The pre-allocated-buffer hook detects the position mismatch on the next
    call and skips the redundant copy (data is already in the buffer).
    """
    for mod in cache:
        t = cache[mod]
        if t.shape[1] == n_audio_ctx:
            continue  # cross-attention — fixed size, never truncate
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
        # Long-form audio: fall back to standard greedy transcription
        return target.transcribe(audio, language=language, fp16=fp16, beam_size=1)

    dtype = torch.float16 if fp16 else torch.float32
    t_dev = target.device
    d_dev = draft.device
    n_audio_ctx = target.dims.n_audio_ctx  # 1500 for all Whisper models

    # Both models share the same mel frame count but may differ in n_mels
    # (tiny/small/medium: 80, large-v3: 128)
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

    t_cache, t_hooks = target.install_kv_cache_hooks()
    d_cache, d_hooks = draft.install_kv_cache_hooks()

    with torch.no_grad():
        # Prime both KV caches with init[:-1].  tokens[-1] (the last init
        # token) is fed as the first input each round so the models see the
        # correct absolute position for every proposal.
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

            # Round 1: cache at pos-1 (primed with init[:-1]); rounds 2+: cache
            # at pos (after truncate/sync).  Worst case offset is pos, so
            # pos + 1 + ew must not exceed n_text_ctx.
            effective_window = min(spec_window, target.dims.n_text_ctx - pos - 1)
            if effective_window <= 0:
                break

            # ── DRAFT PHASE: propose effective_window tokens ──────────────────
            # Draft cache starts at pos-1; feeding tokens[-1] advances it to pos,
            # then each proposal advances it one more step.
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
            # After the draft loop:
            #   draft fed [tokens[-1], p0, ..., p_{k-2}] → k calls → cache at pos-1+k = pos+k-1
            #   (proposals[-1] was generated but NOT fed back)
            # After the target verification pass (v_inp has k+1 tokens):
            #   target cache at pos-1+(k+1) = pos+k
            # Invariant we maintain: caches at len(tokens)-1 before each round.

            # ── VERIFICATION PHASE: target scores all proposals in one pass ──
            # v_inp = [tokens[-1], p0, ..., p_{k-1}]; target cache starts at pos-1.
            v_inp = torch.tensor(
                [[tokens[-1]] + proposals], device=t_dev, dtype=torch.long
            )
            t_log = target.decoder(v_inp, t_feat, kv_cache=t_cache)
            # target cache now at pos+k; draft cache at pos+k-1
            # t_log[0, i] predicts the token at position (pos + i)

            # ── ACCEPT / REJECT ───────────────────────────────────────────────
            n_acc = 0
            correction: Optional[int] = None
            for i, dp in enumerate(proposals):
                tp = int(t_log[0, i].argmax())
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
                # All k draft tokens accepted — claim the free bonus token.
                bonus = int(t_log[0, n_acc].argmax())
                tokens.append(bonus)
                if bonus == eot:
                    break
                # Target at pos+k = len(tokens)-1 ✓
                # Draft at pos+k-1; feed proposals[-1] (position pos+k-1) to reach pos+k.
                draft.decoder(
                    torch.tensor([[proposals[-1]]], device=d_dev, dtype=torch.long),
                    d_feat, kv_cache=d_cache,
                )
                # Both caches now at pos+k = len(tokens)-1 ✓
            else:
                tokens.append(correction)
                if correction == eot:
                    break

                want = len(tokens)  # = pos + n_acc + 1

                # Target at pos+k, draft at pos+k-1.  Truncate both to want-1 = pos+n_acc.
                # This removes the rejected proposal (and any overshoot) from both caches.
                # correction is now tokens[-1] and will be fed as the first token next round.
                _truncate_kv(t_cache, n_audio_ctx, want - 1)
                _truncate_kv(d_cache, n_audio_ctx, want - 1)
                # Both caches now at want-1 = len(tokens)-1 ✓

    for h in t_hooks + d_hooks:
        h.remove()

    text = tokenizer.decode([t for t in tokens[n_init:] if t < eot]).strip()
    return {"text": text}
