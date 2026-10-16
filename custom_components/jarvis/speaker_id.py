"""
JARVIS — in-process speaker identification (vocal biometrics, Path B; issue #397).

The voice analog of ``llm_recognition.py``: turn an utterance into a voiceprint
and cosine-match it against the people enrolled in ``voice_roster``. Like the
LLM face matcher, a result here is a **guess** — it is panel/advisory by default
and only becomes a *trusted* identity signal above a conservative threshold with
enough enrolled samples, so a mis-match can never silence an intrusion alert.

This module owns the matching math (pure Python, no heavy deps) and the seam for
the embedding backend. **Step 1 ships the matcher only** — ``embed`` has no
backend yet, so ``backend_available()`` is False and ``identify`` is inert until
Step 2 wires an audio-capture path and a speaker-embedding model. Everything is
best-effort and never raises to the caller.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

_LOGGER = logging.getLogger(__name__)

# Decision thresholds (cosine over L2-normalized embeddings, so in [-1, 1]).
# A *trusted* match must clear TRUSTED_THRESHOLD and have at least
# MIN_SAMPLES_FOR_TRUST enrolled samples; a weaker match down to GUESS_THRESHOLD
# is returned as a non-trusted guess (panel only). Conservative on purpose.
TRUSTED_THRESHOLD = 0.75
GUESS_THRESHOLD = 0.55
MIN_SAMPLES_FOR_TRUST = 2


def _cfg(key: str, default):
    try:
        from . import jarvis_config
        return jarvis_config.get(key, default)
    except Exception:
        return default


def cosine(a, b) -> float:
    """Cosine similarity of two equal-length vectors, in [-1, 1]. Returns 0.0
    for mismatched lengths or a zero vector (never raises)."""
    try:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / (na * nb)
    except Exception:
        return 0.0


def match(embedding, profiles: list[dict], *,
          trusted_threshold: Optional[float] = None,
          guess_threshold: Optional[float] = None,
          min_samples: Optional[int] = None) -> Optional[dict]:
    """Best cosine match of ``embedding`` against enrolled ``profiles`` (each a
    ``voice_roster`` profile dict). Returns the richest result it can:

      ``{"name", "score" (0-100), "similarity" (-1..1), "count", "trusted": bool}``

    or None when nothing clears the guess floor. ``trusted`` is True only above
    the trusted threshold *and* with enough enrolled samples — that is the line
    a caller must require before letting voice stand an intrusion alert down."""
    t_thr = TRUSTED_THRESHOLD if trusted_threshold is None else trusted_threshold
    g_thr = GUESS_THRESHOLD if guess_threshold is None else guess_threshold
    min_n = MIN_SAMPLES_FOR_TRUST if min_samples is None else min_samples
    if not embedding or not profiles:
        return None

    best = None
    for prof in profiles:
        centroid = prof.get("centroid")
        if not centroid or len(centroid) != len(embedding):
            continue
        sim = cosine(embedding, centroid)
        if best is None or sim > best[0]:
            best = (sim, prof)
    if best is None:
        return None

    sim, prof = best
    if sim < g_thr:
        return None
    count = int(prof.get("count", 0) or 0)
    trusted = bool(sim >= t_thr and count >= min_n)
    return {
        "name": prof.get("name"),
        "score": round(max(0.0, min(1.0, sim)) * 100.0, 1),
        "similarity": round(sim, 4),
        "count": count,
        "trusted": trusted,
    }


def identify(embedding) -> Optional[dict]:
    """Match ``embedding`` against every enrolled voiceprint, using the
    configured thresholds. Returns the match dict (see :func:`match`) or None.
    Reads the roster from disk — call off the event loop. Never raises."""
    try:
        from . import voice_roster
        return match(
            embedding, voice_roster.all_profiles(),
            trusted_threshold=float(_cfg("voice_local_trust_threshold", TRUSTED_THRESHOLD)),
            guess_threshold=float(_cfg("voice_local_guess_threshold", GUESS_THRESHOLD)),
            min_samples=int(_cfg("voice_local_min_samples", MIN_SAMPLES_FOR_TRUST)),
        )
    except Exception as exc:
        _LOGGER.debug("speaker identify failed: %s", exc)
        return None


def enroll(name: str, embedding) -> bool:
    """Fold a voice sample into ``name``'s enrolled profile (delegates to
    ``voice_roster``). Does file I/O — call off the event loop."""
    try:
        from . import voice_roster
        return voice_roster.enroll(name, embedding)
    except Exception as exc:
        _LOGGER.debug("speaker enroll failed: %s", exc)
        return False


# ── embedding backend (Step 2) ───────────────────────────────────────────────
# JARVIS does NOT force a speaker-embedding runtime on every install. The backend
# is an OPTIONAL, lazily-imported dependency (``onnxruntime``) plus a model file
# the maintainer supplies — so a default install carries no heavy wheel and the
# whole in-process path stays inert (``backend_available() → False``) until both
# are present. The model takes 16 kHz mono waveform and returns a speaker
# embedding; that is the raw-waveform contract most compact speaker models use.
# A feature-based model (fbank/x-vector) needs its own front-end — that front-end
# is the one piece a maintainer may have to adapt, and it lives behind this seam.
_TARGET_SR = 16000
_MIN_VOICED_SAMPLES = _TARGET_SR  # ~1.0s of audio; shorter clips ID poorly
_MAX_SAMPLES = _TARGET_SR * 30    # hard cap so a runaway stream can't blow memory

_session = None           # cached onnxruntime InferenceSession
_session_tried = False    # we attempt the load at most once per process
_input_name = None
_input_rank = 2           # how many dims the model's input tensor wants


def _model_path() -> str:
    """Where the speaker model lives: an explicit config override, else the
    conventional ``<config>/jarvis/models/speaker.onnx``."""
    override = (_cfg("voice_embedding_model_path", "") or "").strip()
    if override:
        return override
    try:
        from .paths import config_path_str
        return config_path_str("jarvis", "models", "speaker.onnx")
    except Exception:
        return ""


def _load_session():
    """Lazily load the ONNX speaker model once. Returns the session or None.
    Best-effort: a missing runtime or model just leaves the path inert."""
    global _session, _session_tried, _input_name, _input_rank
    if _session_tried:
        return _session
    _session_tried = True
    try:
        path = _model_path()
        import os
        if not path or not os.path.exists(path):
            _LOGGER.debug("speaker model not found at %s — in-process voice ID inert", path)
            return None
        import onnxruntime  # optional dependency, not in manifest requirements
        sess = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        _input_name = inp.name
        # Waveform models want either [batch, samples] or [batch, 1, samples].
        _input_rank = len([d for d in (inp.shape or [])]) or 2
        _session = sess
        _LOGGER.info("speaker-embedding model loaded (%s, input=%s rank=%d)",
                     path, _input_name, _input_rank)
    except Exception as exc:
        _LOGGER.warning("speaker-embedding backend unavailable: %s", exc)
        _session = None
    return _session


def backend_available() -> bool:
    """Whether a speaker-embedding model is loaded and usable. False unless the
    optional runtime + a model file are both present, so the in-process path is
    a strict no-op on a default install."""
    return _load_session() is not None


def _decode_wav(wav: bytes):
    """(samples:list[float] in [-1,1], sample_rate) from WAV bytes, mono-downmixed.
    stdlib only. Returns (None, 0) on anything it can't read."""
    try:
        import io
        import wave
        with wave.open(io.BytesIO(wav), "rb") as w:
            ch, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
        return _pcm_to_float(raw, width, ch), sr
    except Exception as exc:
        _LOGGER.debug("wav decode failed: %s", exc)
        return None, 0


def _pcm_to_float(raw: bytes, width: int, channels: int) -> Optional[list[float]]:
    """16-bit (or 8/32-bit) little-endian PCM → mono float samples in [-1, 1]."""
    import struct
    if width == 2:
        fmt, scale = "<%dh" % (len(raw) // 2), 32768.0
    elif width == 1:
        fmt, scale = "%dB" % len(raw), 128.0   # unsigned 8-bit, centered below
    elif width == 4:
        fmt, scale = "<%di" % (len(raw) // 4), 2147483648.0
    else:
        return None
    try:
        vals = struct.unpack(fmt, raw[: struct.calcsize(fmt)])
    except Exception:
        return None
    if width == 1:
        vals = [v - 128 for v in vals]
    mono = [v / scale for v in vals]
    if channels > 1:
        mono = [sum(mono[i:i + channels]) / channels
                for i in range(0, len(mono) - channels + 1, channels)]
    return mono


def _resample_to_target(samples: list[float], sr: int) -> list[float]:
    """Linear-interpolation resample to _TARGET_SR. Good enough for a speaker
    embedding and keeps us dependency-free (no scipy/numpy)."""
    if sr == _TARGET_SR or sr <= 0 or not samples:
        return samples
    ratio = _TARGET_SR / float(sr)
    n_out = int(len(samples) * ratio)
    out = []
    for i in range(n_out):
        src = i / ratio
        lo = int(src)
        hi = min(lo + 1, len(samples) - 1)
        frac = src - lo
        out.append(samples[lo] * (1.0 - frac) + samples[hi] * frac)
    return out


def _run_model(samples: list[float]) -> Optional[list[float]]:
    """Feed 16 kHz mono float samples to the model and L2-normalize the output.
    Imports numpy only if the backend is present (it ships with onnxruntime)."""
    sess = _load_session()
    if sess is None:
        return None
    try:
        import numpy as np
        arr = np.asarray(samples, dtype="float32")
        feed = arr.reshape(1, 1, -1) if _input_rank >= 3 else arr.reshape(1, -1)
        out = sess.run(None, {_input_name: feed})[0]
        vec = [float(x) for x in np.asarray(out).reshape(-1)]
        n = math.sqrt(sum(x * x for x in vec))
        return [x / n for x in vec] if n > 0 else None
    except Exception as exc:
        _LOGGER.debug("speaker model inference failed: %s", exc)
        return None


def embed_pcm(pcm: bytes, sample_rate: int) -> Optional[list[float]]:
    """Turn raw 16-bit mono PCM into a speaker embedding (L2-normalized), or None
    if the backend is absent or the clip is too short. Runs the model — call on
    the executor. Never raises."""
    samples = _pcm_to_float(pcm, 2, 1)
    if samples is None:
        return None
    return _embed_samples(samples, sample_rate)


def embed_wav(wav: bytes) -> Optional[list[float]]:
    """Speaker embedding from a WAV clip (what the Voices panel / an automation
    hands us). Decodes, mono-downmixes, resamples, embeds. Never raises."""
    samples, sr = _decode_wav(wav)
    if samples is None:
        return None
    return _embed_samples(samples, sr)


def _embed_samples(samples: list[float], sr: int) -> Optional[list[float]]:
    if not backend_available():
        _LOGGER.debug("speaker embedding backend not available")
        return None
    samples = _resample_to_target(samples, sr)
    if len(samples) < _MIN_VOICED_SAMPLES:
        _LOGGER.debug("utterance too short for speaker ID (%d samples)", len(samples))
        return None
    if len(samples) > _MAX_SAMPLES:
        samples = samples[:_MAX_SAMPLES]
    return _run_model(samples)
