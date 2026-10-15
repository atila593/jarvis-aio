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


# ── embedding backend seam (wired in Step 2) ─────────────────────────────────

def backend_available() -> bool:
    """Whether a speaker-embedding model is loaded and usable. Always False in
    Step 1 — no model/backend ships yet, so the identify path stays inert until
    the capture+model work (Step 2) lands."""
    return False


def embed_pcm(pcm: bytes, sample_rate: int) -> Optional[list[float]]:
    """Turn 16-bit mono PCM into a speaker embedding. No backend in Step 1, so
    this returns None (and the whole in-process path is a no-op). Step 2 fills
    this in with a small ONNX speaker model run on the executor. Never raises."""
    if not backend_available():
        _LOGGER.debug("speaker embedding backend not available yet (pending Step 2)")
        return None
    return None  # pragma: no cover - backend arrives in Step 2
