"""
JARVIS — enrolled voiceprint store (vocal biometrics, Path B Step 1; issue #397).

The voice analog of the resident reference-photo store in ``recognition.py``, but
privacy-first: it stores **speaker embeddings, never raw audio**. Each enrolled
person gets one running-mean *centroid* of L2-normalized embeddings plus a sample
count, at ``<config>/jarvis/voices_ref/<norm>.json``. New samples update the
centroid incrementally, so profiles improve over time without ever keeping a
recording.

This module is pure storage — it does not capture audio, run a model, or decide
identity (that is ``speaker_id.py`` in later steps). stdlib only, atomic writes,
a reentrant lock, and it never raises to the caller. Inert until the capture
path (Step 2) and the identity wiring (Step 3) are built.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from typing import Optional

from .paths import config_path_str

_LOGGER = logging.getLogger(__name__)

VOICE_REF_DIR = config_path_str("jarvis", "voices_ref")

_lock = threading.RLock()

# A voiceprint shorter/longer than this is almost certainly not an embedding from
# a real speaker model — reject it rather than corrupt a profile.
_MIN_DIM = 16
_MAX_DIM = 4096


def _norm(name: str) -> str:
    """Filesystem-safe key for an enrolled name — the on-disk filename stem.
    Starts from identity.normalize (consistent with the roster), then hard-
    restricts to ``[a-z0-9_-]`` and caps length so a name can never yield a path
    separator, ``..``, or anything that escapes the voices_ref directory.
    Returns ``""`` for a name with no usable characters."""
    try:
        from .identity import normalize
        base = normalize(name)
    except Exception:
        base = "_".join((name or "").strip().lower().split())
    return re.sub(r"[^a-z0-9_-]", "", (base or "").lower())[:80]


def _clean_vector(embedding) -> Optional[list[float]]:
    """Coerce ``embedding`` to a list of finite floats, or None if it isn't a
    usable vector (wrong type, empty, out-of-range dim, non-finite values)."""
    if not isinstance(embedding, (list, tuple)):
        return None
    if not (_MIN_DIM <= len(embedding) <= _MAX_DIM):
        return None
    out: list[float] = []
    for v in embedding:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        out.append(f)
    return out


def _l2norm(vec: list[float]) -> Optional[list[float]]:
    """Unit-length version of ``vec`` (what cosine matching wants), or None if
    the vector is all-zero."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return None
    return [x / norm for x in vec]


def _path(key: str) -> str:
    return os.path.join(VOICE_REF_DIR, f"{key}.json")


def _read(key: str) -> Optional[dict]:
    try:
        p = _path(key)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        centroid = _clean_vector(data.get("centroid"))
        count = int(data.get("count", 0) or 0)
        if centroid is None or count <= 0:
            return None
        return {
            "name": str(data.get("name") or key),
            "centroid": centroid,
            "count": count,
            "dim": len(centroid),
            "updated": float(data.get("updated", 0) or 0),
        }
    except Exception as exc:
        _LOGGER.debug("voice profile read failed for %s: %s", key, exc)
        return None


def _write(key: str, profile: dict) -> bool:
    try:
        os.makedirs(VOICE_REF_DIR, exist_ok=True)
        tmp = _path(key) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profile, f)
        os.replace(tmp, _path(key))
        return True
    except Exception as exc:
        _LOGGER.debug("voice profile write failed for %s: %s", key, exc)
        return False


def enroll(name: str, embedding) -> bool:
    """Fold one voice sample into ``name``'s profile. The embedding is
    L2-normalized, then averaged into the stored centroid (running mean over the
    sample count). A sample whose dimensionality disagrees with the existing
    profile is rejected (it's from a different model). Does file I/O — call off
    the event loop. Returns True on success. Never raises."""
    key = _norm(name)
    if not key:
        return False
    vec = _clean_vector(embedding)
    if vec is None:
        return False
    unit = _l2norm(vec)
    if unit is None:
        return False
    with _lock:
        existing = _read(key)
        if existing is None:
            centroid, count = unit, 1
        else:
            if existing["dim"] != len(unit):
                _LOGGER.warning(
                    "voice enroll for %s rejected: embedding dim %d != profile dim %d",
                    name, len(unit), existing["dim"])
                return False
            n = existing["count"]
            centroid = [(c * n + u) / (n + 1) for c, u in zip(existing["centroid"], unit)]
            count = n + 1
        return _write(key, {
            "name": str(name).strip() or key,
            "centroid": centroid,
            "count": count,
            "updated": time.time(),
        })


def get(name: str) -> Optional[dict]:
    """A person's profile ``{name, centroid, count, dim, updated}`` or None."""
    key = _norm(name)
    if not key:
        return None
    with _lock:
        return _read(key)


def has(name: str) -> bool:
    return get(name) is not None


def sample_count(name: str) -> int:
    prof = get(name)
    return prof["count"] if prof else 0


def all_profiles() -> list[dict]:
    """Every enrolled profile, for the matcher. Never raises."""
    out: list[dict] = []
    try:
        if not os.path.isdir(VOICE_REF_DIR):
            return out
        with _lock:
            for fn in sorted(os.listdir(VOICE_REF_DIR)):
                if not fn.endswith(".json"):
                    continue
                prof = _read(fn[:-5])
                if prof:
                    out.append(prof)
    except Exception as exc:
        _LOGGER.debug("voice profile listing failed: %s", exc)
    return out


def profiles() -> list[str]:
    """Display names that currently have an enrolled voiceprint, sorted."""
    return sorted((p["name"] for p in all_profiles()), key=str.casefold)


def remove(name: str) -> bool:
    """Delete ``name``'s voiceprint. Returns True if one was removed. Never raises."""
    key = _norm(name)
    if not key:
        return False
    with _lock:
        try:
            p = _path(key)
            if not os.path.exists(p):
                return False
            os.remove(p)
            return True
        except Exception as exc:
            _LOGGER.debug("voice profile remove failed for %s: %s", name, exc)
            return False
