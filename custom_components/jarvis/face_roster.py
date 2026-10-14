"""
JARVIS — household face roster (resident whitelist).

A small, JARVIS-native list of people who are household *residents*. JARVIS does
not run its own face engine — identities come from the vision backend (Frigate
face recognition, DoubleTake, CompreFace; see ``recognition.py``). This module
layers a user-curated "who is a resident" flag on top of those recognized names,
so JARVIS can tell a known resident from an unknown face and use that to refine
intrusion escalation and presence (issue #140).

Stored as a plain JSON list at ``<config>/jarvis/face_roster.json`` and matched
by the same normalized-name key ``identity.normalize`` uses, so "Sam", "sam",
and "  Sam  " are the same resident. Dependency-light stdlib only, atomic
writes, never raises to the caller.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

from .paths import config_path_str

_LOGGER = logging.getLogger(__name__)

ROSTER_PATH = config_path_str("jarvis", "face_roster.json")

_lock = threading.RLock()
# {normalized_name: display_name}
_roster: dict[str, str] = {}
_loaded = False


def _normalize(name: str) -> str:
    try:
        from .identity import normalize
        return normalize(name)
    except Exception:
        return "_".join((name or "").strip().lower().split())


def _load() -> None:
    global _loaded, _roster
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        data = {}
        try:
            if os.path.exists(ROSTER_PATH):
                with open(ROSTER_PATH) as f:
                    raw = json.load(f)
                # Accept a bare list of names (preferred) or a {norm: display} map.
                if isinstance(raw, list):
                    for n in raw:
                        key = _normalize(str(n))
                        if key:
                            data[key] = str(n).strip()
                elif isinstance(raw, dict):
                    for k, v in raw.items():
                        key = _normalize(str(k))
                        if key:
                            data[key] = str(v or k).strip()
        except Exception as exc:
            _LOGGER.warning("face roster load failed: %s", exc)
            data = {}
        _roster = data
        _loaded = True


def _save() -> None:
    # Called with _lock held. Persists the display names as a sorted list.
    try:
        os.makedirs(os.path.dirname(ROSTER_PATH), exist_ok=True)
        tmp = ROSTER_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sorted(_roster.values(), key=str.casefold), f)
        os.replace(tmp, ROSTER_PATH)
    except Exception as exc:
        _LOGGER.debug("face roster persist failed: %s", exc)


def residents() -> list[str]:
    """Household resident display names, sorted (case-insensitive). Never raises."""
    _load()
    return sorted(_roster.values(), key=str.casefold)


def is_resident(name: str) -> bool:
    """Whether ``name`` (any casing/spacing) is flagged a household resident."""
    if not name:
        return False
    _load()
    return _normalize(name) in _roster


# Names the vision backend uses for "no identity" — these are strangers, not
# guests. Kept in step with ``recognition._is_unknown``.
_UNKNOWN_NAMES = frozenset({"", "unknown", "unknown person", "unknown_face"})


def is_guest(name: str) -> bool:
    """Whether ``name`` is a recognized *guest* — a person the vision backend
    identified by name who is NOT a flagged household resident (a friend,
    relative or other visitor). Blank/unknown names are strangers, not guests,
    so they return False (issue #388). Never raises."""
    if (name or "").strip().lower() in _UNKNOWN_NAMES:
        return False
    return not is_resident(name)


def relationship(name: str) -> str:
    """Classify a recognized person as ``"resident"``, ``"guest"`` or
    ``"unknown"``. This is the single source of truth for the
    household/guest/stranger distinction surfaced to Home Assistant automations
    (the ``jarvis_face_recognized`` event) and the Faces panel (issue #388):
    residents are flagged household members, guests are recognized people who
    aren't, and unknown is an unidentified face. Never raises."""
    if (name or "").strip().lower() in _UNKNOWN_NAMES:
        return "unknown"
    return "resident" if is_resident(name) else "guest"


def add_resident(name: str) -> bool:
    """Flag ``name`` a resident. Returns True if it was newly added. Never raises."""
    key = _normalize(name)
    if not key:
        return False
    _load()
    with _lock:
        if key in _roster:
            # Keep the latest display spelling, but report "not new".
            _roster[key] = str(name).strip()
            _save()
            return False
        _roster[key] = str(name).strip()
        _save()
        return True


def remove_resident(name: str) -> bool:
    """Un-flag ``name``. Returns True if it was present. Never raises."""
    key = _normalize(name)
    if not key:
        return False
    _load()
    with _lock:
        if key not in _roster:
            return False
        _roster.pop(key, None)
        _save()
        return True
