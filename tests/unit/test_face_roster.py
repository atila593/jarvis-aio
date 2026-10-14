"""Tests for the household resident whitelist (#140) — a JARVIS-native
'who is a resident' flag layered on backend face recognition. Matching is
name-normalized (case/space-insensitive) and the list persists to JSON."""
import json

import pytest


@pytest.fixture
def fr(load, tmp_path, monkeypatch):
    mod = load("face_roster")
    monkeypatch.setattr(mod, "ROSTER_PATH", str(tmp_path / "face_roster.json"))
    mod._loaded = False
    mod._roster = {}
    return mod


def test_add_and_is_resident_normalizes(fr):
    assert fr.add_resident("Sam") is True
    assert fr.add_resident("Sam") is False          # already present → not new
    assert fr.is_resident("sam") is True            # case-insensitive
    assert fr.is_resident("  SAM  ") is True         # whitespace-insensitive
    assert fr.is_resident("Quentin") is False
    assert fr.residents() == ["Sam"]


def test_remove(fr):
    fr.add_resident("Eliana")
    assert fr.remove_resident("eliana") is True     # normalized match removes it
    assert fr.remove_resident("eliana") is False    # already gone
    assert fr.residents() == []


def test_blank_names_ignored(fr):
    assert fr.add_resident("") is False
    assert fr.add_resident("   ") is False
    assert fr.residents() == []


def test_persists_across_reload(fr):
    fr.add_resident("Sam")
    fr.add_resident("Lee")
    # Simulate a fresh process: drop in-memory state, reload from disk.
    fr._loaded = False
    fr._roster = {}
    assert sorted(fr.residents()) == ["Lee", "Sam"]


def test_on_disk_format_is_a_plain_list(fr):
    fr.add_resident("Sam")
    with open(fr.ROSTER_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list) and "Sam" in data


def test_tolerates_legacy_dict_file(fr):
    # An older {normalized: display} map should still load.
    with open(fr.ROSTER_PATH, "w") as f:
        json.dump({"sam_smith": "Sam Smith"}, f)
    fr._loaded = False
    fr._roster = {}
    assert fr.is_resident("Sam Smith") is True
    assert fr.residents() == ["Sam Smith"]


# ── resident / guest / unknown classification (issue #388) ──────────────────────

def test_relationship_resident_guest_unknown(fr):
    fr.add_resident("Sam")
    # A flagged resident.
    assert fr.relationship("Sam") == "resident"
    assert fr.relationship("  sam ") == "resident"      # normalized match
    # A recognized but unlisted person is a guest.
    assert fr.relationship("Quentin") == "guest"
    # Backend "no identity" markers are strangers, not guests.
    for n in ("Unknown", "unknown person", "unknown_face", "", "   "):
        assert fr.relationship(n) == "unknown", n


def test_is_guest(fr):
    fr.add_resident("Sam")
    assert fr.is_guest("Quentin") is True               # recognized, not a resident
    assert fr.is_guest("sam") is False                  # a resident is not a guest
    assert fr.is_guest("Unknown") is False              # a stranger is not a guest
    assert fr.is_guest("") is False


def test_relationship_flips_when_residency_toggled(fr):
    # The whole point of #388: managing status without touching face recognition.
    assert fr.relationship("Quentin") == "guest"
    fr.add_resident("Quentin")
    assert fr.relationship("Quentin") == "resident"
    fr.remove_resident("Quentin")
    assert fr.relationship("Quentin") == "guest"        # back to a guest, still recognized
