"""Issue #404 — regressions from the #401 TTS work (shipped in 9.24.2).

1. `language` must NOT ride inside `tts.speak` options (HA rejects it there:
   "Invalid options found: ['language']") — it is a top-level service field.
2. `jarvis.speak` must follow JARVIS's auto-picked TTS engine, not a hardcoded
   `tts.piper` that may not be installed.
3. After an HA restart the empty in-memory delivery-id map must not strand an
   open delivery episode (kernel_present stuck True → no new announcements).
"""
import sys
import types

import pytest

from fakes import FakeHass


# ── #404.1: language never in options; exposed via voice_language ─────────────

@pytest.fixture
def tts(load, monkeypatch):
    monkeypatch.setitem(sys.modules, "aiohttp", types.ModuleType("aiohttp"))
    sys.modules.pop("jc.tts_helper", None)
    mod = load("tts_helper")
    yield mod
    sys.modules.pop("jc.tts_helper", None)


def test_voice_options_never_carries_language(tts):
    # Even with a language configured, options carries the voice only.
    rc = [{"tts_voice": "RyanNeural", "tts_language": "en-GB"}]
    opts = tts.voice_options(FakeHass(), "tts.home_assistant_cloud", rc)
    assert opts == {"voice": "RyanNeural"}
    assert "language" not in opts


def test_voice_language_non_piper_returns_lang(tts):
    rc = [{"tts_voice": "RyanNeural", "tts_language": "en-GB"}]
    assert tts.voice_language(FakeHass(), "tts.home_assistant_cloud", rc) == "en-GB"


def test_voice_language_none_for_piper(tts):
    # Piper encodes language in the voice name; a language hint can knock it onto
    # a default voice, so Piper never gets a language field.
    rc = [{"tts_voice": "x", "tts_language": "en-GB"}]
    assert tts.voice_language(FakeHass(), "tts.piper", rc) is None


def test_voice_language_none_when_no_voice(tts):
    assert tts.voice_language(FakeHass(), "tts.home_assistant_cloud", [{}]) is None
    assert tts.voice_language(FakeHass(), "", [{}]) is None


# ── #404.2: jarvis.speak follows the auto-picked engine, not tts.piper ────────

@pytest.fixture
def pa(load, monkeypatch):
    helpers = sys.modules.setdefault(
        "homeassistant.helpers", types.ModuleType("homeassistant.helpers"))
    for name, mod in (
        ("area_registry", types.ModuleType("homeassistant.helpers.area_registry")),
        ("config_validation", types.ModuleType("homeassistant.helpers.config_validation")),
        ("event", types.ModuleType("homeassistant.helpers.event")),
    ):
        monkeypatch.setitem(sys.modules, f"homeassistant.helpers.{name}", mod)
        monkeypatch.setattr(helpers, name, mod, raising=False)
    sys.modules["homeassistant.helpers.area_registry"].async_get = lambda hass: types.SimpleNamespace(
        async_get_area=lambda *a, **k: None, async_get_area_by_name=lambda *a, **k: None,
        async_list_areas=lambda *a, **k: [])
    cv = sys.modules["homeassistant.helpers.config_validation"]
    cv.string, cv.boolean = str, bool
    ev = sys.modules["homeassistant.helpers.event"]
    ev.async_call_later = lambda *a, **k: (lambda: None)
    ev.async_track_time_interval = lambda *a, **k: (lambda: None)
    if "homeassistant.exceptions" not in sys.modules:
        exc = types.ModuleType("homeassistant.exceptions")
        class HomeAssistantError(Exception): ...
        exc.HomeAssistantError = HomeAssistantError
        monkeypatch.setitem(sys.modules, "homeassistant.exceptions", exc)
    monkeypatch.setitem(sys.modules, "aiohttp", types.ModuleType("aiohttp"))
    sys.modules.pop("jc.proactive_audio", None)
    mod = load("proactive_audio")
    yield mod
    sys.modules.pop("jc.proactive_audio", None)


def test_speak_entity_auto_picks_available_engine(pa, load, monkeypatch):
    # No proactive_tts_entity override, tts_engine 'auto', and only a Cloud TTS
    # entity exists → jarvis.speak must resolve it, NOT the hardcoded tts.piper.
    const = load("const")
    h = FakeHass()
    h.data = {const.DOMAIN: {"e1": {"runtime_config": {}}}}
    h.states.set("tts.home_assistant_cloud", "idle")
    # jarvis_config.get returns the default for every key in this sandbox.
    assert pa._resolve_tts_entity(h) == "tts.home_assistant_cloud"


def test_speak_entity_honours_explicit_override(pa, load):
    const = load("const")
    h = FakeHass()
    h.data = {const.DOMAIN: {"e1": {"runtime_config": {
        "proactive_tts_entity": "tts.elevenlabs"}}}}
    h.states.set("tts.home_assistant_cloud", "idle")
    assert pa._resolve_tts_entity(h) == "tts.elevenlabs"


# ── #404.3: delivery episode recovers after a restart (empty id map) ──────────

@pytest.fixture
def ds(load, tmp_path, monkeypatch):
    mod = load("delivery_situation")
    S = load("kernel.situation")
    mgr = S.SituationManager(str(tmp_path / "situations.db"))
    monkeypatch.setattr(mod, "_mgr", mgr)
    monkeypatch.setattr(mod, "_delivery_situation_ids", {})
    monkeypatch.setattr(mod, "_last_parity", None)
    monkeypatch.setattr(mod, "_last_presence_parity", None)
    return mod, S, mgr


def test_removed_resolves_stale_episode_after_restart(ds, monkeypatch):
    mod, S, mgr = ds
    h = FakeHass()
    mod.mirror_delivery_sync(h, "camera.porch", "delivered", 1)
    assert mod.kernel_package_present_sync(h, "camera.porch") is True
    # Simulate an HA restart: the in-memory id map is wiped, but the OPEN episode
    # survives in situations.db.
    monkeypatch.setattr(mod, "_delivery_situation_ids", {})
    mod.mirror_delivery_sync(h, "camera.porch", "removed", 0)
    # Recovered by subject and resolved — so a new delivery can announce again.
    assert mod.kernel_package_present_sync(h, "camera.porch") is False


def test_delivered_after_restart_does_not_duplicate(ds, monkeypatch):
    mod, S, mgr = ds
    h = FakeHass()
    mod.mirror_delivery_sync(h, "camera.porch", "delivered", 1)
    monkeypatch.setattr(mod, "_delivery_situation_ids", {})
    # A second 'delivered' after restart should recover the open episode, not open
    # a duplicate — exactly one open episode for the subject.
    mod.mirror_delivery_sync(h, "camera.porch", "delivered", 1)
    opens = [s for s in mgr.open_situations("delivery") if s.subject == "camera.porch"]
    assert len(opens) == 1
