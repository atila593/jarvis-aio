"""Issue #400 — the 10-point omnibus report (shipped in 9.24.0).

Covers the pure/seam-level logic for each fix so the behaviour-preserving
guarantees (new options default to the prior behaviour; the bug fixes do the
right thing out of the box) are testable without a live Home Assistant.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from fakes import FakeHass

# bootstrap.py imports aiohttp + the HA aiohttp client helper at module scope;
# stub them before any load("bootstrap") (mirrors test_bootstrap.py).
if "aiohttp" not in sys.modules:
    _aiohttp = types.ModuleType("aiohttp")
    _aiohttp.ClientTimeout = lambda **kw: None
    _aiohttp.ClientSession = object
    sys.modules["aiohttp"] = _aiohttp
if "homeassistant.helpers.aiohttp_client" not in sys.modules:
    _ac = types.ModuleType("homeassistant.helpers.aiohttp_client")
    _ac.async_get_clientsession = lambda hass: None
    sys.modules["homeassistant.helpers.aiohttp_client"] = _ac


# ── #8: honorific output tidier ───────────────────────────────────────────────

@pytest.fixture
def tidy(load):
    return load("text_tidy").tidy_address


@pytest.mark.parametrize("raw,expected", [
    (", the front door is open.", "The front door is open."),
    ("Done, .", "Done."),
    ("Worth mentioning,  — the screen is on", "Worth mentioning — the screen is on"),
    ("turned off , please", "turned off, please"),
    ("Technical difficulties, . Even I have them.", "Technical difficulties. Even I have them."),
    ("Activating the kettle now, .", "Activating the kettle now."),
])
def test_tidy_removes_blank_honorific_artifacts(tidy, raw, expected):
    assert tidy(raw) == expected


@pytest.mark.parametrize("good", [
    "Done, Sir.",
    "Good evening, Captain. Two new messages.",
    "Good evening, Sir. All quiet.",
    "The hallway light is on.",
    "",
])
def test_tidy_is_noop_on_wellformed_text(tidy, good):
    # A non-empty honorific (or ordinary prose) must pass through untouched.
    assert tidy(good) == good


def test_tidy_handles_none(tidy):
    assert tidy(None) is None


# ── #8: honorific persistence (blank no longer snaps back to "sir") ───────────

@pytest.fixture
def real_ws():
    """conftest installs a tiny `jc.websocket` stub (only jarvis_log), so load the
    real module here (stub websocket_api's decorators) and restore the stub on
    teardown so later tests still see the lightweight one."""
    import importlib.util
    import pathlib
    comp = pathlib.Path(__file__).resolve().parents[2] / "custom_components" / "jarvis"
    prev = sys.modules.get("jc.websocket")
    ws_api = types.ModuleType("homeassistant.components.websocket_api")
    ws_api.websocket_command = lambda schema: (lambda func: func)
    ws_api.async_response = lambda func: func
    ws_api.require_admin = lambda func: func
    ws_api.async_register_command = lambda hass, func: None
    sys.modules["homeassistant.components.websocket_api"] = ws_api
    import homeassistant.components as ha_components
    ha_components.websocket_api = ws_api
    spec = importlib.util.spec_from_file_location("jc.websocket", comp / "websocket.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jc.websocket"] = mod
    spec.loader.exec_module(mod)
    yield mod
    if prev is not None:
        sys.modules["jc.websocket"] = prev
    else:
        sys.modules.pop("jc.websocket", None)


def test_honorific_payload_preserves_blank(real_ws):
    # None (truly unset) → default; "" (explicit blank) → preserved.
    assert real_ws._honorific_payload(None) == "sir"
    assert real_ws._honorific_payload("") == ""
    assert real_ws._honorific_payload("Captain") == "Captain"


def test_review_enabled_is_panel_writable(real_ws):
    assert "review_enabled" in real_ws.PANEL_WRITABLE_KEYS
    # the new #400 keys are writable too
    assert "quiet_hours_push_critical_only" in real_ws.PANEL_WRITABLE_KEYS
    assert "presence_only_entities" in real_ws.PANEL_WRITABLE_KEYS
    assert "proactive_questions_interactive" in real_ws.PANEL_WRITABLE_KEYS
    assert "arrival_greeting_enabled" in real_ws.PANEL_WRITABLE_KEYS


# ── #4: presence-only exclusion class ─────────────────────────────────────────

def test_presence_only_entity_detected(load):
    ef, const = load("entity_filter"), load("const")
    h = FakeHass()
    h.data = {const.DOMAIN: {"e1": {"runtime_config": {
        "presence_only_entities": ["binary_sensor.hall_motion"]}}}}
    assert ef.is_presence_only(h, "binary_sensor.hall_motion") is True
    assert ef.is_presence_only(h, "binary_sensor.kitchen_motion") is False


def test_presence_only_default_empty(load):
    ef, const = load("entity_filter"), load("const")
    h = FakeHass()
    h.data = {const.DOMAIN: {"e1": {"runtime_config": {}}}}
    assert ef.is_presence_only(h, "binary_sensor.hall_motion") is False
    assert ef.is_presence_only(h, "") is False


def test_presence_only_does_not_exclude_from_awareness(load):
    """A presence-only entity stays VISIBLE to presence (not is_excluded)."""
    ef, const = load("entity_filter"), load("const")
    h = FakeHass()
    h.data = {const.DOMAIN: {"e1": {"runtime_config": {
        "presence_only_entities": ["binary_sensor.hall_motion"]}}}}
    assert ef.is_excluded(h, "binary_sensor.hall_motion") is False


# ── #5: TTS voice options on every engine ─────────────────────────────────────

@pytest.fixture
def tts(load, monkeypatch):
    monkeypatch.setitem(sys.modules, "aiohttp", types.ModuleType("aiohttp"))
    sys.modules.pop("jc.tts_helper", None)
    mod = load("tts_helper")
    yield mod
    sys.modules.pop("jc.tts_helper", None)


def test_voice_options_piper_uses_jarvis_voice(tts):
    opts = tts.voice_options(FakeHass(), "tts.piper", [{"voice_quality": "high"}])
    assert opts == {"voice": "en_GB-jarvis-high"}


def test_voice_options_non_piper_uses_config_voice(tts):
    opts = tts.voice_options(
        FakeHass(), "tts.google_cloud",
        [{"tts_voice": "en-GB-Standard-B", "tts_language": "en-GB"}])
    assert opts.get("voice") == "en-GB-Standard-B"
    assert opts.get("language") == "en-GB"   # language allowed for non-Piper


def test_voice_options_piper_never_sends_language(tts):
    # A language hint can push a Piper build onto a default voice, so it is
    # suppressed for Piper even if configured.
    opts = tts.voice_options(
        FakeHass(), "tts.piper",
        [{"tts_use_ha_voice": True, "tts_voice": "x", "tts_language": "en-GB"}])
    assert "language" not in opts


def test_voice_options_empty_when_nothing_known(tts):
    assert tts.voice_options(FakeHass(), "tts.google_cloud", [{}]) == {}
    assert tts.voice_options(FakeHass(), "", [{}]) == {}


# ── #6: proactive questions interactive routing ───────────────────────────────

@pytest.fixture
def pa(load, monkeypatch):
    # proactive_audio binds several HA helpers at *import* scope (e.g.
    # `area_registry as ar`), so a cached module would pin whatever stub the
    # first loader installed. Load it fresh here AND drop it again on teardown,
    # so a later test (test_proactive_area_resolution) re-imports against its own
    # registry stub instead of ours.
    sys.modules.pop("jc.proactive_audio", None)
    helpers = sys.modules.setdefault(
        "homeassistant.helpers", types.ModuleType("homeassistant.helpers"))
    ar = types.ModuleType("homeassistant.helpers.area_registry")
    ar.async_get = lambda hass: SimpleNamespace(
        async_get_area=lambda *_a, **_k: None,
        async_get_area_by_name=lambda *_a, **_k: None,
        async_list_areas=lambda *_a, **_k: [])
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.area_registry", ar)
    monkeypatch.setattr(helpers, "area_registry", ar, raising=False)
    cv = types.ModuleType("homeassistant.helpers.config_validation")
    cv.string, cv.boolean = str, bool
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.config_validation", cv)
    monkeypatch.setattr(helpers, "config_validation", cv, raising=False)
    ev = types.ModuleType("homeassistant.helpers.event")
    ev.async_call_later = lambda *a, **k: (lambda: None)
    ev.async_track_time_interval = lambda *a, **k: (lambda: None)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.event", ev)
    monkeypatch.setattr(helpers, "event", ev, raising=False)
    if "homeassistant.exceptions" not in sys.modules:
        exc = types.ModuleType("homeassistant.exceptions")
        class HomeAssistantError(Exception): ...
        exc.HomeAssistantError = HomeAssistantError
        monkeypatch.setitem(sys.modules, "homeassistant.exceptions", exc)
    mod = load("proactive_audio")
    yield mod
    sys.modules.pop("jc.proactive_audio", None)


def test_is_question_detects_trailing_qmark(pa):
    assert pa._is_question("Shall I turn the tablet screen off?") is True
    assert pa._is_question('Close the garage?"') is True       # trailing quote
    assert pa._is_question("The garage is closed.") is False
    assert pa._is_question("") is False


# ── #6: satellite resolution for a target speaker ─────────────────────────────

def test_satellite_for_speaker_passthrough(load):
    routing = load("audio_routing")
    h = FakeHass()
    assert routing.satellite_for_speaker(h, "assist_satellite.office") \
        == "assist_satellite.office"


def test_satellite_for_speaker_none_when_no_satellite(load):
    routing = load("audio_routing")
    h = FakeHass()
    assert routing.satellite_for_speaker(h, "media_player.kitchen") is None
    assert routing.satellite_for_speaker(h, "") is None


# ── #3: arrival greeting gating ───────────────────────────────────────────────

@pytest.fixture
def briefing(load):
    return load("proactive_briefing")


def _states(away_minutes):
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    old = SimpleNamespace(state="not_home",
                          last_changed=now - dt.timedelta(minutes=away_minutes))
    new = SimpleNamespace(state="home", last_changed=now)
    return old, new


def _patch_quiet(load, monkeypatch, value):
    # briefing does `from . import sleep_detection` inside the function, so patch
    # the resolved module, not a briefing attribute.
    sd = load("sleep_detection")
    monkeypatch.setattr(sd, "_in_quiet_hours", lambda *a, **k: value)


def test_arrival_greeting_off_switch(briefing, load, monkeypatch):
    _patch_quiet(load, monkeypatch, False)
    old, new = _states(30)
    assert briefing._arrival_greeting_allowed(
        FakeHass(), {"arrival_greeting_enabled": False}, old, new) is False


def test_arrival_greeting_requires_min_away(briefing, load, monkeypatch):
    _patch_quiet(load, monkeypatch, False)
    old, new = _states(2)   # only 2 min away — a brief Wi-Fi dropout
    assert briefing._arrival_greeting_allowed(FakeHass(), {}, old, new) is False
    old, new = _states(30)  # genuinely away
    assert briefing._arrival_greeting_allowed(FakeHass(), {}, old, new) is True


def test_arrival_greeting_respects_quiet_hours(briefing, load, monkeypatch):
    _patch_quiet(load, monkeypatch, True)
    old, new = _states(30)
    assert briefing._arrival_greeting_allowed(FakeHass(), {}, old, new) is False


# ── #7 / #9: source-level guards (live Supervisor/pipeline can't run here) ─────

def test_pipeline_agent_respects_reassignment(load):
    """#7: async_ensure_pipeline_agent must leave a deliberately-reassigned agent
    alone — only claim None/empty/HA-default."""
    import inspect
    src = inspect.getsource(load("bootstrap").async_ensure_pipeline_agent)
    assert "conversation.home_assistant" in src
    assert "continue" in src


def test_bootstrap_remembers_removed_addons(load):
    """#9: the Phase-1 loop keys off a persisted managed_addons set and skips an
    add-on the owner removed."""
    import inspect
    src = inspect.getsource(load("bootstrap").async_run_bootstrap)
    assert "managed_addons" in src
    assert "not reinstalling" in src
