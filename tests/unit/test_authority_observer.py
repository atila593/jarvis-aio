"""Step B — the log-only authority parity state-change observer.

The observer turns real device transitions on the enforce-allowlisted domains
(lock / alarm_control_panel / cover) into engine-vs-gate parity samples, feeding
the SAME tally the agent path feeds. These tests cover the pure transition→
capability mapping, the record path (into the real tally), the domain / no-op /
availability filtering, the kill-switch, and registration.
"""
import sys
import types

import pytest

from fakes import FakeHass


@pytest.fixture
def obs(load):
    return load("authority_observer")


# ── pure transition → capability mapping ──────────────────────────────────────

def test_lock_transitions(obs):
    assert obs._service_for_transition("lock", "locked", "unlocked") == "unlock"
    assert obs._service_for_transition("lock", "unlocked", "locked") == "lock"
    assert obs._service_for_transition("lock", "locked", "open") == "open"


def test_cover_transitions(obs):
    assert obs._service_for_transition("cover", "closed", "open") == "open_cover"
    assert obs._service_for_transition("cover", "closed", "opening") == "open_cover"
    assert obs._service_for_transition("cover", "open", "closed") == "close_cover"
    assert obs._service_for_transition("cover", "open", "closing") == "close_cover"


def test_alarm_transitions(obs):
    assert obs._service_for_transition(
        "alarm_control_panel", "disarmed", "armed_away") == "alarm_arm_away"
    assert obs._service_for_transition(
        "alarm_control_panel", "disarmed", "armed_home") == "alarm_arm_home"
    assert obs._service_for_transition(
        "alarm_control_panel", "disarmed", "armed_night") == "alarm_arm_night"
    assert obs._service_for_transition(
        "alarm_control_panel", "armed_away", "disarmed") == "alarm_disarm"


def test_noop_and_unavailable_transitions_ignored(obs):
    # same state → not an action
    assert obs._service_for_transition("lock", "locked", "locked") is None
    # availability churn → not an action
    assert obs._service_for_transition("lock", "unavailable", "locked") is None
    assert obs._service_for_transition("lock", "locked", "unknown") is None
    assert obs._service_for_transition("cover", None, "open") is None
    # unmapped domain / state
    assert obs._service_for_transition("climate", "off", "heat") is None
    assert obs._service_for_transition("lock", "locked", "jammed") is None


# ── record path (feeds the real authority_bridge tally) ───────────────────────

def _real_policy(load, monkeypatch, *, protected: bool):
    """Load the real policy module and pin requires_confirmation to a known
    verdict so the engine-vs-gate comparison is deterministic."""
    policy = load("policy")
    monkeypatch.setattr(policy, "requires_confirmation",
                        lambda hass, d, s, e="": protected)
    return policy


def test_record_transition_records_a_sample(obs, load, monkeypatch):
    ab = load("authority_bridge")
    _real_policy(load, monkeypatch, protected=True)   # gate would CONFIRM
    hass = FakeHass()
    assert obs._record_transition(hass, "lock.front", "locked", "unlocked") is True
    s = ab.parity_summary(hass)
    assert (s["agree"] + s["disagree"]) == 1
    assert "lock.unlock" in s["by_capability"]


def test_record_transition_matches_agent_path_semantics(obs, load, monkeypatch):
    # The observer records with the SAME minimal context the agent path uses (no
    # identity). For a security capability with no identified requester the engine
    # DENYs, so it disagrees with the gate's CONFIRM — exactly as the existing
    # agent-path recording does (test_authority_bridge: mismatch for a no-identity
    # security action). The observer must reproduce that, not diverge from it, so
    # its samples stay comparable to the ones already in the tally.
    ab = load("authority_bridge")
    _real_policy(load, monkeypatch, protected=True)   # gate would CONFIRM
    hass = FakeHass()
    obs._record_transition(hass, "lock.front", "locked", "unlocked")
    s = ab.parity_summary(hass)
    assert s["disagree"] == 1 and s["agree"] == 0
    assert s["recent_mismatches"][0]["capability"] == "lock.unlock"


def test_record_transition_skips_nonwatched_domain(obs, load, monkeypatch):
    ab = load("authority_bridge")
    _real_policy(load, monkeypatch, protected=False)
    hass = FakeHass()
    assert obs._record_transition(hass, "light.kitchen", "off", "on") is False
    assert ab.parity_summary(hass).get("agree", 0) == 0
    assert ab.parity_summary(hass).get("disagree", 0) == 0


def test_record_transition_skips_noop(obs, load, monkeypatch):
    ab = load("authority_bridge")
    _real_policy(load, monkeypatch, protected=True)
    hass = FakeHass()
    assert obs._record_transition(hass, "cover.garage", "open", "open") is False
    assert ab.parity_summary(hass).get("agree", 0) == 0


# ── kill-switch ───────────────────────────────────────────────────────────────

def test_enabled_default_true(obs):
    assert obs._enabled(FakeHass()) is True


def test_module_killswitch_disables(obs, monkeypatch):
    monkeypatch.setattr(obs, "AUTHORITY_OBSERVER_ENABLED", False)
    assert obs._enabled(FakeHass()) is False
    # register short-circuits to None when disabled
    assert obs.register(FakeHass()) is None


# ── registration ──────────────────────────────────────────────────────────────

def test_register_wires_a_listener_for_watched_entities(obs, monkeypatch):
    # Stub the HA state-change tracker so registration can wire without a real bus.
    captured = {}

    def _track(hass, entity_ids, cb):
        captured["entity_ids"] = list(entity_ids)
        captured["cb"] = cb
        return lambda: captured.setdefault("unsubbed", True)

    ev = types.ModuleType("homeassistant.helpers.event")
    ev.async_track_state_change_event = _track
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.event", ev)

    hass = FakeHass()
    hass.states.set("lock.front", "locked")
    hass.states.set("cover.garage", "closed")
    hass.states.set("alarm_control_panel.home", "disarmed")
    hass.states.set("light.kitchen", "on")      # not watched
    hass.states.set("sensor.temp", "20")         # not watched

    unsub = obs.register(hass)
    assert callable(unsub)
    watched = captured["entity_ids"]
    assert set(watched) == {"lock.front", "cover.garage", "alarm_control_panel.home"}
    assert "light.kitchen" not in watched and "sensor.temp" not in watched
    unsub()
    assert captured.get("unsubbed") is True


def test_register_none_when_no_watched_entities(obs, monkeypatch):
    ev = types.ModuleType("homeassistant.helpers.event")
    ev.async_track_state_change_event = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not be called when nothing to watch"))
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.event", ev)
    hass = FakeHass()
    hass.states.set("light.kitchen", "on")
    assert obs.register(hass) is None
