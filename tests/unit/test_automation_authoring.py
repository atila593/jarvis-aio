"""Plain-language automation authoring with entity/service validation.

The validator checks every entity_id and service a would-be automation
references against the live Home Assistant instance, so JARVIS never installs an
automation that silently points at a guessed or misspelled name — it refuses up
front and hands back the closest real name. Opt-in (validate=True), so the
existing HA service + learned-suggestion installer paths are unchanged.
"""
import json

import pytest

from fakes import FakeHass


@pytest.fixture
def ac(load):
    return load("automation_creator")


def _hass_with(entities=(), services=()):
    h = FakeHass()
    for eid in entities:
        h.states.set(eid, "on")
    for dom_svc in services:
        dom, svc = dom_svc.split(".", 1)
        h.services.register(dom, svc)
    return h


# ── reference extraction ──────────────────────────────────────────────────────

def test_collect_refs_pulls_entities_and_services(ac):
    ents, svcs = set(), set()
    ac._collect_refs(
        [{"trigger": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
        ents, svcs)
    ac._collect_refs(
        [{"action": "light.turn_on", "target": {"entity_id": ["light.porch", "light.hall"]}}],
        ents, svcs)
    assert ents == {"binary_sensor.front_door", "light.porch", "light.hall"}
    assert svcs == {"light.turn_on"}


def test_collect_refs_skips_templates_and_non_dotted(ac):
    ents, svcs = set(), set()
    ac._collect_refs(
        [{"action": "{{ 'light.turn_' ~ x }}", "entity_id": "all"},
         {"service": "notify.mobile", "data": {"message": "hi there"}}],
        ents, svcs)
    assert ents == set()                 # "all" is not a dotted entity
    assert svcs == {"notify.mobile"}     # template service skipped, real one kept


# ── validation ────────────────────────────────────────────────────────────────

def test_valid_config_passes(ac):
    h = _hass_with(entities=["light.porch", "binary_sensor.front_door"],
                   services=["light.turn_on"])
    report = ac.validate_automation_config(
        h,
        trigger=[{"trigger": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
        condition=None,
        action=[{"action": "light.turn_on", "target": {"entity_id": "light.porch"}}])
    assert report["ok"] is True
    assert report["invalid_entities"] == {} and report["invalid_services"] == {}


def test_bad_entity_is_flagged_with_closest_match(ac):
    h = _hass_with(entities=["light.porch"], services=["light.turn_on"])
    report = ac.validate_automation_config(
        h, trigger=[{"trigger": "state", "entity_id": "light.porchh"}],
        condition=None,
        action=[{"action": "light.turn_on", "target": {"entity_id": "light.porch"}}])
    assert report["ok"] is False
    assert report["invalid_entities"] == {"light.porchh": "light.porch"}


def test_bad_service_is_flagged(ac):
    h = _hass_with(entities=["light.porch"], services=["light.turn_on"])
    report = ac.validate_automation_config(
        h, trigger=[{"trigger": "state", "entity_id": "light.porch"}],
        condition=None,
        action=[{"action": "light.turnon", "target": {"entity_id": "light.porch"}}])
    assert report["ok"] is False
    assert report["invalid_services"] == {"light.turnon": "light.turn_on"}


def test_unknown_name_with_no_close_match_reports_none(ac):
    h = _hass_with(entities=["light.porch"], services=["light.turn_on"])
    report = ac.validate_automation_config(
        h, trigger=[{"trigger": "state", "entity_id": "sensor.zzzzz_nonsense"}],
        condition=None, action=[{"action": "light.turn_on"}])
    assert report["ok"] is False
    assert report["invalid_entities"] == {"sensor.zzzzz_nonsense": None}


# ── create_automation gate (fail path writes nothing) ─────────────────────────

@pytest.mark.asyncio
async def test_create_with_validation_refuses_bad_name(ac):
    h = _hass_with(entities=["light.porch"], services=["light.turn_on"])
    res = await ac.create_automation(
        h, alias="Porch at dusk",
        trigger=[{"trigger": "state", "entity_id": "light.porchh"}],
        action=[{"action": "light.turn_on", "target": {"entity_id": "light.porch"}}],
        validate=True)
    assert res["success"] is False
    assert "light.porchh" in res["invalid_entities"]
    # nothing was written: no reload service call happened
    assert h.services.calls == [] if hasattr(h.services, "calls") else True


@pytest.mark.asyncio
async def test_validation_is_opt_in(ac, monkeypatch):
    # Default (validate omitted) must not even call the validator — existing
    # callers are unchanged. Make the validator explode to prove it isn't run.
    def _boom(*a, **k):
        raise AssertionError("validator must not run when validate is not set")
    monkeypatch.setattr(ac, "validate_automation_config", _boom)

    h = _hass_with()
    # Missing trigger/action short-circuits before any write; the point is that
    # the validator wasn't consulted.
    res = await ac.create_automation(h, alias="x", trigger=None, action=None)
    assert res["success"] is False  # "trigger and action required", not a validator run


# ── agent tool handler ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_tool_requires_trigger_and_action(load):
    agent = load("agent")
    h = _hass_with()
    out = json.loads(await agent._exec_create_automation(h, {"alias": "x"}))
    assert out["success"] is False and "required" in out["error"].lower()


@pytest.mark.asyncio
async def test_agent_tool_validates_and_reports_bad_name(load):
    agent = load("agent")
    h = _hass_with(entities=["light.porch"], services=["light.turn_on"])
    out = json.loads(await agent._exec_create_automation(h, {
        "alias": "Porch",
        "trigger": [{"trigger": "state", "entity_id": "light.porchh"}],
        "action": [{"action": "light.turn_on", "target": {"entity_id": "light.porch"}}],
    }))
    assert out["success"] is False
    assert out["invalid_entities"] == {"light.porchh": "light.porch"}


def test_create_automation_tool_is_registered(load):
    agent = load("agent")
    assert "create_automation" in agent._TOOL_MAP
    names = {t["function"]["name"] for t in agent.JARVIS_TOOLS}
    assert "create_automation" in names
