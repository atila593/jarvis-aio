"""Issue #395 — JARVIS must be able to control input_boolean (and other helper)
entities. The control path was always domain-generic; the bug was discovery:
input_boolean was absent from the home-context, entity-search and bulk_control
domain lists, so the model never learned the entity existed and fell back to a
vague HassTurnOff that matched nothing. These tests pin the control path for
input_boolean and that it is now discoverable."""
import json
import sys
import types

import pytest

if "homeassistant.helpers.llm" not in sys.modules:
    _llm = types.ModuleType("homeassistant.helpers.llm")
    _llm.async_get_api = lambda *a, **k: None
    sys.modules["homeassistant.helpers.llm"] = _llm


@pytest.fixture
def agent(load):
    return load("agent")


def _calls(fake_hass, domain, service):
    return [c for c in fake_hass.service_calls if c[0] == domain and c[1] == service]


async def test_control_device_turns_off_input_boolean(agent, fake_hass):
    """turn_off on an input_boolean routes to input_boolean.turn_off (the action
    map keys off the entity's own domain, so no special-casing is needed)."""
    fake_hass.states.set("input_boolean.luces_automaticas", "on",
                         friendly_name="Luces Automaticas")
    out = await agent._exec_control_device(
        fake_hass, {"entity_id": "input_boolean.luces_automaticas",
                    "action": "turn_off"})
    await fake_hass.drain()
    res = json.loads(out)
    assert res.get("success") is True, res
    assert _calls(fake_hass, "input_boolean", "turn_off"), fake_hass.service_calls


async def test_control_device_toggles_input_boolean(agent, fake_hass):
    fake_hass.states.set("input_boolean.guest_mode", "off")
    out = await agent._exec_control_device(
        fake_hass, {"entity_id": "input_boolean.guest_mode", "action": "toggle"})
    await fake_hass.drain()
    assert json.loads(out).get("success") is True
    assert _calls(fake_hass, "input_boolean", "toggle")


async def test_search_entities_finds_input_boolean_by_name(agent, fake_hass):
    """Discovery: searching by friendly name returns the input_boolean so the
    model can resolve its entity_id instead of giving up (the #395 failure)."""
    fake_hass.states.set("input_boolean.luces_automaticas", "on",
                         friendly_name="Luces Automaticas")
    out = await agent._exec_search_entities(fake_hass, {"query": "luces automaticas"})
    hits = json.loads(out)
    ids = [h.get("entity_id") for h in hits] if isinstance(hits, list) else []
    assert "input_boolean.luces_automaticas" in ids, out


async def test_home_context_lists_input_boolean(agent, fake_hass):
    fake_hass.states.set("input_boolean.luces_automaticas", "on",
                         friendly_name="Luces Automaticas")
    ctx = agent._build_home_context(fake_hass, learned={})
    assert "Luces Automaticas" in ctx, ctx


def test_input_boolean_is_in_bulk_control_enum(agent):
    """bulk_control can target the input_boolean domain (turn all off, etc.)."""
    tool = next(t for t in agent.JARVIS_TOOLS
                if t["function"]["name"] == "bulk_control")
    enum = tool["function"]["parameters"]["properties"]["domain"]["enum"]
    assert "input_boolean" in enum
