"""Milestone C — live closure of the cognitive/agency loop (audit P0 #3).

The existing gates prove the loop stays *wired* (kernel_integration_check) and
the certification suite stays *observed* (kernel_certification_check). Neither
drives one real action the whole way round and checks each stage actually did
something. These do.

C1 drives ONE real control action through the live path and asserts a single
*correlated* chain — world-read -> authorize -> act -> verify -> **verified**
terminal outcome — under one correlation id, failing if any stage is only
recorded-as-present rather than exercised. C2 proves the *decide* stage is
genuinely in the loop: a stubbed model (no live LLM) emits a tool call that
drives the same real closure. The principle, per the audit: assert the EFFECT at
each stage (a service call, an authority decision, an emitted event, a verified
outcome) keyed to one trace — never "the stage reported itself present". Only
the model and the verify-sleep are stubbed; every kernel stage runs for real.
"""
import json
from types import SimpleNamespace

import pytest

from fakes import FakeHass


@pytest.fixture
def agent(load):
    # Load actuation first so the module object agent.py imports (and the one we
    # monkeypatch) are the same instance.
    load("actuation")
    return load("agent")


def _device_responds(h):
    """Make FakeHass's service call mutate state like a real device, so the
    verify read-back can confirm closure. Records into h.service_calls too."""
    async def _acall(domain, service, data=None, blocking=True, **kw):
        h.service_calls.append((domain, service, dict(data or {})))
        reached = {"turn_on": "on", "turn_off": "off", "lock": "locked",
                   "unlock": "unlocked", "open_cover": "open",
                   "close_cover": "closed"}.get(service)
        if reached is not None:
            eid = (data or {}).get("entity_id")
            if eid:
                h.states.set(eid, reached)
    h.services.async_call = _acall


def _device_ignores(h):
    """A device that accepts the command but never changes state — the service
    call is recorded, the world does not move. For the fail-to-land branch."""
    async def _acall(domain, service, data=None, blocking=True, **kw):
        h.service_calls.append((domain, service, dict(data or {})))
    h.services.async_call = _acall


def _no_verify_sleep(agent, monkeypatch):
    async def _nosleep(_secs):
        return None
    monkeypatch.setattr(agent, "_VERIFY_SLEEP", _nosleep)


# ── C1: one real action, one trace, verified outcome ─────────────────────────

@pytest.mark.asyncio
async def test_allowed_action_closes_the_loop_with_one_trace_and_verified_outcome(
        agent, load, monkeypatch):
    actuation = load("actuation")
    authority_bridge = load("authority_bridge")
    correlation = load("kernel.correlation")
    _no_verify_sleep(agent, monkeypatch)

    h = FakeHass()
    h.states.set("light.den", "off", friendly_name="Den light")
    _device_responds(h)

    # AUTHORIZE: keep the real belt, but record that it was consulted.
    authz = []
    real_authz = authority_bridge.enforced_decision
    monkeypatch.setattr(authority_bridge, "enforced_decision",
                        lambda *a, **k: authz.append((a, k)) or real_authz(*a, **k))

    # ACT→EVENT: capture the canonical actuation event + the request it carries.
    events = []
    real_emit = actuation.emit_event
    monkeypatch.setattr(actuation, "emit_event",
                        lambda hass, cap, eid, **k: events.append(
                            {"cap": cap, "eid": eid, "request": k.get("request")})
                        or real_emit(hass, cap, eid, **k))

    # VERIFY→OUTCOME: capture the terminal ActuatorOutcome.
    outcomes = []
    monkeypatch.setattr(actuation, "outcome",
                        lambda request, status, hass, entity_id, detail="":
                        outcomes.append({"request": request, "status": status}))

    with correlation.scope("loop-c1-0001"):
        res = json.loads(await agent._exec_control_device(
            h, {"action": "turn_on", "entity_id": "light.den"}))
    await h.drain()

    # end-to-end success
    assert res.get("success") is True and res.get("new_state") == "on"
    # AUTHORIZE — the belt genuinely ran for this capability (not skipped)
    assert authz, "authority gate was never consulted"
    assert authz[0][0][1:3] == ("light", "turn_on")
    # ACT — the real service call landed
    assert ("light", "turn_on") in [(d, s) for d, s, _ in h.service_calls]
    # ACT→EVENT — exactly one actuation event carrying the request
    assert len(events) == 1 and events[0]["cap"] == "light.turn_on"
    areq = events[0]["request"]
    assert areq is not None
    # TRACE — the request rides the correlation id set for this pass
    assert areq.correlation_id == "loop-c1-0001"
    # VERIFY→OUTCOME — a VERIFIED terminal outcome, on the SAME request/trace,
    # and it must actually exist: a stage recorded-as-present but not exercised
    # (no outcome) fails here.
    assert len(outcomes) == 1, "no terminal outcome — verify stage not exercised"
    assert outcomes[0]["status"] == agent._OUT_VERIFIED
    assert outcomes[0]["request"] is areq, "outcome not keyed to the same trace"


@pytest.mark.asyncio
async def test_authority_denial_blocks_actuation_in_the_live_path(
        agent, load, monkeypatch):
    # Prove AUTHORIZE is a real decision-maker, not a rubber stamp: with the
    # legacy gate allowing, the kernel engine (no identity -> security hold) must
    # stop the actuation before any service call happens.
    load("actuation")
    policy = load("policy")
    _no_verify_sleep(agent, monkeypatch)

    async def _allow(hass, domain, service, entity_id="", label=""):
        return True, ""
    monkeypatch.setattr(policy, "confirm_gate", _allow)

    h = FakeHass()
    h.states.set("lock.front", "unlocked", friendly_name="Front door")
    _device_responds(h)

    res = json.loads(await agent._exec_control_device(
        h, {"action": "unlock", "entity_id": "lock.front"}))

    # The engine belt held it: no unlock service call was ever made.
    assert ("lock", "unlock") not in [(d, s) for d, s, _ in h.service_calls]
    assert res.get("status") == "awaiting_confirmation"


# ── C2: the decide stage is in the loop, with the model stubbed ──────────────

class _ScriptedModel:
    """A provider with no live backend: it emits one control_device tool call,
    then a closing line — standing in for the LLM's decision so the real
    decide -> act -> verify loop runs deterministically."""
    def __init__(self, action, entity_id):
        self._tc = {"id": "call_1", "name": "control_device",
                    "args": {"action": action, "entity_id": entity_id}}
        self.turns = 0

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.7, **kwargs):
        self.turns += 1
        if self.turns == 1:
            return {"text": "", "tool_calls": [self._tc], "raw": None}
        return {"text": "Done, sir.", "tool_calls": [], "raw": None}

    def supports_vision(self):
        return False


@pytest.mark.asyncio
async def test_stubbed_model_decision_drives_the_real_closure(agent, load, monkeypatch):
    actuation = load("actuation")
    _no_verify_sleep(agent, monkeypatch)

    h = FakeHass()
    h.states.set("light.den", "off", friendly_name="Den light")
    _device_responds(h)

    model = _ScriptedModel("turn_on", "light.den")

    async def _fake_provider(*a, **k):
        return model
    monkeypatch.setattr(agent, "_create_provider_with_fallback", _fake_provider)

    outcomes = []
    monkeypatch.setattr(actuation, "outcome",
                        lambda request, status, hass, entity_id, detail="":
                        outcomes.append(status))

    reply = await agent.run_agent(
        h,
        messages=[{"role": "user", "content": "turn on the den light"}],
        persona="", provider_name="ollama", api_key="", model="scripted",
        hass_api=None, config={},
    )
    await h.drain()

    # The model's decision reached the real seam: the service call happened and
    # the world changed — decide -> act closed for real, no live LLM involved.
    assert model.turns >= 1
    assert ("light", "turn_on") in [(d, s) for d, s, _ in h.service_calls]
    assert h.states.get("light.den").state == "on"
    # and the loop verified the result
    assert agent._OUT_VERIFIED in outcomes
    assert isinstance(reply, str)


# ── C4: the branches — denied / failed / offline / restart ───────────────────
# C1/C2 prove the happy path closes. These prove the loop is equally honest when
# it does NOT close: a blocked, failed, or unperceivable action must fabricate no
# closure, and a closure is claimed only once it has actually been observed.

@pytest.mark.asyncio
async def test_denied_action_fabricates_no_closure(agent, load, monkeypatch):
    # DENIED. The engine belt holds an allowlisted security capability
    # (cover.open) even with the legacy gate forced to allow. A held action must
    # produce NOTHING downstream: no service call, no actuation event, no
    # terminal outcome — the loop never half-acts.
    actuation = load("actuation")
    authority_bridge = load("authority_bridge")
    policy = load("policy")
    _no_verify_sleep(agent, monkeypatch)

    async def _allow(hass, domain, service, entity_id="", label=""):
        return True, ""
    monkeypatch.setattr(policy, "confirm_gate", _allow)
    # Make the engine deterministically HOLD (independent of identity wiring).
    monkeypatch.setattr(authority_bridge, "record_control_parity",
                        lambda *a, **k: SimpleNamespace(decision="confirm",
                                                        reason="test"))

    events = []
    monkeypatch.setattr(actuation, "emit_event",
                        lambda *a, **k: events.append(k.get("request")))
    outcomes = []
    monkeypatch.setattr(actuation, "outcome",
                        lambda request, status, hass, entity_id, detail="":
                        outcomes.append(status))

    h = FakeHass()
    h.states.set("cover.garage", "closed", friendly_name="Garage door")
    _device_responds(h)

    res = json.loads(await agent._exec_control_device(
        h, {"action": "open", "entity_id": "cover.garage"}))
    await h.drain()

    assert res.get("status") == "awaiting_confirmation"
    assert ("cover", "open_cover") not in [(d, s) for d, s, _ in h.service_calls]
    assert events == [], "a blocked action must not emit an actuation event"
    assert outcomes == [], "a blocked action must not record a terminal outcome"


@pytest.mark.asyncio
async def test_action_that_never_lands_closes_as_mismatch_not_verified(
        agent, load, monkeypatch):
    # FAILED. The device accepts the command but never reaches the expected
    # state. The verify stage must read the world back, retry once, and then
    # record an HONEST terminal outcome — MISMATCH, on the same request — never a
    # false VERIFIED. This is the loop refusing to claim a closure that the world
    # never actually reached.
    actuation = load("actuation")
    database = load("database")
    correlation = load("kernel.correlation")
    _no_verify_sleep(agent, monkeypatch)
    monkeypatch.setattr(database, "save_activity", lambda **k: None)

    h = FakeHass()
    h.states.set("light.den", "off", friendly_name="Den light")
    _device_ignores(h)          # records the calls, never changes state

    outcomes = []
    monkeypatch.setattr(actuation, "outcome",
                        lambda request, status, hass, entity_id, detail="":
                        outcomes.append({"request": request, "status": status}))

    with correlation.scope("loop-c4-mismatch"):
        await agent._exec_control_device(
            h, {"action": "turn_on", "entity_id": "light.den"})
    await h.drain()

    # the service call DID land (the seam acted, and verify retried once) ...
    assert ("light", "turn_on") in [(d, s) for d, s, _ in h.service_calls]
    # ... but the world never reached "on", so the one terminal outcome is a
    # MISMATCH keyed to the same trace — not a VERIFIED.
    assert len(outcomes) == 1, "verify must record exactly one terminal outcome"
    assert outcomes[0]["status"] == agent._OUT_MISMATCH
    assert outcomes[0]["status"] != agent._OUT_VERIFIED
    assert getattr(outcomes[0]["request"], "correlation_id", None) == \
        "loop-c4-mismatch"


@pytest.mark.asyncio
async def test_unperceivable_device_is_refused_with_no_actuation(
        agent, load, monkeypatch):
    # OFFLINE / unknown. A device the world-model can't see at all: the loop must
    # refuse at the world-read, before any authorize or act. It never actuates
    # something it cannot perceive, and records no closure.
    actuation = load("actuation")
    _no_verify_sleep(agent, monkeypatch)

    events = []
    monkeypatch.setattr(actuation, "emit_event",
                        lambda *a, **k: events.append(k.get("request")))
    outcomes = []
    monkeypatch.setattr(actuation, "outcome",
                        lambda request, status, hass, entity_id, detail="":
                        outcomes.append(status))

    h = FakeHass()              # light.ghost is never set — unknown to the world
    _device_responds(h)

    res = json.loads(await agent._exec_control_device(
        h, {"action": "turn_on", "entity_id": "light.ghost"}))
    await h.drain()

    assert "not found" in (res.get("error") or "")
    assert h.service_calls == [], "nothing may be actuated for an unknown entity"
    assert events == []
    assert outcomes == []


@pytest.mark.asyncio
async def test_closure_is_only_claimed_once_verify_actually_runs(
        agent, load, monkeypatch):
    # RESTART. The verify-after-act is SCHEDULED, not awaited inline — it records
    # the terminal outcome only when it actually runs. If the process is torn
    # down between act and verify (a restart), no VERIFIED is on record: the loop
    # never claims a closure it hasn't observed. Draining (verify runs) then
    # produces it — closure is gated on observation, not on the act alone.
    actuation = load("actuation")
    _no_verify_sleep(agent, monkeypatch)

    h = FakeHass()
    h.states.set("light.den", "off", friendly_name="Den light")
    _device_responds(h)

    outcomes = []
    monkeypatch.setattr(actuation, "outcome",
                        lambda request, status, hass, entity_id, detail="":
                        outcomes.append(status))

    res = json.loads(await agent._exec_control_device(
        h, {"action": "turn_on", "entity_id": "light.den"}))

    # The act happened and the world moved — but verify has NOT run yet.
    assert res.get("success") is True
    assert h.states.get("light.den").state == "on"
    assert ("light", "turn_on") in [(d, s) for d, s, _ in h.service_calls]
    assert outcomes == [], "no closure may be claimed before verify runs"

    # Now the scheduled verify runs (as it would after the loop comes back).
    await h.drain()
    assert outcomes == [agent._OUT_VERIFIED], \
        "closure is recorded exactly when the deferred verify observes the world"
