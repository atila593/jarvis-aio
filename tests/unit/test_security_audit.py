"""Read-only HA security-posture audit.

Covers the two checks with real signal — unencrypted remote access and
sensitive actions voice can trigger without confirmation — plus ordering,
the clean-home case, fail-soft behaviour, and the agent tool wiring. The
exposure and long-lived-token checks are HA-version-sensitive and guarded to
no-op without their APIs, so they're exercised only for that defensive shape.
"""
import asyncio
import json
import types

import pytest

from fakes import FakeHass


@pytest.fixture
def sa(load):
    return load("security_audit")


def _hass(entities=(), *, external_url=None):
    h = FakeHass()
    h.config = types.SimpleNamespace(time_zone="UTC", external_url=external_url)
    for item in entities:
        if isinstance(item, tuple):
            eid, attrs = item
            h.states.set(eid, "on", **attrs)
        else:
            h.states.set(item, "on")
    return h


def _run(sa, hass):
    # asyncio.run() builds and tears down its own loop, so it's robust whether or
    # not a prior async test left the default event loop closed (full-suite order).
    return asyncio.run(sa.run_audit(hass))


# ── insecure remote URL ───────────────────────────────────────────────────────

def test_http_external_url_is_high(sa, monkeypatch):
    # No sensitive entities, so only the URL check can fire.
    h = _hass(external_url="http://my-home.example.com:8123")
    out = _run(sa, h)
    assert any(f["severity"] == "high" and f["category"] == "remote-access"
               for f in out["findings"])


def test_https_external_url_is_clean(sa):
    h = _hass(external_url="https://my-home.example.com")
    out = _run(sa, h)
    assert [f for f in out["findings"] if f["category"] == "remote-access"] == []


def test_no_external_url_is_clean(sa):
    out = _run(sa, _hass())
    assert out["findings"] == []
    assert "No security-posture issues" in out["summary"]


# ── unconfirmed sensitive actions ─────────────────────────────────────────────

def _patch_policy(sa, load, monkeypatch, *, protected):
    """Pin policy.requires_confirmation to a fixed verdict (True=protected)."""
    policy = load("policy")
    monkeypatch.setattr(policy, "requires_confirmation",
                        lambda hass, d, s, e="": protected)
    return policy


def test_unprotected_lock_is_medium(sa, load, monkeypatch):
    _patch_policy(sa, load, monkeypatch, protected=False)
    h = _hass(entities=["lock.front_door"])
    out = _run(sa, h)
    hit = [f for f in out["findings"] if f["category"] == "unconfirmed-action"]
    assert hit and hit[0]["severity"] == "medium"
    assert "lock.unlock" in hit[0]["title"]
    assert "lock.front_door" in hit[0]["entities"]


def test_protected_lock_produces_no_finding(sa, load, monkeypatch):
    _patch_policy(sa, load, monkeypatch, protected=True)
    h = _hass(entities=["lock.front_door"])
    out = _run(sa, h)
    assert [f for f in out["findings"] if f["category"] == "unconfirmed-action"] == []


def test_only_security_covers_are_checked(sa, load, monkeypatch):
    _patch_policy(sa, load, monkeypatch, protected=False)
    h = _hass(entities=[
        ("cover.garage", {"device_class": "garage"}),   # security → flagged
        ("cover.living_blinds", {"device_class": "shade"}),  # not security → ignored
    ])
    out = _run(sa, h)
    hit = [f for f in out["findings"] if f["category"] == "unconfirmed-action"]
    assert hit and hit[0]["entities"] == ["cover.garage"]


# ── ordering, summary, fail-soft ──────────────────────────────────────────────

def test_findings_sorted_high_before_medium(sa, load, monkeypatch):
    _patch_policy(sa, load, monkeypatch, protected=False)
    h = _hass(entities=["lock.front_door"], external_url="http://x.example.com")
    out = _run(sa, h)
    sevs = [f["severity"] for f in out["findings"]]
    assert sevs.index("high") < sevs.index("medium")
    assert out["counts"].get("high") == 1 and out["counts"].get("medium") == 1


def test_run_audit_never_raises_on_bad_config(sa):
    # A hass whose config access explodes must not sink the audit.
    class _BoomConfig:
        def __getattr__(self, name):
            raise RuntimeError("boom")
    h = FakeHass()
    h.config = _BoomConfig()
    out = _run(sa, h)
    assert "findings" in out and isinstance(out["findings"], list)


# ── agent tool wiring ─────────────────────────────────────────────────────────

def test_security_audit_tool_registered(load):
    agent = load("agent")
    assert "security_audit" in agent._TOOL_MAP
    names = {t["function"]["name"] for t in agent.JARVIS_TOOLS}
    assert "security_audit" in names


@pytest.mark.asyncio
async def test_agent_tool_returns_findings(load, monkeypatch):
    agent = load("agent")
    policy = load("policy")
    monkeypatch.setattr(policy, "requires_confirmation", lambda *a, **k: False)
    h = _hass(entities=["lock.front_door"])
    out = json.loads(await agent._exec_security_audit(h, {}))
    assert "findings" in out and "summary" in out
    assert any(f["category"] == "unconfirmed-action" for f in out["findings"])
