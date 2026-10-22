"""Home Assistant security-posture audit (read-only).

A self-contained check of the home's security posture from what JARVIS can see
in-process — no LAN scanning, no external service, nothing written. It answers
"is anything about this setup leaving the home exposed?" and, in keeping with
JARVIS's own governance, "are the dangerous actions actually gated?".

Checks (each independent and fail-soft — one failing check never sinks the rest):
  * remote access served over plain http:// (unencrypted)
  * sensitive actions (unlock, disarm, open a garage/gate/door cover) that voice
    can trigger WITHOUT confirmation — the gap JARVIS's own voice-confirm closes
  * sensitive entities (locks, alarms, cameras) exposed to the conversation
    agent, so the surface is visible
  * long-lived access tokens, and any that have never been used

Report-only: `run_audit` returns a findings list the agent narrates; it drives
nothing and changes nothing. Exposed as the `security_audit` agent tool.
"""
from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Domains whose actions are security-relevant.
_LOCK = "lock"
_ALARM = "alarm_control_panel"
# Cover device_classes that secure the home (as opposed to blinds/shades/awnings).
_SECURITY_COVER_CLASSES = frozenset({"garage", "gate", "door"})

# Severities, most to least serious — used to order and summarize findings.
_SEV_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _finding(sev: str, category: str, title: str, detail: str,
             entities: list | None = None) -> dict:
    f = {"severity": sev, "category": category, "title": title, "detail": detail}
    if entities:
        f["entities"] = sorted(entities)[:25]
    return f


def _check_insecure_remote_url(hass, findings: list) -> None:
    """Remote access served over plain http:// exposes credentials and traffic."""
    try:
        ext = getattr(hass.config, "external_url", None)
        if isinstance(ext, str) and ext.lower().startswith("http://"):
            findings.append(_finding(
                "high", "remote-access",
                "Remote access is unencrypted",
                f"external_url is {ext!r} — it uses http://, so anything reaching "
                "Home Assistant from outside travels in the clear. Put it behind "
                "https (a reverse proxy with TLS, or Home Assistant Cloud)."))
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.debug("security_audit remote-url check error: %s", exc)


def _check_unprotected_sensitive_actions(hass, findings: list) -> None:
    """Sensitive actions that voice can trigger with no confirmation. This is the
    gap JARVIS's voice-confirm is meant to close, surfaced per entity."""
    try:
        from . import policy
    except Exception:  # pragma: no cover - policy must be importable
        return
    # (domain, service, predicate over the state) for each sensitive family.
    probes = [
        (_LOCK, "unlock", lambda st: True),
        (_ALARM, "alarm_disarm", lambda st: True),
        ("cover", "open_cover",
         lambda st: (st.attributes.get("device_class") in _SECURITY_COVER_CLASSES)),
    ]
    unprotected: dict[str, list] = {}
    try:
        states = list(hass.states.async_all())
    except Exception:  # pragma: no cover - defensive
        return
    for st in states:
        eid = getattr(st, "entity_id", "")
        domain = eid.split(".", 1)[0] if "." in eid else ""
        for pdomain, service, pred in probes:
            if domain != pdomain:
                continue
            try:
                if not pred(st):
                    continue
                if not policy.requires_confirmation(hass, domain, service, eid):
                    unprotected.setdefault(f"{domain}.{service}", []).append(eid)
            except Exception:  # pragma: no cover - per-entity, never fatal
                continue
    for cap, eids in unprotected.items():
        findings.append(_finding(
            "medium", "unconfirmed-action",
            f"{cap} can be triggered by voice without confirmation",
            f"{len(eids)} entity(ies) accept {cap} with no confirmation step. "
            "Enable voice confirmation (voice_confirm_enabled) or add these to the "
            "per-entity protection list so a spoken or misheard command can't "
            "unlock/disarm/open them unchallenged.",
            entities=eids))


def _check_exposed_sensitive_entities(hass, findings: list) -> None:
    """Locks / alarms / cameras reachable by the conversation agent. Best-effort:
    if HA's exposure helper isn't importable, the check simply no-ops."""
    try:
        from homeassistant.helpers import entity as _entity
        should_expose = getattr(_entity, "async_should_expose", None)
        if should_expose is None:
            return
    except Exception:
        return
    exposed: list = []
    try:
        for st in hass.states.async_all():
            eid = getattr(st, "entity_id", "")
            if eid.split(".", 1)[0] in (_LOCK, _ALARM, "camera"):
                try:
                    if should_expose(hass, "conversation", eid):
                        exposed.append(eid)
                except Exception:
                    continue
    except Exception:  # pragma: no cover - defensive
        return
    if exposed:
        findings.append(_finding(
            "info", "exposed-entity",
            "Security entities are reachable by the voice agent",
            f"{len(exposed)} lock/alarm/camera entity(ies) are exposed to the "
            "conversation agent. That's expected if you control them by voice — "
            "just make sure the confirmation gate above covers the dangerous ones.",
            entities=exposed))


async def _check_long_lived_tokens(hass, findings: list) -> None:
    """Count long-lived access tokens and flag any that have never been used.
    Best-effort: the auth manager/shape varies by HA version, so guard heavily."""
    try:
        auth = getattr(hass, "auth", None)
        if auth is None or not hasattr(auth, "async_get_refresh_tokens"):
            return
        tokens = await auth.async_get_refresh_tokens()
        lived = [t for t in (tokens or [])
                 if getattr(t, "token_type", None) == "long_lived_access_token"]
        if not lived:
            return
        unused = [getattr(t, "client_name", None) or "unnamed"
                  for t in lived if getattr(t, "last_used_at", "sentinel") is None]
        detail = (f"{len(lived)} long-lived access token(s) exist. These never "
                  "expire; revoke any you don't recognize in your profile.")
        sev = "info"
        if unused:
            sev = "low"
            detail += f" {len(unused)} has/have never been used: {', '.join(sorted(set(unused))[:10])}."
        findings.append(_finding(
            sev, "long-lived-token", "Long-lived access tokens are present", detail))
    except Exception as exc:  # pragma: no cover - defensive, version-sensitive
        _LOGGER.debug("security_audit token check error: %s", exc)


async def run_audit(hass) -> dict[str, Any]:
    """Run every check and return ``{"findings": [...], "summary": str,
    "counts": {sev: n}}``. Findings are ordered most-serious first. Never raises:
    a check that errors is simply absent from the report."""
    findings: list = []
    _check_insecure_remote_url(hass, findings)
    _check_unprotected_sensitive_actions(hass, findings)
    _check_exposed_sensitive_entities(hass, findings)
    await _check_long_lived_tokens(hass, findings)

    findings.sort(key=lambda f: _SEV_ORDER.get(f["severity"], 9))
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1

    if not findings:
        summary = "No security-posture issues found in the checks JARVIS can run."
    else:
        bits = [f"{counts[s]} {s}" for s in ("high", "medium", "low", "info") if s in counts]
        summary = "Security audit: " + ", ".join(bits) + "."
    return {"findings": findings, "summary": summary, "counts": counts}
