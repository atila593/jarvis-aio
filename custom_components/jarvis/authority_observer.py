"""Log-only state-change parity observer for the authority enforce-allowlist.

Step B of the authority enforce rollout (follow-up to the Step A parity tally,
PR #377). The engine-vs-gate agreement tally in :mod:`authority_bridge` only
accrues when JARVIS *itself* actuates a protected device through the agent —
``control_device`` / ``bulk_control``. For the enforce-allowlisted security
capabilities (lock / alarm / cover) that almost never happens in a normal
household: people arm the alarm from a dashboard, unlock a door physically or
from their phone, open the garage from the car. So the per-capability buckets
the allowlist-widening decision depends on stay empty no matter how long we
wait — which is exactly what a real two-week diagnostics snapshot showed
(``agree=0, disagree=0, by_capability={}``).

This observer closes that gap WITHOUT touching any authoritative path. It
watches the enforce-allowlisted domains for real state transitions — however
they were initiated — maps each transition to the capability it corresponds to,
and records engine-vs-gate parity for it, exactly as the agent path does and
into the same tally. For an externally-initiated change the live gate was never
actually consulted, so the "actual" outcome compared against is the gate's
*hypothetical* verdict (:func:`policy.requires_confirmation`): "had this action
come through JARVIS, would the capability engine have agreed with the gate?" —
precisely the question the enforce flip must answer. The engine request is built
with the same minimal context the agent path uses (no identity / confidence),
so the samples are directly comparable to the ones already in the tally.

Observe-only: it never calls a service, never blocks, never raises into the
bus, and drives nothing. Kill-switch: ``AUTHORITY_OBSERVER_ENABLED`` (and the
``authority_observer`` config key, default on).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

_LOGGER = logging.getLogger(__name__)

# Kill-switch. Flip to False (or set the ``authority_observer`` config key to
# False) to silence the observer entirely on the next load.
AUTHORITY_OBSERVER_ENABLED = True

# The domains the authority enforce allowlist covers. We only watch these, so
# the per-event cost is bounded to the handful of security devices in the home.
_WATCHED_DOMAINS = ("lock", "alarm_control_panel", "cover")

# States that carry no actuation meaning — a transition to/from one of these is
# not a real control action and is ignored (avoids recording boot/availability
# churn as parity samples).
_IGNORED_STATES = frozenset({"", "unknown", "unavailable", "none", None})


def _service_for_transition(domain: str, old: Optional[str],
                            new: Optional[str]) -> Optional[str]:
    """The service name a ``domain`` state transition corresponds to, or None
    when the transition is not a protected control action worth recording.

    Pure: only the (domain, old, new) triple decides the mapping. A no-op
    transition (``old == new``) and any transition touching an ignored state
    returns None.
    """
    if new in _IGNORED_STATES or old in _IGNORED_STATES:
        return None
    if old == new:
        return None
    if domain == "lock":
        if new == "unlocked":
            return "unlock"
        if new == "locked":
            return "lock"
        if new == "open":
            return "open"
        return None
    if domain == "cover":
        if new in ("open", "opening"):
            return "open_cover"
        if new in ("closed", "closing"):
            return "close_cover"
        return None
    if domain == "alarm_control_panel":
        if new == "disarmed":
            return "alarm_disarm"
        if new == "armed_away":
            return "alarm_arm_away"
        if new == "armed_home":
            return "alarm_arm_home"
        if new == "armed_night":
            return "alarm_arm_night"
        if new == "armed_vacation":
            return "alarm_arm_vacation"
        if new == "armed_custom_bypass":
            return "alarm_arm_custom_bypass"
        return None
    return None


def _enabled(hass) -> bool:
    """Whether the observer should run (module switch AND config key)."""
    if not AUTHORITY_OBSERVER_ENABLED:
        return False
    try:
        from . import jarvis_config
        return jarvis_config.get("authority_observer", True) is not False
    except Exception:
        return True


def _record_transition(hass, entity_id: str, old: Optional[str],
                       new: Optional[str]) -> bool:
    """Map one transition to a capability and record engine-vs-gate parity for
    it (log-only). Returns True when a sample was recorded. Never raises."""
    try:
        domain = entity_id.split(".", 1)[0]
        if domain not in _WATCHED_DOMAINS:
            return False
        service = _service_for_transition(domain, old, new)
        if service is None:
            return False
        from . import policy, authority_bridge
        # The gate's hypothetical verdict for this action (True = it would have
        # required confirmation → the engine's parity "actual" is CONFIRM).
        held = policy.requires_confirmation(hass, domain, service, entity_id)
        authority_bridge.record_control_parity(
            hass, domain, service, allowed=(not held), scope=entity_id,
            intent=f"observed:{old}->{new}")
        return True
    except Exception as exc:  # pragma: no cover - log-only, never disrupt the bus
        _LOGGER.debug("authority observer record failed for %s: %s",
                      entity_id, exc)
        return False


def register(hass) -> Optional[Callable[[], None]]:
    """Subscribe to state changes on the enforce-allowlisted domains and record
    engine-vs-gate parity for each real actuation. Returns an unsubscribe
    callable (collected by the resource registry so unload tears it down), or
    None when the observer is disabled or cannot be wired.

    Best-effort: a failure here never affects setup or the authoritative path.
    """
    if not _enabled(hass):
        _LOGGER.debug("authority observer disabled; not registering")
        return None
    try:
        from homeassistant.helpers.event import async_track_state_change_event

        entity_ids = [
            s.entity_id for s in hass.states.async_all()
            if s.entity_id.split(".", 1)[0] in _WATCHED_DOMAINS
        ]
        if not entity_ids:
            _LOGGER.debug(
                "authority observer: no %s entities to watch yet",
                "/".join(_WATCHED_DOMAINS))
            return None

        def _on_state_change(event) -> None:
            try:
                data = event.data
                old = data.get("old_state")
                new = data.get("new_state")
                if new is None:
                    return
                _record_transition(
                    hass, new.entity_id,
                    old.state if old is not None else None, new.state)
            except Exception as exc:  # pragma: no cover - never raise into bus
                _LOGGER.debug("authority observer handler error: %s", exc)

        unsub = async_track_state_change_event(hass, entity_ids, _on_state_change)
        _LOGGER.info(
            "authority observer: watching %d device(s) across %s for "
            "engine-vs-gate parity (log-only)",
            len(entity_ids), "/".join(_WATCHED_DOMAINS))
        return unsub
    except Exception as exc:  # never raise into setup
        _LOGGER.debug("authority observer registration failed: %s", exc)
        return None
