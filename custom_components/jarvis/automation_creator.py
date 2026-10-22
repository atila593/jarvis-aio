"""
JARVIS Automation Creator (v5.6.0).

Registers a custom HA service + tool that allows JARVIS to create
Home Assistant automations from natural language instructions.

Usage via conversation: "JARVIS, create an automation that turns off
the living room lights at midnight."

The LLM generates the automation YAML, this module validates and
registers it with HA.
"""
from __future__ import annotations

import difflib
import logging
import re
import yaml
from typing import Any

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# A HA entity_id / dotted service reference: domain.object_id, all lowercase.
_DOTTED_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$")


def _collect_refs(obj: Any, entities: set, services: set) -> None:
    """Walk a trigger/condition/action config and collect the entity_ids and
    service names it references. Skips templates ({{ … }}) and the sentinel
    values ("all"/"none"); only dotted, HA-shaped names are collected so free
    text can't masquerade as a reference. Pure; never raises."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "entity_id":
                for e in (val if isinstance(val, list) else [val]):
                    if isinstance(e, str) and _DOTTED_RE.match(e):
                        entities.add(e)
            elif key in ("service", "action") and isinstance(val, str) \
                    and "{{" not in val and _DOTTED_RE.match(val):
                # action step service: old "service: light.turn_on" or new
                # HA "action: light.turn_on". Only a dotted value is a service.
                services.add(val)
            else:
                _collect_refs(val, entities, services)
    elif isinstance(obj, list):
        for item in obj:
            _collect_refs(item, entities, services)


def _known_services(hass: HomeAssistant) -> set:
    """The set of "domain.service" strings HA currently has registered."""
    out: set = set()
    try:
        for domain, svcs in (hass.services.async_services() or {}).items():
            for svc in svcs:
                out.add(f"{domain}.{svc}")
    except Exception:  # pragma: no cover - defensive
        pass
    return out


def validate_automation_config(
    hass: HomeAssistant, *, trigger, condition, action
) -> dict[str, Any]:
    """Check every entity_id and service a would-be automation references against
    what Home Assistant actually has, so JARVIS never installs an automation that
    silently refers to a guessed/misspelled name. Returns
    ``{"ok": bool, "invalid_entities": {bad: closest|None},
       "invalid_services": {bad: closest|None}}``. Best-effort and never raises;
    on any internal error it reports ``ok=True`` (fail-open — validation must
    never be what blocks a legitimate automation)."""
    try:
        entities: set = set()
        services: set = set()
        for section in (trigger, condition, action):
            _collect_refs(section, entities, services)

        known_entities = {s.entity_id for s in hass.states.async_all()}
        known_services = _known_services(hass)

        invalid_entities: dict = {}
        for e in sorted(entities):
            if e not in known_entities:
                match = difflib.get_close_matches(e, known_entities, n=1, cutoff=0.6)
                invalid_entities[e] = match[0] if match else None

        invalid_services: dict = {}
        for s in sorted(services):
            if s not in known_services:
                match = difflib.get_close_matches(s, known_services, n=1, cutoff=0.6)
                invalid_services[s] = match[0] if match else None

        return {
            "ok": not invalid_entities and not invalid_services,
            "invalid_entities": invalid_entities,
            "invalid_services": invalid_services,
        }
    except Exception as exc:  # pragma: no cover - defensive, fail-open
        _LOGGER.debug("automation validation error (fail-open): %s", exc)
        return {"ok": True, "invalid_entities": {}, "invalid_services": {}}


async def create_automation(
    hass: HomeAssistant,
    *,
    alias: str,
    description: str = "",
    trigger: list[dict] | dict = None,
    condition: list[dict] | dict | None = None,
    action: list[dict] | dict = None,
    mode: str = "single",
    validate: bool = False,
) -> dict[str, Any]:
    """
    Create and register a new HA automation programmatically.

    Args:
        alias: Human-readable name for the automation
        description: Optional description
        trigger: Trigger configuration (HA automation trigger format)
        condition: Optional condition(s)
        action: Action(s) to perform
        mode: Execution mode (single, restart, queued, parallel)

    Returns:
        {"success": True, "automation_id": "...", "alias": "..."}
        or {"success": False, "error": "..."}
    """
    if not trigger or not action:
        return {"success": False, "error": "Both trigger and action are required"}

    # Normalize to lists
    if isinstance(trigger, dict):
        trigger = [trigger]
    if isinstance(action, dict):
        action = [action]
    if condition and isinstance(condition, dict):
        condition = [condition]

    # Validate every referenced entity_id and service against the live HA
    # instance before writing anything (opt-in). An LLM can confidently emit a
    # plausible-but-wrong name; installing that produces an automation that
    # silently never fires. Refuse up front and hand back the closest real name
    # so the caller can correct and retry. Off by default, so existing callers
    # (the HA service + the learned-suggestion installer) are unchanged.
    if validate:
        report = validate_automation_config(
            hass, trigger=trigger, condition=condition, action=action)
        if not report["ok"]:
            parts = []
            for bad, near in report["invalid_entities"].items():
                parts.append(f"entity '{bad}'" + (f" (did you mean '{near}'?)" if near else " (no close match)"))
            for bad, near in report["invalid_services"].items():
                parts.append(f"service '{bad}'" + (f" (did you mean '{near}'?)" if near else " (no close match)"))
            return {
                "success": False,
                "error": "Validation failed — these names don't exist in Home "
                         "Assistant: " + "; ".join(parts) + ". Use search_entities "
                         "to find the real entity_id, then retry.",
                "invalid_entities": report["invalid_entities"],
                "invalid_services": report["invalid_services"],
            }

    # Build the automation config
    automation_id = f"jarvis_auto_{alias.lower().replace(' ', '_')[:40]}"
    auto_config = {
        "id": automation_id,
        "alias": f"JARVIS · {alias}",
        "description": description or f"Created by JARVIS: {alias}",
        "mode": mode,
        "triggers": trigger,
        "actions": action,
    }
    if condition:
        auto_config["conditions"] = condition

    # Validate the YAML is well-formed
    try:
        yaml_str = yaml.dump([auto_config], default_flow_style=False)
        _LOGGER.debug("JARVIS automation YAML:\n%s", yaml_str)
    except Exception as exc:
        return {"success": False, "error": f"YAML generation failed: {exc}"}

    # Write to automations.yaml
    try:
        automations_path = hass.config.path("automations.yaml")

        def _write_automation():
            # Read existing
            try:
                with open(automations_path) as f:
                    existing = yaml.safe_load(f) or []
            except FileNotFoundError:
                existing = []
            except Exception:
                existing = []

            if not isinstance(existing, list):
                existing = []

            # Check for duplicate ID
            existing = [a for a in existing if a.get("id") != automation_id]

            # Append new
            existing.append(auto_config)

            # Write back
            with open(automations_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

        await hass.async_add_executor_job(_write_automation)

        # Reload automations
        await hass.services.async_call("automation", "reload", blocking=True)

        _LOGGER.info("JARVIS created automation: %s (id=%s)", alias, automation_id)
        return {
            "success": True,
            "automation_id": automation_id,
            "alias": f"JARVIS · {alias}",
        }

    except Exception as exc:
        _LOGGER.error("JARVIS automation creation failed: %s", exc)
        return {"success": False, "error": str(exc)}
