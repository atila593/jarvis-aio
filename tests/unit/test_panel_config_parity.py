"""Config-flow ↔ main-panel settings parity (9.19.0).

Every field offered by the integration's options (Configure) flow must also be
settable from the main JARVIS panel — EXCEPT secrets, which deliberately stay in
the config flow / secrets.yaml (provider API keys and the IMAP password). This
test is the guard that keeps the two surfaces in step: add a non-secret option
field and you must also expose it in the panel (writable + shown), or this fails.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JARVIS = ROOT / "custom_components" / "jarvis"

# Fields that are intentionally config-flow-only:
#   - provider API keys + the IMAP password live in secrets.yaml, never the panel
#   - friday_confirm is a transient acknowledgement tick, not a stored setting
CONFIG_FLOW_ONLY = {"api_key", "imap_secret_key", "friday_confirm"}


def _option_fields() -> set[str]:
    strings = json.loads((JARVIS / "strings.json").read_text(encoding="utf-8"))
    steps = strings.get("options", {}).get("step", {})
    fields: set[str] = set()
    for body in steps.values():
        fields |= set((body or {}).get("data", {}) or {})
    return fields


def _panel_writable_keys() -> set[str]:
    src = (JARVIS / "websocket.py").read_text(encoding="utf-8")
    block = re.search(r"PANEL_WRITABLE_KEYS\s*=\s*\{(.*?)\n\}", src, re.S).group(1)
    return set(re.findall(r'"([a-z0-9_]+)"', block))


def _panel_readback_keys() -> set[str]:
    """Keys the panel's get_panel_data surfaces as config values (so a control can
    show its current state)."""
    src = (JARVIS / "websocket.py").read_text(encoding="utf-8")
    return set(re.findall(r'"([a-z0-9_]+)":', src))


def test_every_non_secret_option_field_is_panel_writable():
    missing = sorted(
        (_option_fields() - CONFIG_FLOW_ONLY) - _panel_writable_keys()
    )
    assert not missing, (
        "config-flow option fields missing from PANEL_WRITABLE_KEYS "
        f"(expose them in the panel or add to CONFIG_FLOW_ONLY): {missing}"
    )


def test_every_non_secret_option_field_is_shown_in_the_panel():
    # The readback dict in ws_get_panel_data must carry each field so its control
    # reflects the stored value (honorific, llm_provider, model, etc. included).
    missing = sorted(
        (_option_fields() - CONFIG_FLOW_ONLY) - _panel_readback_keys()
    )
    assert not missing, (
        "config-flow option fields not surfaced in get_panel_data config "
        f"(the panel control would show a blank/default): {missing}"
    )


def test_secrets_are_not_panel_writable():
    # Credentials must never be settable from the panel (owner decision).
    writable = _panel_writable_keys()
    assert "api_key" not in writable
    assert "imap_secret_key" not in writable
