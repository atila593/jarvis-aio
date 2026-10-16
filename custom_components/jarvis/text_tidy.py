"""Small, pure text tidiers for spoken/notified output.

Issue #400 #8: the honorific is substituted into dozens of templates — both
f-strings (``f"Done, {honorific}."``) and placeholder replacement
(``tmpl.replace("{honorific}", honorific)``). When the owner sets a *blank*
honorific (they want JARVIS to address them without one), those templates leave
artifacts: a leading address-comma (``", the door is open"``), a dangling comma
before sentence punctuation (``"Done, ."``), or a comma stranded before an inset
dash (``"Worth mentioning,  — the screen is on"``).

Rewiring every template is infeasible and error-prone, so instead we clean the
artifacts at the output boundary. ``tidy_address`` is deliberately conservative:
each rule only matches a shape that is grammatically invalid in well-formed
English, so it is a **no-op on normal text** (including any non-empty honorific)
and safe to apply unconditionally to every spoken/notified string.
"""
from __future__ import annotations

import re

__all__ = ["tidy_address"]

# A comma stranded before an inset dash: "Worth mentioning,  — the screen…"
#   → "Worth mentioning — the screen…"  (em-dash, en-dash or spaced hyphen)
_COMMA_BEFORE_DASH = re.compile(r",\s+(—|–|-)(\s)")
# A comma stranded before sentence punctuation: "Done, ." / "Noted , !"
_COMMA_BEFORE_PUNCT = re.compile(r"\s*,\s*([.!?;:])")
# A space before a comma: "turned off , please" → "turned off, please"
_SPACE_BEFORE_COMMA = re.compile(r"\s+,")
# Two commas with only whitespace between: ",  ," → ","
_DOUBLE_COMMA = re.compile(r",\s*,")
# A leading address-comma at the very start (or just after an opening quote):
#   ", the door is open" → "the door is open"
_LEADING_COMMA = re.compile(r"^(\s*[\"'“‘(]?\s*),\s*")
# Runs of whitespace left behind by a dropped word.
_MULTISPACE = re.compile(r"[ \t]{2,}")


def tidy_address(text: str) -> str:
    """Remove address-comma artifacts left by an empty honorific.

    No-op on well-formed text and on ``None``/empty input. Order matters: fix the
    stranded/leading commas first, then collapse the whitespace they leave."""
    if not text or not isinstance(text, str):
        return text
    out = text
    out = _DOUBLE_COMMA.sub(",", out)
    out = _COMMA_BEFORE_DASH.sub(r" \1\2", out)
    out = _COMMA_BEFORE_PUNCT.sub(r"\1", out)
    out = _SPACE_BEFORE_COMMA.sub(",", out)
    # Strip a leading address-comma and re-capitalise the new first letter.
    m = _LEADING_COMMA.match(out)
    if m:
        lead = m.group(1)
        rest = out[m.end():]
        if rest[:1].islower():
            rest = rest[:1].upper() + rest[1:]
        out = lead + rest
    out = _MULTISPACE.sub(" ", out)
    return out
