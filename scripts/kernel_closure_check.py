#!/usr/bin/env python3
"""Milestone C closure gate — the cognitive/agency loop is PROVEN to close.

The integration gate (``kernel_integration_check``) proves the loop stays
*wired* and the certification gate (``kernel_certification_check``) proves the
observable scenario classes stay *observed*. Neither drives one real action the
whole way round and checks each stage actually did something. The closure proof
``tests/unit/test_cognitive_loop_closure.py`` does — it drives a real control
action through the live path (world-read → authorize → act → verify → **verified**
terminal outcome) under one correlation id, proves a stubbed model's decision
drives that same real closure, and proves the loop stays honest on the branches
that do NOT close (denied / failed / offline / restart).

This gate is that proof's teeth in CI. A proof that silently disappears — a test
file deleted, a load-bearing assertion gutted into a no-op, a closure test
``skip``/``xfail``'d — would let the loop stop closing while the rest of CI stays
green. This gate fails the build if that happens, so the closure proof can only
be removed or weakened deliberately (by updating this gate too). The companion
CI step runs the proof itself; this one guards that the proof is still there and
still asserts what it claims to.

Invariants (all must hold):
  1. the closure proof file is present and readable.
  2. every load-bearing closure test is still defined (by name): the one-trace
     verified-outcome happy path, the stubbed-model decide-in-loop proof, and
     the four non-closure branches (denied / failed / offline / restart).
  3. the proofs still assert what they claim — the closure VOCABULARY is present
     (``_OUT_VERIFIED`` / ``_OUT_MISMATCH``, a single ``correlation.scope`` trace
     threaded onto the request, a VERIFIED outcome ``is areq``, a blocked action
     ``awaiting_confirmation``, the stubbed model driving ``run_agent``).
  4. no closure test is disabled (no ``skip`` / ``skipif`` / ``xfail``), so the
     proof can't be silenced while still "present".

Pure stdlib; no Home Assistant import. Run ``--check`` in CI; a bare run prints
each invariant's status.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
_PROOF = ROOT / "tests" / "unit" / "test_cognitive_loop_closure.py"

# The load-bearing proofs — one per claim the closure suite makes. Renaming or
# removing any is a deliberate act that must update this gate too.
_REQUIRED_TESTS = (
    "test_allowed_action_closes_the_loop_with_one_trace_and_verified_outcome",
    "test_stubbed_model_decision_drives_the_real_closure",
    "test_denied_action_fabricates_no_closure",
    "test_action_that_never_lands_closes_as_mismatch_not_verified",
    "test_unperceivable_device_is_refused_with_no_actuation",
    "test_closure_is_only_claimed_once_verify_actually_runs",
)

# The assertion vocabulary that makes those proofs mean something — if a proof
# were hollowed into a no-op, these would be the first things to vanish.
_REQUIRED_TOKENS = (
    ("a VERIFIED terminal outcome", "_OUT_VERIFIED"),
    ("a MISMATCH (honest non-closure) outcome", "_OUT_MISMATCH"),
    ("a single correlation trace", "correlation.scope("),
    ("the trace threaded onto the request", ".correlation_id =="),
    ("the outcome keyed to the same request", "is areq"),
    ("a blocked action is held", "awaiting_confirmation"),
    ("the stubbed model drives the real loop", "run_agent("),
)

# A disabled proof is no proof — these must never appear in the closure suite.
_FORBIDDEN = ("@pytest.mark.skip", "pytest.skip(", "@pytest.mark.xfail")


def _checks() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    try:
        text = _PROOF.read_text(encoding="utf-8")
    except Exception as exc:
        return [("closure proof present", False,
                 f"cannot read {_PROOF.name}: {exc}")]
    out.append(("closure proof present", True, _PROOF.name))

    for name in _REQUIRED_TESTS:
        out.append((f"proof defined: {name}", f"def {name}(" in text, ""))

    for label, token in _REQUIRED_TOKENS:
        out.append((f"asserts {label}", token in text, token))

    for marker in _FORBIDDEN:
        out.append((f"no disabled proof ({marker})", marker not in text, ""))

    return out


def main(argv: list[str]) -> int:
    checks = _checks()
    failed = [c for c in checks if not c[1]]
    if "--check" not in argv:
        for name, ok, detail in checks:
            print(f"  {'ok ' if ok else 'XX '}{name}"
                  + (f" — {detail}" if detail else ""))
        print(f"\n{len(checks)} invariants, {len(failed)} failing")
    if "--check" in argv:
        if failed:
            print("CLOSURE PROOF WEAKENED — the loop is no longer proven to "
                  "close end-to-end:", file=sys.stderr)
            for name, _ok, detail in failed:
                print(f"  - {name}" + (f" ({detail})" if detail else ""),
                      file=sys.stderr)
            print("\ntests/unit/test_cognitive_loop_closure.py must keep driving "
                  "a real action the whole way round (Milestone C). Restore the "
                  "proof, or — if the closure suite is being deliberately "
                  "restructured — update this gate to match.", file=sys.stderr)
            return 1
        print(f"closure gate OK — the cognitive/agency loop is proven to close "
              f"end-to-end ({len(checks)} invariants, Milestone C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
