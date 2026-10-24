"""Milestone C — the closure gate guards the loop-closure proof.

scripts/kernel_closure_check.py is the CI teeth of Milestone C: it fails the
build if tests/unit/test_cognitive_loop_closure.py stops proving the loop closes
end-to-end (a proof deleted, a load-bearing assertion gutted, a closure test
disabled). These tests run the gate against the real tree (it must pass) and
confirm it actually catches a weakened proof.
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
GATE = ROOT / "scripts" / "kernel_closure_check.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("kernel_closure_check", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gate_passes_on_the_real_tree():
    gate = _load_gate()
    assert gate.main(["--check"]) == 0


def test_all_invariants_green():
    gate = _load_gate()
    checks = gate._checks()
    assert checks, "gate produced no invariants"
    assert all(ok for _name, ok, _detail in checks)


def test_gate_catches_a_deleted_proof(monkeypatch):
    # If a load-bearing closure test were removed, the gate must fail.
    gate = _load_gate()
    real = pathlib.Path.read_text

    def _fake_read(self, *a, **k):
        txt = real(self, *a, **k)
        if self.name == "test_cognitive_loop_closure.py":
            txt = txt.replace(
                "def test_action_that_never_lands_closes_as_mismatch_not_verified(",
                "def _removed_(")
        return txt

    monkeypatch.setattr(pathlib.Path, "read_text", _fake_read)
    assert gate.main(["--check"]) == 1


def test_gate_catches_a_gutted_assertion(monkeypatch):
    # If the verified-outcome assertion were hollowed out, the gate must fail.
    gate = _load_gate()
    real = pathlib.Path.read_text

    def _fake_read(self, *a, **k):
        txt = real(self, *a, **k)
        if self.name == "test_cognitive_loop_closure.py":
            txt = txt.replace("_OUT_VERIFIED", "_NOPE")
        return txt

    monkeypatch.setattr(pathlib.Path, "read_text", _fake_read)
    assert gate.main(["--check"]) == 1


def test_gate_catches_a_disabled_proof(monkeypatch):
    # If a closure test were skipped, the gate must fail — a silenced proof is
    # no proof.
    gate = _load_gate()
    real = pathlib.Path.read_text

    def _fake_read(self, *a, **k):
        txt = real(self, *a, **k)
        if self.name == "test_cognitive_loop_closure.py":
            txt = txt.replace(
                "@pytest.mark.asyncio\nasync def "
                "test_closure_is_only_claimed_once_verify_actually_runs(",
                "@pytest.mark.skip(reason='x')\n@pytest.mark.asyncio\nasync def "
                "test_closure_is_only_claimed_once_verify_actually_runs(")
        return txt

    monkeypatch.setattr(pathlib.Path, "read_text", _fake_read)
    assert gate.main(["--check"]) == 1
