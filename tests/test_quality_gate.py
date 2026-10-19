"""Tests for scripts/check_model_quality.py (the CI model quality gate)."""

import importlib.util
import json

import pytest

from heart import config

SCRIPT = config.PROJECT_ROOT / "scripts" / "check_model_quality.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("quality_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def meta(auc: float) -> dict:
    return {"model_family": "logistic_regression", "feature_set": "base", "best_params": {},
            "test_metrics": {"test_roc_auc": auc, "test_accuracy": 0.88, "test_precision": 0.84,
                             "test_recall": 0.93, "test_f1": 0.88}}


@pytest.mark.parametrize("candidate_auc, ok", [
    (0.962, True),    # identical
    (0.950, True),    # small drop within tolerance
    (0.990, True),    # improvement
    (0.930, False),   # dropped 0.032 > 0.02
    (0.840, False),   # below absolute floor (and dropped)
])
def test_gate_decision(gate, candidate_auc, ok):
    failures = gate.evaluate(meta(0.962), meta(candidate_auc), max_drop=0.02, min_auc=0.85)
    assert (not failures) is ok


def test_absolute_floor_applies_even_without_drop(gate):
    failures = gate.evaluate(meta(0.80), meta(0.80), max_drop=0.02, min_auc=0.85)
    assert any("floor" in f for f in failures)


def test_cli_exit_codes_and_report(gate, tmp_path, capsys):
    base, good, bad = (tmp_path / n for n in ("base.json", "good.json", "bad.json"))
    base.write_text(json.dumps(meta(0.96)))
    good.write_text(json.dumps(meta(0.955)))
    bad.write_text(json.dumps(meta(0.90)))

    assert gate.main(["--baseline", str(base), "--candidate", str(good)]) == 0
    assert "Quality gate: PASSED" in capsys.readouterr().out

    assert gate.main(["--baseline", str(base), "--candidate", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "Quality gate: FAILED" in out and "| roc_auc | 0.96 | 0.9 | -0.0600 |" in out


def test_committed_model_passes_against_itself(gate):
    assert gate.main(["--baseline", str(config.METADATA_FILE),
                      "--candidate", str(config.METADATA_FILE)]) == 0
