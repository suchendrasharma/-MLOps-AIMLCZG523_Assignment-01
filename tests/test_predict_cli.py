"""Tests for the scripts/predict.py command-line tool."""

import importlib.util
import json

import pytest

from heart import config

SCRIPT = config.PROJECT_ROOT / "scripts" / "predict.py"
SAMPLES = config.PROJECT_ROOT / "samples"

pytestmark = pytest.mark.skipif(
    not config.MODEL_FILE.exists(), reason="no exported model; run python -m heart.train"
)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("predict_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, capsys, *args):
    code = cli.main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.mark.parametrize("name", ["patient_high_risk.json", "patient_low_risk.json",
                                  "patients_batch.json"])
def test_sample_files_are_complete(name):
    data = json.loads((SAMPLES / name).read_text())
    for record in data if isinstance(data, list) else [data]:
        assert set(record) == set(config.FEATURES)


def test_single_patient(cli, capsys):
    code, out, _ = run(cli, capsys, SAMPLES / "patient_high_risk.json")
    assert code == 0
    result = json.loads(out)
    assert result["model"].startswith(config.REGISTERED_MODEL_NAME)
    (pred,) = result["predictions"]
    assert pred["prediction"] == 1 and pred["risk_label"] == "high risk"


def test_low_risk_sample_is_predicted_low(cli, capsys):
    _, out, _ = run(cli, capsys, SAMPLES / "patient_low_risk.json")
    assert json.loads(out)["predictions"][0]["prediction"] == 0


def test_batch_including_missing_value(cli, capsys):
    code, out, _ = run(cli, capsys, SAMPLES / "patients_batch.json")
    assert code == 0
    assert len(json.loads(out)["predictions"]) == 4


def test_csv_input_with_limit(cli, capsys):
    code, out, _ = run(cli, capsys, config.CLEAN_FILE, "--limit", 3)
    assert code == 0
    assert len(json.loads(out)["predictions"]) == 3


def test_missing_field_is_reported(cli, capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"age": 50}))
    code, _, err = run(cli, capsys, bad)
    assert code == 1
    assert "missing fields" in err and "thalach" in err


def test_unsupported_file_type(cli, capsys, tmp_path):
    bad = tmp_path / "patients.txt"
    bad.write_text("x")
    code, _, err = run(cli, capsys, bad)
    assert code == 1 and "unsupported" in err
