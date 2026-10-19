"""Reproducibility checks on the committed model artifacts.

These guard the claim that anyone can rebuild the exact served model from the repo:
same data (by hash) + metadata (family, params, feature set) -> identical predictions.
"""

import json
import re

import mlflow
import numpy as np
import pytest

from heart import config, train
from heart.data import load_clean
from heart.evaluate import compute_metrics
from heart.predict import load_metadata, load_pipeline

pytestmark = pytest.mark.skipif(
    not config.MODEL_FILE.exists(), reason="no exported model; run python -m heart.train"
)


@pytest.fixture(scope="module")
def meta():
    return load_metadata()


@pytest.fixture(scope="module")
def model():
    return load_pipeline()


@pytest.fixture(scope="module")
def data():
    return load_clean(config.CLEAN_FILE)


def test_metadata_describes_a_complete_model_card(meta):
    for key in ("model_version", "model_family", "feature_set", "best_params",
                "input_features", "cv_metrics", "test_metrics", "model_sha256",
                "data_sha256", "decision_threshold"):
        assert key in meta, key
    assert meta["input_features"] == config.FEATURES


def test_model_file_matches_recorded_checksum(meta):
    assert train.file_sha256(config.MODEL_FILE) == meta["model_sha256"]


def test_training_data_unchanged_since_export(meta, data):
    assert train.data_fingerprint(data) == meta["data_sha256"], (
        "heart_clean.csv differs from the data the model was trained on; retrain")


def test_rebuilding_from_metadata_reproduces_the_model(meta, model, data):
    X_train, X_test, y_train, _ = train.split(data)
    rebuilt = train.rebuild_from_metadata(meta).fit(X_train, y_train)
    assert np.allclose(rebuilt.predict_proba(X_test), model.predict_proba(X_test), atol=1e-9)


def test_recorded_test_metrics_are_reproducible(meta, model, data):
    _, X_test, _, y_test = train.split(data)
    metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
    for name, value in metrics.items():
        assert round(value, 4) == pytest.approx(meta["test_metrics"][f"test_{name}"]), name


def test_mlflow_format_model_agrees_with_joblib(model, data):
    if not (config.MLFLOW_MODEL_DIR / "MLmodel").exists():
        pytest.skip("models/mlflow_model not exported")
    pyfunc = mlflow.pyfunc.load_model(str(config.MLFLOW_MODEL_DIR))
    X = data[config.FEATURES].astype(float)
    assert np.array_equal(np.asarray(pyfunc.predict(X)), model.predict(X))


def test_requirements_are_fully_pinned():
    pattern = re.compile(r"^[A-Za-z0-9_.\-\[\]]+==[\w.]+")
    lines = [ln.split("#")[0].strip()
             for ln in (config.PROJECT_ROOT / "requirements.txt").read_text().splitlines()]
    unpinned = [ln for ln in lines if ln and not pattern.match(ln)]
    assert not unpinned, f"unpinned requirements: {unpinned}"


def test_metadata_is_valid_json_on_disk():
    json.loads(config.METADATA_FILE.read_text())
