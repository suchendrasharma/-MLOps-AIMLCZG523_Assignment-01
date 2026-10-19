"""Tests for model training, evaluation, selection, persistence and prediction."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from heart import config, train
from heart.data import load_clean
from heart.evaluate import compute_metrics, feature_importance
from heart.features import build_pipeline
from heart.predict import load_metadata, load_pipeline, predict_records


@pytest.fixture(scope="module")
def splits():
    """The committed cleaned dataset, split exactly as in training."""
    return train.split(load_clean(config.CLEAN_FILE))


@pytest.fixture(scope="module")
def fitted_lr(splits):
    X_train, _, y_train, _ = splits
    return build_pipeline(LogisticRegression(solver="liblinear", random_state=0)).fit(
        X_train, y_train)


class TestMetrics:
    def test_known_values(self):
        y = [0, 0, 1, 1]
        proba = [0.1, 0.6, 0.4, 0.9]  # threshold 0.5 -> preds [0, 1, 0, 1]
        m = compute_metrics(y, proba)
        assert m["accuracy"] == 0.5
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["roc_auc"] == 0.75

    def test_threshold_changes_predictions_not_auc(self):
        y, proba = [0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]
        low = compute_metrics(y, proba, threshold=0.3)
        assert low["recall"] == 1.0
        assert low["roc_auc"] == compute_metrics(y, proba)["roc_auc"]


class TestChampionSelection:
    def test_best_auc_wins_when_clear(self):
        summary = pd.DataFrame({"cv_roc_auc": [0.90, 0.85], "cv_recall": [0.70, 0.95]})
        assert train.select_champion(summary) == 0

    def test_near_tie_is_broken_by_recall(self):
        summary = pd.DataFrame({"cv_roc_auc": [0.910, 0.907], "cv_recall": [0.75, 0.82]})
        assert train.select_champion(summary, tol=0.005) == 1

    def test_tie_tolerance_is_respected(self):
        summary = pd.DataFrame({"cv_roc_auc": [0.910, 0.900], "cv_recall": [0.75, 0.82]})
        assert train.select_champion(summary, tol=0.005) == 0


class TestTrainedPipeline:
    def test_split_is_stratified_and_sized(self, splits):
        X_train, X_test, y_train, y_test = splits
        assert len(X_test) == round(config.TEST_SIZE * (len(X_train) + len(X_test)))
        assert abs(y_train.mean() - y_test.mean()) < 0.03

    def test_beats_chance_by_a_wide_margin(self, fitted_lr, splits):
        _, X_test, _, y_test = splits
        proba = fitted_lr.predict_proba(X_test)[:, 1]
        assert compute_metrics(y_test, proba)["roc_auc"] > 0.85

    def test_probabilities_are_valid(self, fitted_lr, splits):
        proba = fitted_lr.predict_proba(splits[1])
        assert ((proba >= 0) & (proba <= 1)).all()
        assert np.allclose(proba.sum(axis=1), 1)

    def test_training_is_deterministic(self, splits):
        X_train, X_test, y_train, _ = splits
        make = lambda: build_pipeline(  # noqa: E731
            RandomForestClassifier(n_estimators=50, random_state=7)).fit(X_train, y_train)
        assert np.array_equal(make().predict_proba(X_test), make().predict_proba(X_test))

    def test_handles_missing_values_at_inference(self, fitted_lr, splits):
        row = splits[1].iloc[[0]].astype(float)
        row[["ca", "thal"]] = np.nan
        assert fitted_lr.predict_proba(row).shape == (1, 2)

    def test_joblib_round_trip(self, fitted_lr, splits, tmp_path):
        path = tmp_path / "model.joblib"
        joblib.dump(fitted_lr, path)
        reloaded = load_pipeline(path)
        assert np.array_equal(reloaded.predict_proba(splits[1]), fitted_lr.predict_proba(splits[1]))


class TestFeatureImportance:
    def test_linear_coefficients_are_signed_and_named(self, fitted_lr):
        imp = feature_importance(fitted_lr)
        assert (imp < 0).any() and (imp > 0).any()
        assert "cp_4.0" in imp.index
        assert imp.abs().is_monotonic_decreasing

    def test_tree_importances_sum_to_one(self, splits):
        X_train, _, y_train, _ = splits
        rf = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0))
        imp = feature_importance(rf.fit(X_train, y_train))
        assert imp.sum() == pytest.approx(1.0)


class TestPredictRecords:
    RECORD = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
              "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
              "ca": 0, "thal": 6}

    def test_output_schema(self, fitted_lr):
        (res,) = predict_records(fitted_lr, [self.RECORD])
        assert set(res) == {"prediction", "risk_label", "probability", "confidence"}
        assert res["prediction"] in (0, 1)
        assert 0.5 <= res["confidence"] <= 1.0

    def test_prediction_consistent_with_probability(self, fitted_lr):
        (res,) = predict_records(fitted_lr, [self.RECORD])
        assert res["prediction"] == int(res["probability"] >= config.DECISION_THRESHOLD)
        expected_conf = res["probability"] if res["prediction"] else 1 - res["probability"]
        assert res["confidence"] == pytest.approx(expected_conf, abs=1e-4)

    def test_batch_and_missing_field_value(self, fitted_lr):
        records = [self.RECORD, {**self.RECORD, "ca": None, "age": 45}]
        assert len(predict_records(fitted_lr, records)) == 2

    def test_missing_model_file_has_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="heart.train"):
            load_pipeline(tmp_path / "missing.joblib")


@pytest.mark.slow
def test_quick_training_run_end_to_end(tmp_path, monkeypatch):
    """Full train.run() in quick mode against a throwaway MLflow store and output folder."""
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path.as_posix()}/m.db")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", (tmp_path / "mlruns").as_uri())
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "METADATA_FILE", tmp_path / "models" / "metadata.json")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")

    champion = train.run(quick=True, do_register=True)

    assert config.MODEL_FILE.exists()
    meta = load_metadata(config.METADATA_FILE)
    assert meta["model_version"] == "1"
    assert meta["mlflow_run_id"] == champion["run_id"]
    assert meta["input_features"] == config.FEATURES
    assert meta["test_metrics"]["test_roc_auc"] > 0.8
    assert len(list((tmp_path / "figures").glob("train_*.png"))) == 4
    json.dumps(meta)  # metadata must be JSON-serialisable for the API
