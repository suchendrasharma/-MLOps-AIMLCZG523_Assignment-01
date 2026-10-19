"""API tests via FastAPI's TestClient (runs the real app, including startup and middleware)."""

import json
import logging
import shutil

import pytest
from fastapi.testclient import TestClient

from api.logging_config import json_formatter
from heart import config

pytestmark = pytest.mark.skipif(
    not config.MODEL_FILE.exists(), reason="no exported model; run python -m heart.train"
)

SAMPLES = config.PROJECT_ROOT / "samples"
HIGH = json.loads((SAMPLES / "patient_high_risk.json").read_text())
LOW = json.loads((SAMPLES / "patient_low_risk.json").read_text())


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:  # `with` runs the lifespan, i.e. loads the model
        yield c


class TestOps:
    def test_health_reports_loaded_model(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        meta = json.loads(config.METADATA_FILE.read_text())
        assert body["status"] == "ok"
        assert body["model_version"] == meta["model_version"]
        assert body["model_sha256"] == meta["model_sha256"]

    def test_root_lists_endpoints(self, client):
        assert client.get("/").json()["docs"] == "/docs"

    def test_openapi_documents_predict(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/predict" in paths and "/predict/batch" in paths


class TestPredict:
    def test_high_risk_patient(self, client):
        r = client.post("/predict", json=HIGH)
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] == 1
        assert body["risk_label"] == "high risk"
        assert 0.5 <= body["confidence"] <= 1
        assert body["probability"] == body["confidence"]

    def test_low_risk_patient(self, client):
        body = client.post("/predict", json=LOW).json()
        assert body["prediction"] == 0
        assert body["confidence"] == pytest.approx(1 - body["probability"], abs=1e-4)

    def test_response_carries_model_version_and_request_id(self, client):
        r = client.post("/predict", json=HIGH, headers={"X-Request-ID": "test-123"})
        assert r.json()["request_id"] == "test-123"
        assert r.headers["X-Request-ID"] == "test-123"
        assert r.json()["model_version"] != "unknown"

    def test_nullable_fields_are_imputed(self, client):
        r = client.post("/predict", json={**HIGH, "ca": None, "thal": None})
        assert r.status_code == 200

    def test_omitted_nullable_field_defaults_to_null(self, client):
        payload = {k: v for k, v in HIGH.items() if k != "ca"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_batch(self, client):
        r = client.post("/predict/batch", json={"patients": [HIGH, LOW]})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert [p["prediction"] for p in body["predictions"]] == [1, 0]


class TestValidation:
    @pytest.mark.parametrize("field, value", [
        ("cp", 5),            # not a valid chest-pain code
        ("thal", 4),          # not a valid thal code
        ("sex", 2),
        ("age", 5),           # below range
        ("trestbps", 400),    # above range
        ("oldpeak", -1.0),
        ("age", "sixty"),     # wrong type
    ])
    def test_invalid_values_are_rejected(self, client, field, value):
        r = client.post("/predict", json={**HIGH, field: value})
        assert r.status_code == 422
        assert any(field in err["loc"] for err in r.json()["detail"])

    def test_missing_required_field(self, client):
        payload = {k: v for k, v in HIGH.items() if k != "thalach"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_unknown_field_is_rejected(self, client):
        assert client.post("/predict", json={**HIGH, "weight": 80}).status_code == 422

    def test_empty_and_oversized_batches(self, client):
        assert client.post("/predict/batch", json={"patients": []}).status_code == 422
        too_many = {"patients": [HIGH] * 101}
        assert client.post("/predict/batch", json=too_many).status_code == 422


class TestMetrics:
    def test_prometheus_metrics_exposed(self, client):
        client.post("/predict", json=HIGH)
        text = client.get("/metrics").text
        assert 'heart_predictions_total{risk_label="high risk"}' in text
        assert "heart_prediction_probability_bucket" in text
        assert "heart_model_info{" in text
        assert 'http_requests_total{handler="/predict"' in text
        assert "http_request_duration_seconds" in text

    def test_latency_histogram_resolves_millisecond_requests(self, client):
        client.post("/predict", json=HIGH)
        text = client.get("/metrics").text
        for le in ("0.005", "0.01", "0.025", "0.05"):
            assert (f'http_request_duration_seconds_bucket{{handler="/predict",le="{le}",'
                    'method="POST"}') in text, le


class TestLogging:
    def test_request_and_prediction_are_logged(self, client, caplog):
        with caplog.at_level(logging.INFO, logger="heart.api"):
            client.post("/predict", json=HIGH, headers={"X-Request-ID": "log-check"})
        mine = [r for r in caplog.records if getattr(r, "request_id", None) == "log-check"]
        request_log = next(r for r in mine if r.getMessage() == "request")
        assert request_log.path == "/predict" and request_log.status == 200
        assert request_log.method == "POST" and request_log.duration_ms >= 0
        prediction_log = next(r for r in mine if r.getMessage() == "prediction")
        assert prediction_log.risk_label == "high risk"

    def test_probe_endpoints_are_not_logged_at_info(self, client, caplog):
        with caplog.at_level(logging.INFO, logger="heart.api"):
            client.get("/health")
        assert not [r for r in caplog.records if getattr(r, "path", None) == "/health"]

    def test_formatter_emits_one_json_object(self):
        record = logging.LogRecord("heart.api", logging.INFO, __file__, 1, "request", None, None)
        record.request_id, record.status = "abc", 200
        line = json_formatter().format(record)
        entry = json.loads(line)
        assert entry["message"] == "request" and entry["level"] == "INFO"
        assert entry["request_id"] == "abc" and entry["status"] == 200
        assert "timestamp" in entry and "\n" not in line


class TestFailureModes:
    def test_unexpected_error_returns_500_with_request_id(self, client, monkeypatch):
        from api import main

        def boom(*args, **kwargs):
            raise ValueError("model exploded")

        monkeypatch.setattr(main, "predict_records", boom)
        r = client.post("/predict", json=HIGH, headers={"X-Request-ID": "err-1"})
        assert r.status_code == 500
        assert r.json() == {"detail": "internal server error", "request_id": "err-1"}
        assert "model exploded" not in r.text  # internals are logged, not leaked

    def test_503_when_model_not_loaded(self, client, monkeypatch):
        from api import main

        monkeypatch.setattr(main.state, "pipeline", None)
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=HIGH).status_code == 503


def test_refuses_to_start_with_tampered_model(tmp_path, monkeypatch):
    from api import main

    shutil.copy(config.METADATA_FILE, tmp_path / "metadata.json")
    tampered = tmp_path / "model.joblib"
    tampered.write_bytes(config.MODEL_FILE.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        main.load_model(tampered)
