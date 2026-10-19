"""Heart disease risk API.

    uvicorn api.main:app --reload            # local dev, docs at http://127.0.0.1:8000/docs

Endpoints: POST /predict, POST /predict/batch, GET /health, GET /metrics, GET /
Config (env): MODEL_PATH (default models/model.joblib), LOG_LEVEL (default INFO).
"""

import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from api.logging_config import configure_logging
from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)
from heart import config
from heart.predict import load_pipeline, predict_records

logger = configure_logging()

# Model-level metrics, on top of the HTTP metrics from the instrumentator.
PREDICTIONS = Counter(
    "heart_predictions_total", "Predictions served, by predicted class", ["risk_label"])
PROBABILITY = Histogram(
    "heart_prediction_probability", "Distribution of predicted P(disease)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MODEL_INFO = Gauge(
    "heart_model_info", "Currently loaded model (value is always 1)", ["version", "family"])

QUIET_PATHS = {"/health", "/metrics"}  # probed every few seconds; log at DEBUG only


class ModelState:
    pipeline = None
    metadata: dict = {}
    started_at = time.time()


state = ModelState()


def load_model(model_path: Path) -> None:
    """Load the pipeline and refuse to serve a file that does not match its metadata."""
    metadata_path = model_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    expected = metadata.get("model_sha256")
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if expected and expected != actual:
        raise RuntimeError(f"model checksum mismatch: {model_path} is not the exported model")
    state.pipeline = load_pipeline(model_path)
    state.metadata = {**metadata, "model_sha256": actual}
    MODEL_INFO.labels(version=metadata.get("model_version", "unknown"),
                      family=metadata.get("model_family", "unknown")).set(1)
    logger.info("model loaded", extra={"model_path": str(model_path),
                                       "model_version": metadata.get("model_version"),
                                       "model_sha256": actual[:12]})


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(Path(os.environ.get("MODEL_PATH", config.MODEL_FILE)))
    state.started_at = time.time()
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart disease risk from 13 clinical features (UCI Cleveland data).",
    version="1.0.0",
    lifespan=lifespan,
)
# The per-handler latency histogram defaults to buckets (0.1, 0.5, 1) s, which makes every
# percentile of a ~10 ms endpoint an interpolation artefact; use buckets that resolve it.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5)
Instrumentator(excluded_handlers=["/metrics"]).instrument(
    app, latency_lowr_buckets=LATENCY_BUCKETS).expose(
    app, endpoint="/metrics", include_in_schema=False)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error", extra={"request_id": request_id,
                                                    "path": request.url.path})
        response = JSONResponse({"detail": "internal server error",
                                 "request_id": request_id}, status_code=500)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    log = logger.debug if request.url.path in QUIET_PATHS else logger.info
    log("request", extra={
        "request_id": request_id, "method": request.method, "path": request.url.path,
        "status": response.status_code, "duration_ms": duration_ms,
        "client": request.client.host if request.client else None,
    })
    return response


def model_version() -> str:
    return str(state.metadata.get("model_version", "unknown"))


def run_model(patients: list[PatientFeatures], request_id: str) -> list[dict]:
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    results = predict_records(state.pipeline, [p.model_dump() for p in patients])
    for res in results:
        PREDICTIONS.labels(risk_label=res["risk_label"]).inc()
        PROBABILITY.observe(res["probability"])
        logger.info("prediction", extra={"request_id": request_id, **res})
    return results


@app.get("/", include_in_schema=False)
def root():
    return {"service": app.title, "model_version": model_version(),
            "docs": "/docs", "health": "/health", "metrics": "/metrics"}


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(
        status="ok",
        model_name=state.metadata.get("model_name", "unknown"),
        model_version=model_version(),
        model_family=state.metadata.get("model_family", "unknown"),
        model_sha256=state.metadata["model_sha256"],
        uptime_seconds=round(time.time() - state.started_at, 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(patient: PatientFeatures, request: Request):
    """Predict heart disease risk for one patient."""
    (result,) = run_model([patient], request.state.request_id)
    return PredictionResponse(**result, model_version=model_version(),
                              request_id=request.state.request_id)


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, request: Request):
    """Predict for up to 100 patients in one call."""
    results = run_model(batch.patients, request.state.request_id)
    return BatchResponse(predictions=results, count=len(results),
                         model_version=model_version(), request_id=request.state.request_id)
