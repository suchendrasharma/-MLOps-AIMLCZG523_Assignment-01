# Heart disease risk API: FastAPI + the exported sklearn pipeline.
#   docker build -t heart-api:latest .
#   docker run --rm -p 8000:8000 heart-api:latest
ARG PYTHON_VERSION=3.11

# ---- build stage: resolve and install dependencies into an isolated venv ----
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-serve.txt /tmp/
# Trim after install: pip/setuptools and the libraries' bundled test suites are never
# used at runtime (about 100 MB). The container smoke test proves nothing needed is lost.
RUN pip install -r /tmp/requirements-serve.txt \
 && pip uninstall -y pip setuptools \
 && find /opt/venv/lib -depth -type d -name tests -path "*/site-packages/*" -exec rm -rf {} +

# ---- runtime stage: only the venv, the code and the model ----
FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (MLOps assignment)" \
      org.opencontainers.image.source="https://github.com/suchendrasharma/-MLOps-AIMLCZG523_Assignment-01"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/heart ./src/heart
COPY api ./api
COPY models/model.joblib models/metadata.json ./models/

USER 10001
EXPOSE 8000

# slim images have no curl; use the Python stdlib for the health probe
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# exec form: uvicorn is PID 1 and receives SIGTERM directly for a graceful shutdown
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
