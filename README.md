# Heart Disease Risk Prediction — MLOps Pipeline

MLOps Assignment 01 (AIMLCZG523). This project builds a classifier on the UCI Heart Disease dataset and serves it as a containerized, monitored FastAPI service on Kubernetes.

> Work in progress. Sections are filled in as each phase is completed.

## Setup (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

Conda alternative: `conda env create -f environment.yml && conda activate heart-mlops`.

## Data

```powershell
python scripts/download_data.py   # UCI Cleveland file -> data/raw/
python -m heart.data              # clean + validate -> data/processed/heart_clean.csv
```

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Cleveland subset (303 rows, 13 features).
The target is binarised (`num` 0 → 0, 1–4 → 1). The 6 missing cells (`ca`, `thal`) are kept as NaN and imputed inside the model pipeline.
EDA: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), with figures in `reports/figures/`.

## Training and experiment tracking

```powershell
python -m heart.train            # full grid search, about 1 min
python -m heart.train --quick    # one candidate per model (CI smoke run)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

- **Pipeline** ([src/heart/features.py](src/heart/features.py)): `FeatureEngineer` → `ColumnTransformer` (median-impute and scale the numeric columns, mode-impute and one-hot encode the categorical ones) → estimator. The whole pipeline is saved as one object, so serving uses exactly the training preprocessing.
- **Models:** Logistic Regression, Random Forest and Gradient Boosting. Each is tuned with `GridSearchCV` (stratified 5-fold, 112 candidates in total) on two feature sets: base features, and base plus the engineered `hr_pct_max = thalach / (220 − age)`.
- **Selection:** highest mean CV ROC-AUC. Candidates within 0.005 of the best count as tied, and the tie goes to the higher CV recall. The 20% hold-out test set is used for reporting only.
- **MLflow** (local `mlflow.db` + `mlruns/`): one parent run per model and feature set, with each grid candidate as a nested run. Logged items: parameters, CV mean and std plus test metrics, ROC, confusion-matrix and feature-importance plots, `cv_results.csv`, and the packaged model with its signature. A `model-selection` run holds the comparison. The champion is registered as `heart-disease-classifier@champion`.
- **Export:** `models/model.joblib` (the full pipeline) and `models/metadata.json` (version, parameters, metrics, data hash, git commit).

| Model | Features | CV ROC-AUC | CV recall | Test ROC-AUC | Test recall |
|---|---|---|---|---|---|
| **Logistic Regression** (champion) | base | **0.912 ± 0.017** | 0.792 | 0.962 | 0.929 |
| Logistic Regression | + hr_pct_max | 0.909 ± 0.012 | 0.783 | 0.955 | 0.893 |
| Random Forest | base | 0.898 ± 0.029 | 0.774 | 0.957 | 0.929 |
| Random Forest | + hr_pct_max | 0.897 ± 0.027 | 0.756 | 0.961 | 0.929 |
| Gradient Boosting | base | 0.884 ± 0.027 | 0.756 | 0.960 | 0.929 |
| Gradient Boosting | + hr_pct_max | 0.884 ± 0.026 | 0.765 | 0.958 | 0.929 |

## Model packaging and reproducibility

| Artifact | Format | Used by |
|---|---|---|
| `models/model.joblib` | Full sklearn `Pipeline` (preprocessing + model) | API, `scripts/predict.py` |
| `models/mlflow_model/` | MLflow pyfunc model, with the `heart` package bundled (`code/`), signature and input example | `mlflow.pyfunc.load_model("models/mlflow_model")`, `mlflow models serve` |
| `models/metadata.json` | Model card: version, family, parameters, CV and test metrics, `model_sha256`, `data_sha256`, git commit | API `/health`, reproducibility tests |

Reproducibility is enforced by tests, not just documented:
- **Pinned environment:** every package in `requirements.txt` is pinned with `==`, including mlflow's SQLAlchemy and Alembic, which mlflow itself leaves unpinned. `.python-version` and `environment.yml` (conda) target Python 3.11.
- **Fixed seeds:** the train/test split, CV folds and every estimator use `random_state=42`.
- **`tests/test_reproducibility.py`** checks:
  - the model file's checksum matches `metadata.json`;
  - the training data hash is unchanged;
  - rebuilding the model from `metadata.json` alone gives identical predictions;
  - the recorded test metrics recompute exactly;
  - the MLflow-format model agrees with the joblib model.

## Inference (CLI)

```powershell
python scripts/predict.py samples/patient_high_risk.json
python scripts/predict.py samples/patients_batch.json        # list input, includes a missing value
python scripts/predict.py data/processed/heart_clean.csv --limit 5
```

Each prediction returns `prediction` (0/1), `risk_label`, `probability` (P(disease)) and `confidence` (the probability of the predicted class).

## Serving API (FastAPI)

```powershell
uvicorn api.main:app --reload        # local dev; Swagger UI at http://127.0.0.1:8000/docs
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | One patient (JSON) → `prediction`, `risk_label`, `probability`, `confidence`, `model_version`, `request_id` |
| `/predict/batch` | POST | `{"patients": [...]}`, 1–100 patients |
| `/health` | GET | Liveness/readiness: model name, version, family, sha256, uptime |
| `/metrics` | GET | Prometheus: HTTP request count and latency, `heart_predictions_total{risk_label}`, `heart_prediction_probability` histogram, `heart_model_info` |
| `/docs` | GET | OpenAPI / Swagger UI |

- **Validation:** Pydantic enforces UCI codes (for example `cp ∈ {1,2,3,4}`), clinical ranges and required fields, and rejects unknown fields with a 422. `ca` and `thal` may be `null`; they are imputed by the pipeline.
- **Logging:** one JSON line per request (`request_id`, method, path, status, duration_ms) and per prediction. `X-Request-ID` is accepted or generated, then echoed back. Probe endpoints log at DEBUG only.
- **Safety:** on startup the API checks the model's sha256 against `metadata.json` and refuses to serve a mismatched file. Unhandled errors return a generic 500 with the request ID; details go to the log only.

Example:
```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data "@samples/patient_high_risk.json"
# {"prediction":1,"risk_label":"high risk","probability":0.9915,"confidence":0.9915,"model_version":"2","request_id":"..."}
```

## Docker

```powershell
docker build -t heart-api:latest .
docker run -d --name heart-api -p 8000:8000 heart-api:latest
python scripts/smoke_test.py --url http://127.0.0.1:8000     # 7 end-to-end checks, exits non-zero on failure
docker logs heart-api                                        # structured JSON logs
```

- **Multi-stage build:** dependencies are installed into a venv in the builder stage, and the runtime stage copies only the venv, `src/heart`, `api` and the model files. `.dockerignore` is an allow-list.
- **Serving dependencies only:** `requirements-serve.txt` has no mlflow, matplotlib or jupyter. Its pins must match `requirements.txt`, which a test enforces. pip, setuptools and the libraries' bundled test suites are removed. The image is about 600 MB, mostly scipy, scikit-learn, pandas and numpy.
- **Runs as non-root** (uid 10001), with a Docker `HEALTHCHECK` on `/health`. uvicorn is PID 1 (exec form), so `docker stop` shuts it down gracefully.

## Kubernetes deployment (Docker Desktop)

Prerequisite: Docker Desktop → Settings → Kubernetes → **Enable Kubernetes** (context `docker-desktop`).

```powershell
.\scripts\k8s_deploy.ps1          # build, load image into the cluster, apply, wait for rollout, smoke test
# or step by step:
kubectl apply -k k8s/
kubectl -n heart-mlops rollout status deployment/heart-api
kubectl -n heart-mlops get pods,svc
python scripts/smoke_test.py --url http://localhost
```

The API is then served at **http://localhost** (`/predict`, `/health`, `/metrics`, `/docs`) through the LoadBalancer Service, which Docker Desktop binds to localhost.

| Manifest | Purpose |
|---|---|
| `namespace.yaml` | `heart-mlops` namespace |
| `configmap.yaml` | `LOG_LEVEL`, `MODEL_PATH` passed to the pods as environment variables |
| `deployment.yaml` | 2 replicas; rolling update with `maxUnavailable: 0`; startup, readiness and liveness probes on `/health`; CPU and memory requests and limits; non-root, read-only root filesystem, all capabilities dropped, seccomp `RuntimeDefault`; `preStop` drain; Prometheus scrape annotations |
| `service.yaml` | `LoadBalancer`, port 80 → container port 8000 |
| `pdb.yaml` | PodDisruptionBudget `minAvailable: 1` |
| `kustomization.yaml` | `kubectl apply -k k8s/` deploys everything |

**Notes and verified behaviour**
- **Images:** Docker Desktop runs Kubernetes in *kind* mode, where the node has its own containerd image store. The deploy script therefore imports the locally built image into the node (`docker save | ctr images import`, which is what `kind load` does). The image tag is pinned (`heart-api:1.0.0`) with `imagePullPolicy: IfNotPresent`, so no registry is needed.
- **Load balancing:** 20 requests were split 14/6 across the two pods.
- **Self-healing:** deleting a pod brings the Deployment back to 2/2 Ready in about 6 seconds.
- **Zero-downtime rollout:** `kubectl rollout restart` under continuous traffic. Without a `preStop` hook, 2 of 161 requests failed (a terminating pod still received traffic until kube-proxy caught up). With a 5 s `preStop` drain, 4 of 5 rounds had **0 failures** out of about 160 requests each.
- **CI validation:** manifests are schema-validated in CI with `kubeconform -strict`, and `tests/test_k8s_manifests.py` checks the production rules (probes, resources, security context, selectors, pinned image).

## Monitoring and logging (Prometheus + Grafana)

```powershell
.\scripts\k8s_deploy.ps1 -Monitoring       # API + monitoring stack
# or: kubectl apply -k k8s/monitoring/
python scripts/load_test.py --duration 300 --rps 8                       # realistic mixed traffic
python scripts/load_test.py --profile high-risk --duration 900           # drives the drift alert
```

| UI | URL | Notes |
|---|---|---|
| Grafana | http://localhost:3000 | The dashboard opens as the home page; anonymous read-only access. Admin login `admin` / `heart-mlops-demo` (local demo only) |
| Prometheus | http://localhost:9090 | `/targets` shows both API pods; `/alerts` shows the rules |

**Logging:** the API writes one JSON line per request (request ID, path, status, duration) and per prediction (label, probability), viewable with `kubectl -n heart-mlops logs -l app.kubernetes.io/name=heart-api`.

**Metrics pipeline:** Prometheus discovers the API pods through the Kubernetes API, using the `prometheus.io/*` annotations. RBAC is limited to reading pods in `heart-mlops`. It scrapes every 10 s.

**Dashboard: *Heart Disease API · Service & Model Monitoring*** (12 panels)
- **Headline stats:** requests/s, `/predict` p95 latency, 5xx rate, pods up, serving model version, high-risk share over 30 minutes.
- **Traffic:** request rate by endpoint and status (4xx = validation rejects), and `/predict` latency at p50, p95 and p99.
- **Model behaviour:** predictions per minute by risk class; high-risk share compared with the **46% training base rate** (an output-drift signal); the distribution of predicted P(disease).
- **Load balancing:** prediction requests per pod.

**Alert rules** ([alerts.yml](k8s/monitoring/alerts.yml), all checked by `promtool`)

| Alert | Condition | Severity |
|---|---|---|
| HeartApiDown | No pod scraped successfully for 1 min | critical |
| HeartApiReplicasDegraded | Fewer than 2 replicas up for 2 min | warning |
| HeartApiHighErrorRate | More than 5% of requests are 5xx for 2 min | critical |
| HeartApiHighLatency | `/predict` p95 above 500 ms for 5 min | warning |
| HeartApiManyInvalidRequests | More than 25% of requests rejected (4xx) for 10 min | info |
| HeartPredictionDrift | High-risk share outside 20–80% (base rate 46%) for 5 min | warning |

Verified live: scaling the API to 1 replica moved `HeartApiReplicasDegraded` to pending after 22 s and firing after 142 s; it resolved 22 s after scaling back to 2.

**Findings from load testing (8 req/s)**
- **Load balancing:** Kubernetes Services balance **connections**, not requests. With one shared keep-alive connection, every request went to a single pod. With one connection per client thread, traffic split about 60/40.
- **Latency breakdown:** a prediction takes about 7 ms in the pod (loopback) and about 9 ms through the Service inside the cluster. Requests through Docker Desktop's `localhost` LoadBalancer take about 60 ms, which is a local networking artefact, and that delay also appears in the server-side histogram (the timer starts when the headers arrive).
- **Latency buckets:** the instrumentator's default buckets (0.1, 0.5, 1 s) cannot resolve millisecond requests, so the API configures 5 ms to 2.5 s buckets.
- **CPU limit:** 500m throttled about 12% of CFS periods under load; 1 CPU removed the throttling.

## CI/CD (GitHub Actions)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request. Each job needs the previous one, so any failure stops the pipeline:

```
lint ──▶ test ──▶ train ──▶ docker
```

| Job | What it does | Fails when | Artifacts / summary |
|---|---|---|---|
| **lint** | `ruff check` (inline annotations on the PR), `hadolint` on the Dockerfile, `kubeconform -strict` on the API and monitoring manifests, `promtool` on the Prometheus config and alert rules | Lint errors | n/a |
| **test** | Installs pinned dependencies, `pip check`, then pytest with coverage | Any test fails, or coverage is below 85% | `test-reports` (JUnit, coverage XML/HTML); coverage table in the job summary |
| **train** | Re-downloads the UCI data and checks the cleaned CSV is byte-identical; runs the full grid search with MLflow; runs the **quality gate** against the committed model | The data differs, training errors, or test ROC-AUC drops by more than 0.02 or falls below 0.85 | `trained-model` (joblib + MLflow model + metadata), `mlflow-tracking` (mlflow.db, mlruns, plots, logs); metrics table in the job summary |
| **docker** | Builds the image **with the model trained in this run**, runs it, then runs `scripts/smoke_test.py` | The build or any smoke check fails (container logs are printed) | Image pushed to `ghcr.io/suchendrasharma/heart-api` (`latest`, `sha-<commit>`) on `main` only |

All steps run under `bash -eo pipefail`, so a failure inside a piped command (for example `train | tee`) still fails the step.

## Tests

```powershell
pytest                    # all tests, about 2 min
pytest -m "not slow"      # skip the end-to-end training test
pytest --cov              # with coverage report
```

| File | Covers |
|---|---|
| `test_data.py` | Loading, `?` → NaN, binary target, dtypes, de-duplication, schema validation |
| `test_download.py` | Download fallback, skip if present, failure exit (network stubbed) |
| `test_features.py` | Engineered feature, NaN coercion, imputation, scaling, one-hot encoding, unseen categories |
| `test_model.py` | Metrics, champion selection rule, determinism, save/load, prediction schema, end-to-end training |
| `test_reproducibility.py` | Committed artifacts: checksums, data hash, rebuild equality, metric recomputation |
| `test_predict_cli.py` | CLI on the sample files, CSV input, error handling |
| `test_api.py` | Endpoints, validation (422s), batch limits, request IDs, JSON logs, Prometheus metrics, 500/503 paths, tampered-model refusal |
| `test_quality_gate.py` | CI quality gate: pass/fail decisions, absolute floor, exit codes, Markdown report |
| `test_k8s_manifests.py` | Kubernetes rules: kustomization completeness, probes, resources, security context, zero-downtime settings, selectors, scrape annotations |
| `test_monitoring.py` | Prometheus scrape config and RBAC, alert rules (severity, summary, only exported metrics, drift bounds), dashboard queries and layout, load-generator request mix |
| `test_api_packaging.py` | Serve pins match the main pins, no training stack in the image, non-root user, health check, `.dockerignore` allow-list |

## Project layout

| Path | Purpose |
|---|---|
| `scripts/` | Data download and load-test scripts |
| `data/raw`, `data/processed` | Raw UCI files (not committed) and the cleaned dataset |
| `notebooks/` | EDA and inference demo |
| `src/heart/` | Data, feature, training, evaluation and prediction code |
| `api/` | FastAPI serving app |
| `models/` | Exported model (joblib + MLflow format) and metadata |
| `samples/` | Example patient JSON inputs for the CLI and API |
| `tests/` | Pytest unit tests |
| `k8s/` | Kubernetes manifests (API and monitoring) |
| `.github/workflows/` | CI pipeline |
| `screenshots/`, `reports/` | Evidence and final report |
