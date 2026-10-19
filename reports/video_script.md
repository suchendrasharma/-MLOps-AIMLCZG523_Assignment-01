# Pipeline video: recording script (about 7 minutes)

A walkthrough of the whole pipeline for the assignment video. Each segment lists what to show and a short narration. Prepare the stack once beforehand so nothing waits on a download during recording.

## Before recording (not filmed)

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\k8s_deploy.ps1 -Monitoring                         # API + Prometheus + Grafana up
mlflow ui --backend-store-uri sqlite:///mlflow.db           # in a second terminal
python scripts/load_test.py --duration 900 --rps 6          # in a third terminal, keeps dashboards live
```

Open browser tabs: the GitHub repo, the latest Actions run, http://127.0.0.1:5000, http://localhost/docs, http://localhost:3000 and http://localhost:9090/alerts. Use a large terminal font.

## 0:00–0:30 · Introduction
- **Show:** the README in the GitHub repo, then `reports/figures/architecture.png`.
- **Say:** "This project predicts heart disease risk from 13 clinical features and takes it all the way to a monitored API on Kubernetes. The diagram shows the four layers: data, training and tracking, CI/CD, and serving with monitoring."

## 0:30–1:15 · Data and EDA
- **Show:** `python scripts/download_data.py` and `python -m heart.data`, then scroll through `notebooks/01_eda.ipynb` (class balance, disease rate by category, correlation heatmap, key findings).
- **Say:** "The script downloads the UCI Cleveland file and validates every code against the code book. Only 6 cells are missing; they are imputed inside the pipeline so CV scores stay honest. The strongest signals are thal, the number of vessels and chest pain type."

## 1:15–2:15 · Training and MLflow
- **Show:** `python -m heart.train --quick` running (or the output of a full run). In MLflow: the experiment, the champion run (parameters, metrics, tags), its artefacts (plots, model) and the registry with `@champion`.
- **Say:** "Three model families are tuned with stratified 5-fold grid search on two feature sets, so the engineered feature is a real ablation. Every candidate is a nested MLflow run. Logistic regression wins on CV ROC-AUC 0.912, and the engineered feature doesn't help, which is an honest negative result."

## 2:15–3:00 · Reproducibility and tests
- **Show:** `models/metadata.json` (checksums, metrics), then `pytest -m "not slow" -q`.
- **Say:** "The model ships as a joblib pipeline and an MLflow model, with a metadata card. Tests prove the committed model can be rebuilt exactly from the metadata. There are 131 tests in total."

## 3:00–3:50 · CI/CD
- **Show:** the GitHub Actions run graph (lint → test → train → docker), the job summary (coverage, metrics, quality-gate table) and the artefacts list. Optionally, a red run from a branch with a broken test.
- **Say:** "Every push runs lint, tests with a coverage gate, a fresh-data check, full training with a quality gate against the committed model, then builds the image with the freshly trained model and smoke-tests it. Any failure stops the pipeline."

## 3:50–4:40 · Docker and the API
- **Show:** `docker build -t heart-api:local .`, `docker run -d -p 8000:8000 heart-api:local`, then `python scripts/smoke_test.py`. Then Swagger at /docs: try `/predict` with the example payload.
- **Say:** "The image is multi-stage, runs as non-root and has a health check. The API validates input against the UCI code book and returns the prediction with its confidence."

## 4:40–5:40 · Kubernetes
- **Show:** `kubectl -n heart-mlops get all,pdb -o wide`, then a curl to `http://localhost/predict`. Delete a pod and run `kubectl get pods -w` to show it being replaced.
- **Say:** "Two replicas sit behind a LoadBalancer, with probes, resource limits and a hardened security context. Kubernetes replaces a deleted pod in a few seconds, and rolling updates under load drop no requests thanks to the preStop drain."

## 5:40–6:40 · Monitoring and logging
- **Show:** the Grafana dashboard with live traffic, then Prometheus `/alerts`, then `kubectl -n heart-mlops logs -l app.kubernetes.io/name=heart-api --tail=5`.
- **Say:** "Prometheus discovers the pods automatically. The dashboard covers traffic, latency, errors and replicas, plus model behaviour: the risk-class mix and the high-risk share against the 46% training base rate, which drives a drift alert. Every request is logged as JSON with a request ID."

## 6:40–7:00 · Wrap-up
- **Show:** the report PDF and the repo.
- **Say:** "Everything is reproducible from the repository: data, model, image, deployment and dashboards. The report covers the design decisions and the issues found along the way."
