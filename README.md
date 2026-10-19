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

## Data

```powershell
python scripts/download_data.py   # UCI Cleveland file -> data/raw/
python -m heart.data              # clean + validate -> data/processed/heart_clean.csv
```

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Cleveland subset (303 rows, 13 features).
The target is binarised (`num` 0 → 0, 1–4 → 1). The 6 missing cells (`ca`, `thal`) are kept as NaN and imputed inside the model pipeline.
EDA: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), with figures in `reports/figures/`.

## Project layout

| Path | Purpose |
|---|---|
| `scripts/` | Data download and load-test scripts |
| `data/raw`, `data/processed` | Raw UCI files (not committed) and the cleaned dataset |
| `notebooks/` | EDA and inference demo |
| `src/heart/` | Data, feature, training, evaluation and prediction code |
| `api/` | FastAPI serving app |
| `models/` | Exported model pipeline and metadata |
| `tests/` | Pytest unit tests |
| `k8s/` | Kubernetes manifests (API and monitoring) |
| `.github/workflows/` | CI pipeline |
| `screenshots/`, `reports/` | Evidence and final report |
