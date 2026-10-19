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
