"""Central configuration: paths, column definitions and seeds."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"

MODEL_FILE = MODELS_DIR / "model.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

# MLflow: SQLite backend (needed for the model registry) + local artifact folder.
# Both can be overridden via env vars (CI and tests point them at a temp dir).
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_ROOT = os.environ.get(
    "HEART_MLFLOW_ARTIFACT_ROOT", (PROJECT_ROOT / "mlruns").as_uri()
)
EXPERIMENT_NAME = "heart-disease-classification"
REGISTERED_MODEL_NAME = "heart-disease-classifier"

# Column order of the UCI "processed" files (no header row in the source).
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed codes per categorical column (UCI documentation), used for validation.
CATEGORY_DOMAINS = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}

# Human-readable labels, used only for EDA plots.
CATEGORY_LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "fbs": {0: "<=120 mg/dl", 1: ">120 mg/dl"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}
