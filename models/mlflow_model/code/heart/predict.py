"""Load the exported pipeline and turn raw patient records into predictions."""

import json
from pathlib import Path

import joblib
import pandas as pd

from heart import config


def load_pipeline(path: Path = config.MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m heart.train")
    return joblib.load(path)


def load_metadata(path: Path = config.METADATA_FILE) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def predict_records(pipeline, records: list[dict],
                    threshold: float = config.DECISION_THRESHOLD) -> list[dict]:
    """Predict for a list of {feature: value} dicts.

    `probability` is P(disease). `confidence` is the probability of the predicted class,
    so it is always >= 0.5 at the default threshold.
    """
    frame = pd.DataFrame.from_records(records, columns=config.FEATURES)
    proba = pipeline.predict_proba(frame)[:, 1]
    results = []
    for p in proba:
        pred = int(p >= threshold)
        results.append({
            "prediction": pred,
            "risk_label": "high risk" if pred else "low risk",
            "probability": round(float(p), 4),
            "confidence": round(float(p if pred else 1 - p), 4),
        })
    return results
