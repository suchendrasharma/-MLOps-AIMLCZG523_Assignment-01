"""Command-line inference with the exported model.

    python scripts/predict.py samples/patient_high_risk.json
    python scripts/predict.py samples/patients_batch.json
    python scripts/predict.py data/processed/heart_clean.csv --limit 5

Input: a JSON object (one patient), a JSON list (several), or a CSV with the feature
columns (extra columns such as `target` are ignored). Output: JSON on stdout.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from heart import config
from heart.predict import load_metadata, load_pipeline, predict_records


def read_records(path: Path, limit: int | None = None) -> list[dict]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        records = frame.astype(object).where(frame.notna(), None).to_dict("records")
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        records = [data] if isinstance(data, dict) else data
    else:
        raise ValueError(f"unsupported input type: {path.suffix} (use .json or .csv)")
    missing = {f for r in records for f in config.FEATURES if f not in r}
    if missing:
        raise ValueError(f"input is missing fields: {sorted(missing)}")
    return records[:limit] if limit else records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict heart disease risk")
    parser.add_argument("input", type=Path, help="JSON or CSV file with patient records")
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    parser.add_argument("--limit", type=int, default=None, help="only the first N records")
    args = parser.parse_args(argv)

    try:
        records = read_records(args.input, args.limit)
        pipeline = load_pipeline(args.model)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    meta = load_metadata(args.model.with_name("metadata.json"))
    output = {
        "model": f"{meta.get('model_name', 'unknown')} v{meta.get('model_version', '?')}",
        "predictions": predict_records(pipeline, records),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
