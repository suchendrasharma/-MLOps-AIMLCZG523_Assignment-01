"""Collect every number the report quotes into reports/build/facts.json.

    python reports/build/collect_facts.py

The report generator (build_report.js) reads only this file, so the report cannot drift
from the actual artefacts: metrics come from models/metadata.json and the MLflow
model-selection run, dataset statistics from the cleaned CSV, test counts from pytest.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

from heart import config
from heart.data import load_clean

OUT = Path(__file__).with_name("facts.json")


def model_comparison() -> list[dict]:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    run = client.search_runs([exp.experiment_id], "attributes.run_name = 'model-selection'",
                             order_by=["attributes.start_time DESC"], max_results=1)[0]
    df = pd.read_csv(client.download_artifacts(run.info.run_id, "model_comparison.csv"))
    cols = ["model", "features", "cv_roc_auc", "cv_roc_auc_std", "cv_recall", "test_roc_auc",
            "test_accuracy", "test_precision", "test_recall", "test_f1"]
    rows = df[cols].round(4).to_dict("records")
    n_runs = len(client.search_runs([exp.experiment_id], max_results=5000))
    return rows, n_runs, run.info.run_id


def dataset_stats() -> dict:
    df = load_clean()
    rates = {}
    for col in ("cp", "thal", "ca", "exang", "sex"):
        g = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        rates[col] = {str(int(k)): {"rate": round(v["mean"], 3), "n": int(v["size"])}
                      for k, v in g.iterrows()}
    return {
        "rows": len(df), "features": len(config.FEATURES),
        "positive_rate": round(float(df[config.TARGET].mean()), 4),
        "missing": {k: int(v) for k, v in df.isna().sum().items() if v},
        "category_rates": rates,
    }


def test_counts() -> dict:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         capture_output=True, text=True, cwd=config.PROJECT_ROOT).stdout
    total = int(re.search(r"(\d+) tests? collected", out).group(1))
    per_file = {}
    for line in out.splitlines():
        if "::" in line:
            f = line.split("::")[0].split("/")[-1]
            per_file[f] = per_file.get(f, 0) + 1
    return {"total": total, "files": per_file}


def main() -> None:
    meta = json.loads(config.METADATA_FILE.read_text())
    comparison, n_runs, selection_run = model_comparison()
    facts = {
        "repo_url": "https://github.com/suchendrasharma/-MLOps-AIMLCZG523_Assignment-01",
        "metadata": meta,
        "comparison": comparison,
        "mlflow": {"n_runs": n_runs, "selection_run": selection_run,
                   "experiment": config.EXPERIMENT_NAME,
                   "registered_model": config.REGISTERED_MODEL_NAME},
        "dataset": dataset_stats(),
        "tests": test_counts(),
        "image_tag": re.search(r"image:\s*(heart-api:\S+)",
                               (config.PROJECT_ROOT / "k8s/deployment.yaml").read_text()).group(1),
        # Measured during the verification runs documented in the README (phases I-J).
        "measured": {
            "rollout_without_prestop": "2 of 161 requests failed",
            "rollout_with_prestop": "0 failures in 4 of 5 rounds (~160 requests each)",
            "load_test_5400": {"total": 5400, "ok": 5138, "rejected_422": 262, "errors": 0,
                               "note": "included a live rolling update to a new image tag"},
            "latency_in_pod_ms": 7.0, "latency_in_cluster_ms": 8.8,
            "latency_via_localhost_lb_ms": 60,
            "cpu_throttled_pct_500m": 12,
            "pod_split": "471 / 312",
            "self_heal_seconds": 6,
            "alert_pending_s": 22, "alert_firing_s": 142, "alert_resolved_s": 22,
            "ci_run_minutes": 5,
        },
    }
    OUT.write_text(json.dumps(facts, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({facts['tests']['total']} tests, {n_runs} MLflow runs)")


if __name__ == "__main__":
    main()
