"""Tune and compare models, track every run in MLflow, and export the champion pipeline.

    python -m heart.train              # full hyper-parameter grids
    python -m heart.train --quick      # one candidate per model (CI / smoke test)
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs

Experiment design
- Stratified 80/20 train/test split. The test set is used only to report final numbers.
- Every model family is tuned with GridSearchCV (stratified 5-fold CV) twice: with the
  base features and with the engineered `hr_pct_max` feature (an ablation).
- Champion = highest mean CV ROC-AUC. Candidates within 0.005 AUC of the best are treated
  as tied, and the tie goes to the higher CV recall (a missed disease case costs more
  than a false alarm). Selection never looks at the test set.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heart import config, plotting
from heart.data import load_clean, validate
from heart.evaluate import (
    CV_SCORERS,
    METRICS,
    compute_metrics,
    plot_confusion,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc,
)
from heart.features import build_pipeline

SEED = config.RANDOM_STATE
AUC_TIE_TOLERANCE = 0.005
FEATURE_VARIANTS = {"base": False, "engineered": True}

MODELS = {
    "logistic_regression": (
        LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED),
        {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {
            "n_estimators": [200, 500],
            "max_depth": [None, 5, 8],
            "min_samples_leaf": [1, 4],
            "max_features": ["sqrt", 0.5],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {
            "n_estimators": [100, 300],
            "learning_rate": [0.03, 0.1],
            "max_depth": [2, 3],
            "subsample": [0.8, 1.0],
        },
    ),
}
QUICK_GRIDS = {
    "logistic_regression": {"C": [1.0], "penalty": ["l2"], "class_weight": [None]},
    "random_forest": {"n_estimators": [100], "max_depth": [5], "min_samples_leaf": [1],
                      "max_features": ["sqrt"]},
    "gradient_boosting": {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [2],
                          "subsample": [1.0]},
}


def split(df: pd.DataFrame):
    X, y = df[config.FEATURES], df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=SEED)


def data_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=config.PROJECT_ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def tune(estimator, grid: dict, add_hr_ratio: bool, X, y) -> GridSearchCV:
    search = GridSearchCV(
        build_pipeline(clone(estimator), add_hr_ratio),
        param_grid={f"model__{k}": v for k, v in grid.items()},
        scoring=CV_SCORERS,
        refit="roc_auc",
        cv=StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    return search.fit(X, y)


def cv_scores(search: GridSearchCV, index: int) -> dict:
    res = search.cv_results_
    scores = {}
    for m in METRICS:
        scores[f"cv_{m}"] = float(res[f"mean_test_{m}"][index])
        scores[f"cv_{m}_std"] = float(res[f"std_test_{m}"][index])
    return scores


def strip_prefix(params: dict) -> dict:
    return {k.removeprefix("model__"): v for k, v in params.items()}


def log_candidates(search: GridSearchCV) -> None:
    """One nested MLflow run per grid candidate, so every configuration is traceable."""
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"candidate-{i:02d}", nested=True):
            mlflow.log_params(strip_prefix(params))
            mlflow.log_metrics(cv_scores(search, i))
            mlflow.log_metric("rank_cv_roc_auc", int(res["rank_test_roc_auc"][i]))


def log_figure(fig: plt.Figure, name: str) -> None:
    mlflow.log_figure(fig, f"plots/{name}")
    plt.close(fig)


def select_champion(summary: pd.DataFrame, tol: float = AUC_TIE_TOLERANCE) -> int:
    """Row label of the champion: best CV ROC-AUC, ties (within tol) broken by CV recall."""
    contenders = summary[summary["cv_roc_auc"] >= summary["cv_roc_auc"].max() - tol]
    return contenders.sort_values(["cv_recall", "cv_roc_auc"], ascending=False).index[0]


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME,
                                 artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def train_one(name, estimator, grid, variant, X_train, X_test, y_train, y_test, tags) -> dict:
    """Tune one model family on one feature set inside its own (parent) MLflow run."""
    with mlflow.start_run(run_name=f"{name}-{variant}") as run:
        mlflow.set_tags({**tags, "model_family": name, "feature_set": variant})
        search = tune(estimator, grid, FEATURE_VARIANTS[variant], X_train, y_train)
        log_candidates(search)

        best = search.best_estimator_
        cv = cv_scores(search, search.best_index_)
        proba = best.predict_proba(X_test)[:, 1]
        test = {f"test_{k}": v for k, v in compute_metrics(y_test, proba).items()}

        mlflow.log_params({
            **strip_prefix(search.best_params_),
            "model": name, "feature_set": variant, "cv_folds": config.CV_FOLDS,
            "n_candidates": len(search.cv_results_["params"]),
            "threshold": config.DECISION_THRESHOLD,
        })
        mlflow.log_metrics({**cv, **test})
        log_figure(plot_roc(y_test, proba, name), "roc_curve.png")
        log_figure(plot_confusion(y_test, proba), "confusion_matrix.png")
        log_figure(plot_feature_importance(best), "feature_importance.png")
        mlflow.log_text(pd.DataFrame(search.cv_results_).to_csv(index=False), "cv_results.csv")

        X_example = X_train.head(5).astype(float)
        mlflow.sklearn.log_model(
            best, artifact_path="model",
            signature=infer_signature(X_example, best.predict_proba(X_example)[:, 1]),
            input_example=X_example,
            code_paths=[str(config.PROJECT_ROOT / "src" / "heart")],  # custom transformer
            pip_requirements=[f"scikit-learn=={sklearn.__version__}",
                              f"pandas=={pd.__version__}", "numpy", "joblib"],
        )
        print(f"  {name:<20} {variant:<10} cv_auc={cv['cv_roc_auc']:.4f} "
              f"(±{cv['cv_roc_auc_std']:.3f})  test_auc={test['test_roc_auc']:.4f}")
        return {"model": name, "features": variant, "run_id": run.info.run_id,
                "best_params": strip_prefix(search.best_params_), **cv, **test,
                "pipeline": best, "holdout_proba": proba}


def register(run_id: str) -> str:
    version = mlflow.register_model(f"runs:/{run_id}/model", config.REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, "champion", version.version)
    return str(version.version)


def export(champion: dict, model_version: str, tags: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion["pipeline"], config.MODEL_FILE)
    metadata = {
        "model_name": config.REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "model_family": champion["model"],
        "feature_set": champion["features"],
        "best_params": champion["best_params"],
        "decision_threshold": config.DECISION_THRESHOLD,
        "input_features": config.FEATURES,
        "cv_metrics": {k: round(champion[k], 4) for k in champion if k.startswith("cv_")},
        "test_metrics": {k: round(champion[k], 4) for k in champion if k.startswith("test_")},
        "mlflow_run_id": champion["run_id"],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **tags,
    }
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2))


def run(quick: bool = False, do_register: bool = True) -> dict:
    plotting.apply_style()
    setup_mlflow()
    df = load_clean()
    validate(df)
    X_train, X_test, y_train, y_test = split(df)
    tags = {"data_sha256": data_fingerprint(df), "git_commit": git_commit(),
            "sklearn_version": sklearn.__version__, "n_train": str(len(X_train)),
            "n_test": str(len(X_test)), "mode": "quick" if quick else "full"}
    print(f"train={len(X_train)} test={len(X_test)} | tracking: {config.MLFLOW_TRACKING_URI}")

    results = []
    for name, (estimator, grid) in MODELS.items():
        grid = QUICK_GRIDS[name] if quick else grid
        for variant in FEATURE_VARIANTS:
            results.append(train_one(name, estimator, grid, variant,
                                     X_train, X_test, y_train, y_test, tags))

    summary = pd.DataFrame(results).drop(columns=["pipeline", "holdout_proba"])
    champion = results[select_champion(summary)]

    with mlflow.start_run(run_name="model-selection"):
        mlflow.set_tags({**tags, "champion_run_id": champion["run_id"]})
        mlflow.log_params({"selection_metric": "cv_roc_auc",
                           "tie_tolerance": AUC_TIE_TOLERANCE, "tie_breaker": "cv_recall",
                           "champion": f"{champion['model']}-{champion['features']}"})
        mlflow.log_metrics({k: champion[k] for k in champion if k.startswith(("cv_", "test_"))})
        mlflow.log_text(summary.to_csv(index=False), "model_comparison.csv")
        proba = champion["holdout_proba"]
        figures = {
            "model_comparison.png": plot_model_comparison(summary),
            "champion_roc_curve.png": plot_roc(y_test, proba, champion["model"]),
            "champion_confusion_matrix.png": plot_confusion(y_test, proba),
            "champion_feature_importance.png": plot_feature_importance(champion["pipeline"]),
        }
        for fname, fig in figures.items():
            plotting.save(fig, config.FIGURES_DIR / f"train_{fname}")
            log_figure(fig, fname)

    model_version = register(champion["run_id"]) if do_register else "unregistered"
    export(champion, model_version, tags)

    cols = ["model", "features", "cv_roc_auc", "cv_roc_auc_std", "cv_recall", "test_roc_auc",
            "test_accuracy", "test_recall", "test_precision", "test_f1"]
    print("\n" + summary[cols].round(4).to_string(index=False))
    print(f"\nChampion: {champion['model']} ({champion['features']} features), "
          f"registry version {model_version} -> {config.MODEL_FILE.name}")
    return champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and track heart disease models")
    parser.add_argument("--quick", action="store_true", help="one candidate per model (CI)")
    parser.add_argument("--no-register", action="store_true", help="skip MLflow registry")
    args = parser.parse_args()
    run(quick=args.quick, do_register=not args.no_register)


if __name__ == "__main__":
    main()
