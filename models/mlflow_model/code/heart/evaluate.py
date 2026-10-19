"""Metrics and evaluation plots for a fitted classification pipeline."""

import matplotlib

matplotlib.use("Agg")  # headless: training runs in CI and scripts without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from heart import config, plotting  # noqa: E402
from heart.features import transformed_feature_names  # noqa: E402

METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]

# GridSearchCV scorers. A heavily regularised candidate can predict no positives in a
# fold; precision/F1 are then scored as 0 explicitly instead of raising warnings.
CV_SCORERS = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
}


def compute_metrics(y_true, proba, threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Threshold-free ROC-AUC plus the threshold-dependent classification metrics."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_roc(y_true, proba, label: str) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=plotting.BASELINE, linestyle="--", linewidth=1, label="chance")
    ax.plot(fpr, tpr, color=plotting.SERIES[0],
            label=f"{label} (AUC = {roc_auc_score(y_true, proba):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve: held-out test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return fig


def plot_confusion(y_true, proba, threshold: float = config.DECISION_THRESHOLD) -> plt.Figure:
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    names = [plotting.CLASS_NAMES[0], plotting.CLASS_NAMES[1]]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap=plotting.SEQUENTIAL, vmin=0)
    for (i, j), n in np.ndenumerate(cm):
        dark = n > cm.max() * 0.6
        ax.text(j, i, f"{n}\n({n / cm[i].sum():.0%} of row)", ha="center", va="center",
                color="white" if dark else plotting.INK, fontsize=9)
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {threshold})")
    ax.grid(False)
    return fig


def feature_importance(pipeline) -> pd.Series:
    """Signed coefficients (linear models) or impurity importances (tree models),
    ordered by magnitude."""
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline)
    if hasattr(model, "coef_"):
        values = model.coef_.ravel()
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        raise TypeError(f"{type(model).__name__} exposes no feature importances")
    imp = pd.Series(values, index=names)
    return imp.reindex(imp.abs().sort_values(ascending=False).index)


def plot_feature_importance(pipeline, top: int = 15) -> plt.Figure:
    imp = feature_importance(pipeline).head(top)[::-1]
    linear = hasattr(pipeline.named_steps["model"], "coef_")
    fig, ax = plt.subplots(figsize=(5.8, 0.32 * len(imp) + 1.4))
    if linear:
        colors = [plotting.DIVERGING(0.9) if v > 0 else plotting.DIVERGING(0.1) for v in imp]
        ax.barh(imp.index, imp.values, color=colors, height=0.6)
        ax.axvline(0, color=plotting.BASELINE, linewidth=1)
        ax.set_xlabel("coefficient (standardised / one-hot inputs)")
        ax.set_title(f"Top {len(imp)} coefficients: red raises risk, blue lowers it")
    else:
        ax.barh(imp.index, imp.values, color=plotting.SERIES[0], height=0.6)
        ax.set_xlabel("impurity importance")
        ax.set_title(f"Top {len(imp)} features by importance")
    ax.grid(axis="y", visible=False)
    return fig


def plot_model_comparison(summary: pd.DataFrame) -> plt.Figure:
    """CV ROC-AUC (mean +/- 1 std) per model family, base vs engineered feature set.

    A dot plot rather than bars: the scores sit in a narrow band, and bars on a
    non-zero baseline would exaggerate the differences.
    """
    # (variant, label, colour, vertical offset, label offset in points)
    variants = [("base", "Base features", plotting.SERIES[0], -0.15, -13),
                ("engineered", "+ hr_pct_max", plotting.SERIES[2], 0.15, 6)]
    families = list(dict.fromkeys(summary["model"]))
    y = np.arange(len(families))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for variant, label, color, offset, text_dy in variants:
        rows = summary[summary["features"] == variant].set_index("model").reindex(families)
        ax.errorbar(rows["cv_roc_auc"], y + offset, xerr=rows["cv_roc_auc_std"], fmt="o",
                    color=color, markersize=8, elinewidth=2, capsize=0, label=label,
                    markeredgecolor=plotting.SURFACE, markeredgewidth=1.5)
        for yy, v in zip(y + offset, rows["cv_roc_auc"], strict=True):
            ax.annotate(f"{v:.3f}", (v, yy), xytext=(0, text_dy), textcoords="offset points",
                        ha="center", fontsize=8, color=plotting.INK_SECONDARY)
    ax.set_yticks(y, families)
    ax.set_ylim(y.min() - 0.55, y.max() + 0.55)
    ax.set_xlabel(f"ROC-AUC, mean ± 1 std over {config.CV_FOLDS} CV folds")
    ax.set_title("Model comparison: best tuned candidate per model")
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, -0.2), borderaxespad=0)
    ax.grid(axis="y", visible=False)
    return fig
