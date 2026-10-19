"""Quality gate: compare a freshly trained model against the committed baseline.

    python scripts/check_model_quality.py --baseline baseline_metadata.json \
        --candidate models/metadata.json [--max-drop 0.02] [--min-auc 0.85]

Fails (exit 1) if the candidate's test ROC-AUC is below an absolute floor, or has dropped
more than --max-drop compared with the baseline. CI uses this to stop a degraded model
from being packaged and deployed. Also prints a Markdown table (for the CI job summary).
"""

import argparse
import json
import sys
from pathlib import Path

METRICS = ["test_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1"]


def evaluate(baseline: dict, candidate: dict, max_drop: float, min_auc: float) -> list[str]:
    """Return the list of gate violations (empty means the candidate passes)."""
    base_auc = baseline["test_metrics"]["test_roc_auc"]
    cand_auc = candidate["test_metrics"]["test_roc_auc"]
    failures = []
    if cand_auc < min_auc:
        failures.append(f"test ROC-AUC {cand_auc:.4f} is below the floor {min_auc:.2f}")
    if base_auc - cand_auc > max_drop:
        failures.append(f"test ROC-AUC dropped {base_auc - cand_auc:.4f} "
                        f"(baseline {base_auc:.4f} -> {cand_auc:.4f}, allowed {max_drop})")
    return failures


def markdown_report(baseline: dict, candidate: dict, failures: list[str]) -> str:
    lines = [
        "| Metric | Baseline (committed) | Candidate (this run) | Δ |",
        "|---|---|---|---|",
    ]
    for m in METRICS:
        b, c = baseline["test_metrics"].get(m), candidate["test_metrics"].get(m)
        delta = f"{c - b:+.4f}" if b is not None and c is not None else "n/a"
        lines.append(f"| {m.removeprefix('test_')} | {b} | {c} | {delta} |")
    lines.append("")
    lines.append(f"Candidate: **{candidate.get('model_family')}** "
                 f"({candidate.get('feature_set')} features), "
                 f"params `{candidate.get('best_params')}`")
    lines.append("")
    lines.append("**Quality gate: " + ("FAILED**\n\n- " + "\n- ".join(failures)
                                         if failures else "PASSED**"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model quality gate")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-drop", type=float, default=0.02)
    parser.add_argument("--min-auc", type=float, default=0.85)
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    failures = evaluate(baseline, candidate, args.max_drop, args.min_auc)
    print(markdown_report(baseline, candidate, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
