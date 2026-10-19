"""Render the system architecture diagram: reports/figures/architecture.png

    python reports/build/architecture.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from heart import config, plotting  # noqa: E402

INK, INK2, MUTED = plotting.INK, plotting.INK_SECONDARY, plotting.MUTED
BLUE, ORANGE, AQUA = plotting.SERIES
LANES = [  # (title, y, height, tint)
    ("1  Data & experimentation", 7.15, 1.55, "#eef4fc"),
    ("2  Training, tracking & packaging", 5.25, 1.55, "#eef4fc"),
    ("3  CI/CD · GitHub Actions", 3.35, 1.55, "#fdf0ea"),
    ("4  Serving & monitoring · Docker Desktop Kubernetes", 0.25, 2.75, "#eaf7f1"),
]


def box(ax, x, y, w, h, title, sub="", edge=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=edge, linewidth=1.4, zorder=3))
    ty = y + h / 2 + (0.13 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=8.6,
            fontweight="semibold", color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.17, sub, ha="center", va="center", fontsize=6.9,
                color=INK2, zorder=4, linespacing=1.15)
    return (x, y, w, h)


def arrow(ax, a, b, label="", color=INK2, rad=0.0, side="right", lw=1.2):
    (ax_, ay, aw, ah), (bx, by, bw, bh) = a, b
    if side == "right":
        p, q = (ax_ + aw, ay + ah / 2), (bx, by + bh / 2)
    elif side == "down":
        p, q = (ax_ + aw / 2, ay), (bx + bw / 2, by + bh)
    else:  # up
        p, q = (ax_ + aw / 2, ay + ah), (bx + bw / 2, by)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color,
                                 linewidth=lw, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.11, label, ha="center", va="bottom",
                fontsize=6.5, color=MUTED, zorder=4)


def elbow(ax, start, end, via_y, label="", label_x=None, color=INK2):
    """Right-angle connector: down from `start` to via_y, across, then down into `end`."""
    (sx, sy), (ex, ey) = start, end
    ax.plot([sx, sx, ex], [sy, via_y, via_y], color=color, linewidth=1.2, zorder=2,
            solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch((ex, via_y), (ex, ey), arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=1.2, zorder=2))
    if label:
        tag(ax, label_x if label_x is not None else (sx + ex) / 2, via_y, label)


def tag(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.6, color=INK2, zorder=5,
            bbox={"facecolor": plotting.SURFACE, "edgecolor": "none", "pad": 1.5})


def n_tests() -> str:
    """Test count from facts.json (written by collect_facts.py), so the diagram stays current."""
    facts = Path(__file__).with_name("facts.json")
    return str(json.loads(facts.read_text())["tests"]["total"]) if facts.exists() else "100+"


def main() -> None:
    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 8.9)
    ax.axis("off")

    for title, y, h, tint in LANES:
        ax.add_patch(FancyBboxPatch((0.1, y), 13.3, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=tint, edgecolor="none", zorder=1))
        # right-aligned so the vertical connectors on the left never cross a lane title
        ax.text(13.25, y + h - 0.2, title, ha="right", fontsize=9.5, fontweight="bold",
                color=INK2, zorder=4)

    # ---- lane 1: data ----
    W, H = 2.05, 0.82
    uci = box(ax, 0.35, 7.35, W, H, "UCI repository", "Heart Disease (id 45)\nCleveland subset")
    dl = box(ax, 2.95, 7.35, W, H, "download_data.py", "zip + fallback URL")
    clean = box(ax, 5.55, 7.35, W, H, "heart.data", "clean · binarise target\nvalidate code book")
    csv = box(ax, 8.15, 7.35, W, H, "heart_clean.csv", "303 rows · LF-normalised\nsha256 tracked")
    eda = box(ax, 10.85, 7.35, W, H, "01_eda.ipynb", "6 figures\nfindings", edge=MUTED)
    for a, b in ((uci, dl), (dl, clean), (clean, csv), (csv, eda)):
        arrow(ax, a, b)

    # ---- lane 2: training ----
    train = box(ax, 0.35, 5.45, 2.6, H, "heart.train",
                "GridSearchCV · 5-fold\nLogReg · RF · GB × 2 feature sets")
    mlf = box(ax, 3.6, 5.45, 2.4, H, "MLflow", "119 runs · plots · metrics\nregistry @champion",
              edge=ORANGE)
    sel = box(ax, 6.65, 5.45, 2.1, H, "Champion selection", "best CV ROC-AUC\nrecall tie-break")
    art = box(ax, 9.4, 5.45, 3.8, H, "models/  (versioned artefacts)",
              "model.joblib · mlflow_model/ · metadata.json\nmodel_sha256 · data_sha256 · metrics")
    elbow(ax, (9.225, 7.35), (1.65, 6.27), 7.0, "cleaned data", label_x=5.4)
    arrow(ax, train, mlf)
    tag(ax, 3.28, 6.02, "log runs")
    arrow(ax, mlf, sel)
    arrow(ax, sel, art)
    tag(ax, 9.08, 6.02, "export")

    # ---- lane 3: CI/CD ----
    W3 = 2.25
    lint = box(ax, 0.35, 3.55, W3, H, "lint", "ruff · hadolint\nkubeconform · promtool",
               edge=ORANGE)
    test = box(ax, 3.05, 3.55, W3, H, "test", f"pytest · {n_tests()} tests\ncoverage ≥ 85%",
               edge=ORANGE)
    gate = box(ax, 5.75, 3.55, W3, H, "train + gate", "fresh UCI data check\nAUC drop ≤ 0.02",
               edge=ORANGE)
    dock = box(ax, 8.45, 3.55, W3, H, "docker", "build · smoke test\n7 checks", edge=ORANGE)
    ghcr = box(ax, 11.15, 3.55, 2.05, H, "GHCR", "heart-api:latest\nsha-<commit>", edge=MUTED)
    for a, b in ((lint, test), (test, gate), (gate, dock), (dock, ghcr)):
        arrow(ax, a, b)
    elbow(ax, (11.3, 5.45), (9.575, 4.37), 5.08, "trained model artefact", label_x=10.45)

    # ---- lane 4: serving & monitoring ----
    client = box(ax, 0.35, 1.45, 1.55, H, "Client", "curl · Swagger\nload_test.py", edge=MUTED)
    lb = box(ax, 2.6, 1.45, 2.0, H, "LoadBalancer :80", "Service heart-api\n(localhost)", edge=AQUA)
    pod1 = box(ax, 5.15, 1.95, 2.55, 0.78, "Pod · FastAPI + model",
               "/predict · /health · /metrics", edge=AQUA)
    pod2 = box(ax, 5.15, 0.95, 2.55, 0.78, "Pod · FastAPI + model",
               "non-root · probes · preStop", edge=AQUA)
    prom = box(ax, 8.35, 1.45, 2.1, H, "Prometheus :9090", "pod discovery\n7 alert/record rules",
               edge=ORANGE)
    graf = box(ax, 11.1, 1.45, 2.1, H, "Grafana :3000", "12-panel service +\nmodel dashboard",
               edge=ORANGE)
    logs = box(ax, 8.35, 0.4, 2.1, 0.7, "JSON logs", "kubectl logs · request_id", edge=MUTED)
    arrow(ax, client, lb)
    tag(ax, 2.25, 2.02, "HTTP/JSON")
    arrow(ax, lb, pod1, rad=-0.15)
    arrow(ax, lb, pod2, rad=0.15)
    arrow(ax, pod1, prom, rad=0.1)
    arrow(ax, pod2, prom, rad=-0.1)
    tag(ax, 8.02, 2.55, "scrape /metrics\nevery 10 s")
    arrow(ax, prom, graf)
    tag(ax, 10.78, 2.02, "PromQL")
    arrow(ax, pod2, logs, rad=0.2)
    # the image runs in the pods; it reaches the cluster via kubectl apply -k
    elbow(ax, (9.575, 3.55), (6.425, 2.73), 3.17, "image heart-api:1.0.1 · kubectl apply -k",
          label_x=8.0)

    ax.text(13.35, 0.08, "Heart disease MLOps pipeline · architecture", ha="right", fontsize=7,
            color=MUTED)
    out = config.FIGURES_DIR / "architecture.png"
    plotting.save(fig, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
