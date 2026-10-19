"""Static checks that the container packaging stays consistent with the project."""

import re

from heart import config

ROOT = config.PROJECT_ROOT


def pins(path):
    out = {}
    for line in (ROOT / path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if m := re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==(\S+)$", line):
            out[m.group(1).lower()] = m.group(3)
    return out


def test_serve_requirements_match_main_pins():
    serve, full = pins("requirements-serve.txt"), pins("requirements.txt")
    assert serve, "requirements-serve.txt has no pinned packages"
    mismatched = {k: (v, full.get(k)) for k, v in serve.items() if full.get(k) != v}
    assert not mismatched, f"serve pins differ from requirements.txt: {mismatched}"


def test_serve_requirements_exclude_training_stack():
    serve = pins("requirements-serve.txt")
    assert not {"mlflow", "matplotlib", "seaborn", "jupyter"} & set(serve)


def test_dockerfile_runs_as_non_root_with_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^USER\s+(?!root|0\b)\S+", dockerfile, re.M)
    assert "HEALTHCHECK" in dockerfile
    assert "requirements-serve.txt" in dockerfile


def test_dockerignore_ships_model_but_not_dev_files():
    rules = (ROOT / ".dockerignore").read_text().splitlines()
    allowed = {r[1:] for r in rules if r.startswith("!")}
    assert {"models/model.joblib", "models/metadata.json", "api/", "src/heart/"} <= allowed
    assert "*" in rules  # everything else is excluded (.venv, mlruns, data, .git, ...)
