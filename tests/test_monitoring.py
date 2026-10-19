"""Consistency checks for the monitoring stack (Prometheus config, alerts, Grafana dashboard)
and the load generator. Semantic validation runs in CI with promtool and kubeconform."""

import importlib.util
import json
import random
import re

import pytest
import yaml

from heart import config

MON = config.PROJECT_ROOT / "k8s" / "monitoring"
# Metrics the API actually exports (see api/main.py and the instrumentator defaults),
# plus Prometheus' own `up` and the recording rule defined in alerts.yml.
KNOWN_METRICS = {
    "http_requests_total", "http_request_duration_seconds_bucket", "heart_predictions_total",
    "heart_prediction_probability_bucket", "heart_model_info", "up",
    "heart:high_risk_share:rate30m",
}
PROMQL_FUNCS = {"sum", "rate", "increase", "histogram_quantile", "max", "by", "absent",
                "vector", "or", "and", "le", "humanizePercentage", "humanizeDuration"}


def metric_names(expr: str) -> set[str]:
    """Identifiers in a PromQL expression that are followed by `{`, `[` or used bare."""
    expr = re.sub(r'"[^"]*"', "", expr)          # drop label values
    expr = re.sub(r"\{[^}]*\}", "{}", expr)     # drop label matchers
    expr = re.sub(r"\[[^\]]*\]", "", expr)      # drop range selectors like [5m]
    expr = re.sub(r"by \([^)]*\)", "", expr)    # drop grouping labels
    tokens = set(re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr))
    return {t for t in tokens if t not in PROMQL_FUNCS and not t.startswith("__")}


@pytest.fixture(scope="module")
def dashboard():
    return json.loads((MON / "dashboards" / "heart-api.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules():
    return yaml.safe_load((MON / "alerts.yml").read_text())


class TestPrometheusConfig:
    def test_scrapes_the_api_namespace_with_annotation_opt_in(self):
        prom = yaml.safe_load((MON / "prometheus.yml").read_text())
        job = next(j for j in prom["scrape_configs"] if j["job_name"] == "heart-api")
        sd = job["kubernetes_sd_configs"][0]
        api_ns = yaml.safe_load((config.PROJECT_ROOT / "k8s" / "namespace.yaml").read_text())
        assert sd["role"] == "pod"
        assert sd["namespaces"]["names"] == [api_ns["metadata"]["name"]]
        keep = job["relabel_configs"][0]
        assert keep["action"] == "keep" and "prometheus_io_scrape" in keep["source_labels"][0]

    def test_rbac_is_namespaced_and_read_only(self):
        docs = list(yaml.safe_load_all((MON / "prometheus-rbac.yaml").read_text()))
        role = next(d for d in docs if d["kind"] == "Role")
        assert role["metadata"]["namespace"] == "heart-mlops"
        assert set(role["rules"][0]["verbs"]) == {"get", "list", "watch"}
        assert not [d for d in docs if d["kind"].startswith("Cluster")]


class TestAlertRules:
    def all_rules(self, rules):
        return [r for g in rules["groups"] for r in g["rules"]]

    def test_expected_alerts_exist(self, rules):
        names = {r.get("alert") for r in self.all_rules(rules)}
        assert {"HeartApiDown", "HeartApiHighErrorRate", "HeartApiHighLatency",
                "HeartPredictionDrift"} <= names

    def test_alerts_have_severity_and_summary(self, rules):
        for r in self.all_rules(rules):
            if "alert" in r:
                assert r["labels"]["severity"] in {"critical", "warning", "info"}, r["alert"]
                assert r["annotations"]["summary"], r["alert"]

    def test_rules_only_use_exported_metrics(self, rules):
        for r in self.all_rules(rules):
            assert metric_names(r["expr"]) <= KNOWN_METRICS, r.get("alert") or r.get("record")

    def test_drift_alert_brackets_the_training_base_rate(self, rules):
        drift = next(r for r in self.all_rules(rules) if r.get("alert") == "HeartPredictionDrift")
        lo, hi = map(float, re.findall(r"[<>]\s*(0\.\d+)", drift["expr"]))
        base_rate = 0.4587  # share of positives in data/processed/heart_clean.csv
        assert lo < base_rate < hi


class TestDashboard:
    def test_identity_and_datasource(self, dashboard):
        assert dashboard["uid"] == "heart-api"
        ds = yaml.safe_load((MON / "grafana-datasources.yml").read_text())["datasources"][0]
        for panel in dashboard["panels"]:
            assert panel["datasource"]["uid"] == ds["uid"], panel["title"]

    def test_panel_ids_unique_and_layout_fits_grid(self, dashboard):
        ids = [p["id"] for p in dashboard["panels"]]
        assert len(ids) == len(set(ids))
        for p in dashboard["panels"]:
            assert p["gridPos"]["x"] + p["gridPos"]["w"] <= 24, p["title"]

    def test_queries_only_use_exported_metrics(self, dashboard):
        for panel in dashboard["panels"]:
            for t in panel["targets"]:
                expr = t["expr"].replace("$__rate_interval", "1m").replace("$__range", "5m")
                assert metric_names(expr) <= KNOWN_METRICS, (panel["title"], expr)

    def test_home_dashboard_path_matches_mounted_file(self):
        grafana = (MON / "grafana.yaml").read_text()
        assert "/var/lib/grafana/dashboards/heart-api.json" in grafana

    def test_kustomize_generators_reference_existing_files(self):
        kust = yaml.safe_load((MON / "kustomization.yaml").read_text())
        for gen in kust["configMapGenerator"]:
            for entry in gen["files"]:
                assert (MON / entry.split("=")[-1]).exists(), entry


class TestLoadGenerator:
    @pytest.fixture(scope="class")
    def lt(self):
        spec = importlib.util.spec_from_file_location(
            "load_test", config.PROJECT_ROOT / "scripts" / "load_test.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_profiles_filter_by_label(self, lt):
        everyone = lt.load_patients("mixed")
        high = lt.load_patients("high-risk")
        assert len(everyone) == 303 and 0 < len(high) < len(everyone)
        assert set(everyone[0]) == set(config.FEATURES)

    def test_request_mix_matches_configured_rates(self, lt):
        patients = lt.load_patients("mixed")
        rng = random.Random(0)
        reqs = [lt.make_request(patients, rng, invalid_rate=0.1, batch_rate=0.2)
                for _ in range(4000)]
        batch = [b for path, b in reqs if path == "/predict/batch"]
        assert 0.17 < len(batch) / len(reqs) < 0.23
        assert all(2 <= len(b["patients"]) <= 8 for b in batch)
        invalid = [b for path, b in reqs if path == "/predict" and b not in patients]
        assert 0.08 < len(invalid) / len(reqs) < 0.12
