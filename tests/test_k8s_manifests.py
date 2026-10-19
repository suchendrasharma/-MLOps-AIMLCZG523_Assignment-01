"""Static checks on the Kubernetes manifests (schema validation runs in CI via kubeconform).

These encode the deployment's production rules so a careless edit fails fast:
probes, resources, non-root, image pull policy, label/selector wiring, scrape annotations.
"""

import re

import pytest
import yaml

from heart import config

K8S = config.PROJECT_ROOT / "k8s"


def load(name: str) -> dict:
    return yaml.safe_load((K8S / name).read_text())


@pytest.fixture(scope="module")
def deployment():
    return load("deployment.yaml")


@pytest.fixture(scope="module")
def container(deployment):
    (c,) = deployment["spec"]["template"]["spec"]["containers"]
    return c


def test_kustomization_lists_every_manifest():
    listed = set(load("kustomization.yaml")["resources"])
    on_disk = {p.name for p in K8S.glob("*.yaml")} - {"kustomization.yaml"}
    assert listed == on_disk


def test_everything_in_one_namespace():
    ns = load("namespace.yaml")["metadata"]["name"]
    for name in ("configmap.yaml", "deployment.yaml", "service.yaml", "pdb.yaml"):
        assert load(name)["metadata"]["namespace"] == ns, name


def test_image_is_pinned_and_not_pulled_from_a_registry(container):
    image = container["image"]
    assert not image.endswith(":latest") and ":" in image
    assert container["imagePullPolicy"] == "IfNotPresent"


def test_port_matches_dockerfile(container):
    exposed = re.search(r"^EXPOSE\s+(\d+)", (config.PROJECT_ROOT / "Dockerfile").read_text(), re.M)
    assert container["ports"][0]["containerPort"] == int(exposed.group(1))


@pytest.mark.parametrize("probe", ["startupProbe", "readinessProbe", "livenessProbe"])
def test_health_probes(container, probe):
    assert container[probe]["httpGet"]["path"] == "/health"


def test_resources_requests_and_limits(container):
    for kind in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(container["resources"][kind])


def test_hardened_security_context(deployment, container):
    pod_sc = deployment["spec"]["template"]["spec"]["securityContext"]
    assert pod_sc["runAsNonRoot"] is True and pod_sc["runAsUser"] != 0
    sc = container["securityContext"]
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["readOnlyRootFilesystem"] is True
    assert sc["capabilities"]["drop"] == ["ALL"]


def test_zero_downtime_rollout_settings(deployment, container):
    rolling = deployment["spec"]["strategy"]["rollingUpdate"]
    assert rolling["maxUnavailable"] == 0
    assert deployment["spec"]["replicas"] >= 2
    assert container["lifecycle"]["preStop"]  # drains before shutdown


def test_service_and_pdb_select_the_pods(deployment):
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    service = load("service.yaml")
    assert service["spec"]["type"] == "LoadBalancer"
    for selector in (service["spec"]["selector"],
                     load("pdb.yaml")["spec"]["selector"]["matchLabels"],
                     deployment["spec"]["selector"]["matchLabels"]):
        assert selector.items() <= pod_labels.items()


def test_prometheus_scrape_annotations(deployment, container):
    ann = deployment["spec"]["template"]["metadata"]["annotations"]
    assert ann["prometheus.io/scrape"] == "true"
    assert ann["prometheus.io/path"] == "/metrics"
    assert int(ann["prometheus.io/port"]) == container["ports"][0]["containerPort"]
