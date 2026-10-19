"""End-to-end smoke test against a running API (local, Docker or Kubernetes).

    python scripts/smoke_test.py                          # http://127.0.0.1:8000
    python scripts/smoke_test.py --url http://localhost:80 --wait 60

Exits non-zero on the first failed check, so CI and deploy scripts can gate on it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def wait_until_healthy(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise AssertionError(f"API at {url} not healthy after {timeout:.0f}s")
        time.sleep(1)


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))
    if not condition:
        raise AssertionError(name)


def run(url: str) -> None:
    high = json.loads((SAMPLES / "patient_high_risk.json").read_text())
    low = json.loads((SAMPLES / "patient_low_risk.json").read_text())
    batch = json.loads((SAMPLES / "patients_batch.json").read_text())

    health = requests.get(f"{url}/health", timeout=5).json()
    check("health ok", health.get("status") == "ok",
          f"model v{health.get('model_version')} ({health.get('model_family')})")

    r = requests.post(f"{url}/predict", json=high, timeout=5,
                      headers={"X-Request-ID": "smoke-high"})
    body = r.json()
    check("high-risk sample predicted 1", r.status_code == 200 and body["prediction"] == 1,
          f"p={body.get('probability')} confidence={body.get('confidence')}")
    check("request id echoed", r.headers.get("X-Request-ID") == "smoke-high")

    body = requests.post(f"{url}/predict", json=low, timeout=5).json()
    check("low-risk sample predicted 0", body.get("prediction") == 0,
          f"p={body.get('probability')}")

    body = requests.post(f"{url}/predict/batch", json={"patients": batch}, timeout=5).json()
    check("batch returns one prediction per patient", body.get("count") == len(batch))

    r = requests.post(f"{url}/predict", json={**high, "cp": 9}, timeout=5)
    check("invalid input rejected with 422", r.status_code == 422)

    metrics = requests.get(f"{url}/metrics", timeout=5).text
    check("prometheus metrics exposed", "heart_predictions_total" in metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait", type=float, default=30, help="seconds to wait for /health")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    print(f"Smoke testing {url}")
    try:
        wait_until_healthy(url, args.wait)
        run(url)
    except (AssertionError, requests.RequestException, KeyError, ValueError) as exc:
        print(f"SMOKE TEST FAILED: {exc}")
        return 1
    print("All smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
