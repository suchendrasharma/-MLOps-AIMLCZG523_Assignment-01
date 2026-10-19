"""Generate realistic traffic against the API (for dashboards, alerts and the demo video).

    python scripts/load_test.py --url http://localhost --duration 120 --rps 10
    python scripts/load_test.py --profile high-risk --duration 600   # trigger the drift alert

Requests replay real patients from the cleaned dataset. A configurable share are batch
calls or deliberately invalid payloads (to exercise 4xx monitoring). Prints a summary of
status codes and client-side latency at the end.
"""

import argparse
import random
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from heart import config

# One broken field per invalid request, so each maps to a clear 422.
INVALID_MUTATIONS = [
    ("cp", 9), ("thal", 4), ("age", 5), ("oldpeak", -2.0), ("sex", "male"), ("chol", None),
]


def load_patients(profile: str) -> list[dict]:
    df = pd.read_csv(config.CLEAN_FILE)
    if profile == "high-risk":
        df = df[df[config.TARGET] == 1]
    elif profile == "low-risk":
        df = df[df[config.TARGET] == 0]
    feats = df[config.FEATURES]
    records = feats.astype(object).where(feats.notna(), None).to_dict("records")
    return [{k: (v if v is None or k == "oldpeak" else int(v)) for k, v in r.items()}
            for r in records]


def make_request(patients: list[dict], rng: random.Random, invalid_rate: float,
                 batch_rate: float) -> tuple[str, dict]:
    """Return (path, json body) for one request."""
    roll = rng.random()
    if roll < invalid_rate:
        field, value = rng.choice(INVALID_MUTATIONS)
        return "/predict", {**rng.choice(patients), field: value}
    if roll < invalid_rate + batch_rate:
        return "/predict/batch", {"patients": rng.sample(patients, k=rng.randint(2, 8))}
    return "/predict", rng.choice(patients)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="API load generator")
    parser.add_argument("--url", default="http://localhost")
    parser.add_argument("--duration", type=float, default=60, help="seconds")
    parser.add_argument("--rps", type=float, default=10, help="target requests per second")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--invalid-rate", type=float, default=0.05)
    parser.add_argument("--batch-rate", type=float, default=0.10)
    parser.add_argument("--profile", choices=["mixed", "high-risk", "low-risk"], default="mixed")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    url = args.url.rstrip("/")
    patients = load_patients(args.profile)
    rng = random.Random(args.seed)
    statuses: Counter = Counter()
    latencies: list[float] = []
    lock = threading.Lock()
    # One keep-alive connection per worker thread, like independent clients. A Kubernetes
    # Service balances connections, not requests: a single shared connection would pin
    # all traffic to one pod.
    local = threading.local()

    def fire(path: str, body: dict) -> None:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        start = time.perf_counter()
        try:
            code = str(local.session.post(url + path, json=body, timeout=5).status_code)
        except requests.RequestException as exc:
            code = type(exc).__name__
        with lock:
            statuses[code] += 1
            latencies.append(time.perf_counter() - start)

    print(f"Load test: {url} | {args.rps} req/s for {args.duration:.0f}s | "
          f"profile={args.profile} | invalid={args.invalid_rate:.0%} batch={args.batch_rate:.0%}")
    interval = 1.0 / args.rps
    deadline = time.monotonic() + args.duration
    next_at = time.monotonic()
    sent = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while time.monotonic() < deadline:
            pool.submit(fire, *make_request(patients, rng, args.invalid_rate, args.batch_rate))
            sent += 1
            if sent % max(1, int(args.rps * 10)) == 0:
                print(f"  {sent} sent, status so far: {dict(statuses)}")
            next_at += interval
            time.sleep(max(0.0, next_at - time.monotonic()))

    lat_ms = sorted(x * 1000 for x in latencies)
    p95 = lat_ms[int(0.95 * (len(lat_ms) - 1))] if lat_ms else float("nan")
    print(f"\nSent {sent} requests: {dict(sorted(statuses.items()))}")
    if lat_ms:
        print(f"Latency (client side): p50={statistics.median(lat_ms):.1f} ms  "
              f"p95={p95:.1f} ms  max={lat_ms[-1]:.1f} ms")
    unexpected = sum(n for code, n in statuses.items() if code not in {"200", "422"})
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
