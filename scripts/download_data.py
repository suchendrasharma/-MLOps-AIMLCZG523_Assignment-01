"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/.

Usage:
    python scripts/download_data.py [--force]

The official UCI archive zip is tried first; the legacy direct-file URL is the fallback.
Only the Cleveland "processed" file is extracted, since it is the standard 303-row,
14-attribute version used in the literature.
"""

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TARGET_NAME = "processed.cleveland.data"

ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
DIRECT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
TIMEOUT = 60


def _from_zip() -> bytes:
    resp = requests.get(ZIP_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        member = next(n for n in zf.namelist() if Path(n).name == TARGET_NAME)
        return zf.read(member)


def _direct() -> bytes:
    resp = requests.get(DIRECT_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content


def download(force: bool = False) -> Path:
    out = RAW_DIR / TARGET_NAME
    if out.exists() and not force:
        print(f"Already present: {out} (use --force to re-download)")
        return out

    content = None
    for name, fetch in (("zip archive", _from_zip), ("direct file", _direct)):
        try:
            print(f"Downloading via {name} ...")
            content = fetch()
            break
        except Exception as exc:  # noqa: BLE001 - try the next source
            print(f"  failed: {exc}")
    if content is None:
        sys.exit("ERROR: could not download the dataset from any source.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.write_bytes(content)
    rows = content.decode("utf-8").strip().count("\n") + 1
    print(f"Saved {out} ({rows} rows, sha256={hashlib.sha256(content).hexdigest()[:16]}...)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    download(force=parser.parse_args().force)
