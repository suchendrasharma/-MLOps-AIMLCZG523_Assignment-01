"""Unit tests for scripts/download_data.py. Network calls are stubbed out."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


@pytest.fixture
def dl(tmp_path, monkeypatch):
    """Import the script as a module and point its output folder at a temp dir."""
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RAW_DIR", tmp_path)
    return module


PAYLOAD = b"63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"


def _fail():
    raise ConnectionError("offline")


def test_falls_back_to_direct_url_when_zip_fails(dl, monkeypatch):
    monkeypatch.setattr(dl, "_from_zip", _fail)
    monkeypatch.setattr(dl, "_direct", lambda: PAYLOAD)
    out = dl.download()
    assert out.read_bytes() == PAYLOAD


def test_skips_download_when_file_exists(dl, monkeypatch):
    (dl.RAW_DIR / dl.TARGET_NAME).write_bytes(b"existing")
    monkeypatch.setattr(dl, "_from_zip", _fail)
    monkeypatch.setattr(dl, "_direct", _fail)
    assert dl.download().read_bytes() == b"existing"


def test_exits_with_error_when_all_sources_fail(dl, monkeypatch):
    monkeypatch.setattr(dl, "_from_zip", _fail)
    monkeypatch.setattr(dl, "_direct", _fail)
    with pytest.raises(SystemExit, match="could not download"):
        dl.download(force=True)
