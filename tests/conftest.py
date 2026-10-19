"""Shared fixtures: small synthetic frames in the raw UCI layout (no network, no real files)."""

import pandas as pd
import pytest

from heart import config

# Rows mirror the raw file: floats everywhere, `num` 0-4, NaN where the source had '?'.
RAW_ROWS = [
    [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0.0, 6.0, 0],
    [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3.0, 3.0, 2],
    [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2.0, 7.0, 1],
    [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0.0, 3.0, 0],
    [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, None, 3.0, 4],
    [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0.0, None, 0],
]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(RAW_ROWS, columns=config.RAW_COLUMNS, dtype=float)


@pytest.fixture
def raw_file(tmp_path, raw_df):
    """Write the fixture in the exact on-disk UCI format: no header, '?' for missing."""
    path = tmp_path / "processed.cleveland.data"
    raw_df.to_csv(path, header=False, index=False, na_rep="?")
    return path
