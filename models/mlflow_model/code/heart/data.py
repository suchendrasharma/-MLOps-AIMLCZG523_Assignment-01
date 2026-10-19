"""Load, clean and validate the Cleveland heart disease data.

Run as a module to produce the cleaned dataset:
    python -m heart.data
"""

from pathlib import Path

import pandas as pd

from heart import config


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks a missing value."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, de-duplicated frame with a binary target.

    Missing values (a handful in `ca` and `thal`) are deliberately kept as NaN here:
    they are imputed inside the model pipeline so that imputation statistics are
    learned from training folds only (no leakage into CV/test scores).
    """
    out = df.copy()
    out = out.drop_duplicates().reset_index(drop=True)

    # `num` is 0 (no disease) or 1-4 (increasing severity) -> presence/absence.
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns="num")

    for col in config.CATEGORICAL_FEATURES:
        out[col] = out[col].astype("Int64")  # nullable int keeps NaN for thal
    out["ca"] = out["ca"].astype(float)
    out["oldpeak"] = out["oldpeak"].astype(float)
    for col in ("age", "trestbps", "chol", "thalach"):
        out[col] = out[col].astype(int)

    return out[config.FEATURES + [config.TARGET]]


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the frame violates the expected schema or value domains."""
    missing = set(config.FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    for col, allowed in config.CATEGORY_DOMAINS.items():
        bad = set(df[col].dropna().astype(int)) - allowed
        if bad:
            raise ValueError(f"{col} has unexpected codes {sorted(bad)}")

    if not df[config.TARGET].isin([0, 1]).all():
        raise ValueError("target must be binary")
    ca = df["ca"].dropna()
    if not ca.between(0, 3).all():
        raise ValueError("ca must be in [0, 3]")
    if (df[["age", "trestbps", "chol", "thalach"]] <= 0).any().any():
        raise ValueError("vital-sign columns must be positive")


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    """Load the cleaned CSV with the same dtypes `clean` produces."""
    df = pd.read_csv(path)
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("Int64")
    return df


def main() -> None:
    df = clean(load_raw())
    validate(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    print(
        f"Saved {config.CLEAN_FILE} | rows={len(df)} | "
        f"missing={df.isna().sum()[lambda s: s > 0].to_dict()} | "
        f"positive rate={df[config.TARGET].mean():.2%}"
    )


if __name__ == "__main__":
    main()
