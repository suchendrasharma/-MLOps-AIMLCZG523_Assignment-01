"""Feature engineering and preprocessing: everything between raw records and the model.

The full sklearn Pipeline is: FeatureEngineer -> ColumnTransformer -> estimator. It is
saved as one object, so training and serving apply identical transformations.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart import config

HR_RATIO = "hr_pct_max"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Select the model inputs, coerce them to float and optionally add derived features.

    `hr_pct_max` = thalach / (220 - age): the share of the age-predicted maximum heart
    rate the patient reached in the stress test. EDA showed thalach falls with age
    (rho = -0.39), so the age-adjusted value should carry a cleaner signal.
    """

    def __init__(self, add_hr_ratio: bool = True):
        self.add_hr_ratio = add_hr_ratio

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(config.FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"input is missing columns: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # float() also turns pandas' nullable <NA> into np.nan, which the imputers expect.
        out = X[config.FEATURES].astype(float)
        if self.add_hr_ratio:
            out[HR_RATIO] = out["thalach"] / (220.0 - out["age"]).clip(lower=1.0)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(config.FEATURES) + ([HR_RATIO] if self.add_hr_ratio else [])
        return np.asarray(names, dtype=object)


def build_preprocessor(add_hr_ratio: bool = True) -> ColumnTransformer:
    """Impute + scale numeric columns; impute + one-hot encode categorical codes."""
    numeric = config.NUMERIC_FEATURES + ([HR_RATIO] if add_hr_ratio else [])
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Rare codes (e.g. restecg=1 has 4 patients) may be absent from a CV fold.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, config.CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, add_hr_ratio: bool = True) -> Pipeline:
    """Full model pipeline: raw records in, predictions out."""
    return Pipeline([
        ("features", FeatureEngineer(add_hr_ratio=add_hr_ratio)),
        ("preprocess", build_preprocessor(add_hr_ratio)),
        ("model", estimator),
    ])


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Column names after preprocessing (e.g. 'cp_4.0'), aligned with model coefficients."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
