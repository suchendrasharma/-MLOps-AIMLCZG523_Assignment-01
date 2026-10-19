"""Unit tests for heart.features: the engineered feature and the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from heart import config
from heart.data import clean
from heart.features import HR_RATIO, FeatureEngineer, build_preprocessor


@pytest.fixture
def X(raw_df):
    return clean(raw_df)[config.FEATURES]


class TestFeatureEngineer:
    def test_adds_age_adjusted_heart_rate(self, X):
        out = FeatureEngineer(add_hr_ratio=True).fit_transform(X)
        # row 0: age 63, thalach 150 -> 150 / (220 - 63)
        assert out.loc[0, HR_RATIO] == pytest.approx(150 / 157)
        assert list(out.columns) == config.FEATURES + [HR_RATIO]

    def test_can_disable_engineered_feature(self, X):
        out = FeatureEngineer(add_hr_ratio=False).fit_transform(X)
        assert list(out.columns) == config.FEATURES

    def test_coerces_nullable_ints_to_float_nan(self, X):
        out = FeatureEngineer().fit_transform(X)
        assert (out.dtypes == "float64").all()
        assert np.isnan(out["thal"]).sum() == 1  # pandas <NA> became np.nan

    def test_ignores_extra_columns_and_reorders(self, X):
        shuffled = X[X.columns[::-1]].assign(patient_id=range(len(X)))
        out = FeatureEngineer(add_hr_ratio=False).fit_transform(shuffled)
        assert list(out.columns) == config.FEATURES

    def test_missing_input_column_raises(self, X):
        with pytest.raises(ValueError, match="thalach"):
            FeatureEngineer().fit(X.drop(columns="thalach"))

    def test_extreme_age_does_not_divide_by_zero(self, X):
        X = X.copy()
        X.loc[0, "age"] = 220
        out = FeatureEngineer().fit_transform(X)
        assert np.isfinite(out.loc[0, HR_RATIO])


class TestPreprocessor:
    def _fit(self, X, add_hr_ratio=True):
        fe = FeatureEngineer(add_hr_ratio=add_hr_ratio)
        pre = build_preprocessor(add_hr_ratio)
        return fe, pre, pre.fit_transform(fe.fit_transform(X))

    def test_output_has_no_missing_values(self, X):
        _, _, Z = self._fit(X)
        assert not np.isnan(Z).any()

    def test_numeric_columns_are_standardised(self, X):
        _, pre, Z = self._fit(X)
        names = list(pre.get_feature_names_out())
        idx = [names.index(c) for c in ["age", "chol", HR_RATIO]]
        assert np.allclose(Z[:, idx].mean(axis=0), 0, atol=1e-9)

    def test_categoricals_are_one_hot_encoded(self, X):
        _, pre, Z = self._fit(X)
        names = list(pre.get_feature_names_out())
        cp_cols = [i for i, n in enumerate(names) if n.startswith("cp_")]
        assert len(cp_cols) == X["cp"].nunique()
        assert np.allclose(Z[:, cp_cols].sum(axis=1), 1)  # exactly one active code per row

    def test_unseen_category_is_tolerated(self, X):
        fe, pre, _ = self._fit(X)
        new = X.iloc[[0]].copy()
        new["restecg"] = 1  # restecg=1 does not occur in the fixture
        Z = pre.transform(fe.transform(new))
        names = list(pre.get_feature_names_out())
        restecg_cols = [i for i, n in enumerate(names) if n.startswith("restecg_")]
        assert np.allclose(Z[0, restecg_cols], 0)  # encoded as "none of the known codes"

    def test_feature_count_with_and_without_engineering(self, X):
        _, _, with_eng = self._fit(X, add_hr_ratio=True)
        _, _, without = self._fit(X, add_hr_ratio=False)
        assert with_eng.shape[1] == without.shape[1] + 1


def test_single_record_dict_round_trip():
    """The API will send one patient as a dict; the transformer must accept that."""
    record = {"age": 54, "sex": 1, "cp": 4, "trestbps": 130, "chol": 250, "fbs": 0,
              "restecg": 0, "thalach": 150, "exang": 1, "oldpeak": 1.5, "slope": 2,
              "ca": None, "thal": 7}
    out = FeatureEngineer().fit_transform(pd.DataFrame([record]))
    assert out.shape == (1, len(config.FEATURES) + 1)
    assert np.isnan(out.loc[0, "ca"])
