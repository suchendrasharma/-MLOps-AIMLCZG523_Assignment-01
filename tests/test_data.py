"""Unit tests for heart.data: loading, cleaning and validation."""

import pandas as pd
import pytest

from heart import config
from heart.data import clean, load_clean, load_raw, validate


class TestLoadRaw:
    def test_parses_question_marks_as_nan(self, raw_file):
        df = load_raw(raw_file)
        assert list(df.columns) == config.RAW_COLUMNS
        assert df.shape == (6, 14)
        assert df["ca"].isna().sum() == 1
        assert df["thal"].isna().sum() == 1

    def test_missing_file_gives_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data.py"):
            load_raw(tmp_path / "nope.data")


class TestClean:
    def test_target_is_binarised(self, raw_df):
        out = clean(raw_df)
        # num = [0, 2, 1, 0, 4, 0]  ->  any severity > 0 means disease
        assert out[config.TARGET].tolist() == [0, 1, 1, 0, 1, 0]
        assert "num" not in out.columns

    def test_output_columns_are_features_then_target(self, raw_df):
        assert list(clean(raw_df).columns) == config.FEATURES + [config.TARGET]

    def test_missing_values_are_preserved_for_pipeline_imputation(self, raw_df):
        out = clean(raw_df)
        assert out["ca"].isna().sum() == 1
        assert out["thal"].isna().sum() == 1

    def test_categoricals_are_nullable_ints(self, raw_df):
        out = clean(raw_df)
        for col in config.CATEGORICAL_FEATURES:
            assert str(out[col].dtype) == "Int64", col

    def test_duplicates_are_dropped(self, raw_df):
        doubled = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
        assert len(clean(doubled)) == len(raw_df)

    def test_does_not_mutate_input(self, raw_df):
        before = raw_df.copy()
        clean(raw_df)
        pd.testing.assert_frame_equal(raw_df, before)


class TestValidate:
    def test_clean_frame_passes(self, raw_df):
        validate(clean(raw_df))  # should not raise

    def test_rejects_unknown_category_code(self, raw_df):
        df = clean(raw_df)
        df.loc[0, "cp"] = 9
        with pytest.raises(ValueError, match="cp"):
            validate(df)

    def test_rejects_missing_column(self, raw_df):
        with pytest.raises(ValueError, match="missing columns"):
            validate(clean(raw_df).drop(columns="thal"))

    @pytest.mark.parametrize("bad_value", [3, None])
    def test_rejects_non_binary_or_missing_target(self, raw_df, bad_value):
        df = clean(raw_df).astype({config.TARGET: float})
        df.loc[0, config.TARGET] = bad_value
        with pytest.raises(ValueError, match="binary"):
            validate(df)

    def test_rejects_out_of_range_ca(self, raw_df):
        df = clean(raw_df)
        df.loc[0, "ca"] = 5
        with pytest.raises(ValueError, match="ca"):
            validate(df)

    def test_rejects_non_positive_vitals(self, raw_df):
        df = clean(raw_df)
        df.loc[0, "chol"] = 0
        with pytest.raises(ValueError, match="positive"):
            validate(df)


def test_csv_round_trip_keeps_schema(tmp_path, raw_df):
    out = clean(raw_df)
    path = tmp_path / "clean.csv"
    out.to_csv(path, index=False)
    back = load_clean(path)
    assert list(back.columns) == list(out.columns)
    assert back[config.CATEGORICAL_FEATURES].dtypes.astype(str).eq("Int64").all()
    validate(back)
