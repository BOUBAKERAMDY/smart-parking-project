import numpy as np
import pandas as pd
import pytest

from train import create_sequences, load_and_transform, load_config, validate_raw_data


@pytest.fixture
def cfg():
    return load_config()


class TestConfig:
    def test_load_config(self, cfg):
        assert "data" in cfg
        assert "model" in cfg
        assert "api" in cfg

    def test_config_has_required_keys(self, cfg):
        assert cfg["model"]["lookback"] == 12
        assert cfg["data"]["resample_freq"] == "h"
        assert "zone_col" in cfg["data"]

    def test_load_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestETL:
    def test_load_and_transform_returns_dataframe(self, cfg):
        df = load_and_transform(cfg)
        assert isinstance(df, pd.DataFrame)

    def test_output_has_zone_columns(self, cfg):
        df = load_and_transform(cfg)
        assert len(df.columns) > 0

    def test_output_values_between_0_and_1(self, cfg):
        df = load_and_transform(cfg)
        assert df.min().min() >= 0.0
        assert df.max().max() <= 1.0

    def test_no_nulls_in_output(self, cfg):
        df = load_and_transform(cfg)
        assert df.isna().sum().sum() == 0

    def test_index_is_datetime(self, cfg):
        df = load_and_transform(cfg)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_hourly_frequency(self, cfg):
        df = load_and_transform(cfg)
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()


class TestValidation:
    def test_validate_missing_columns(self, cfg):
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_raw_data(df, cfg)


class TestSequences:
    def test_create_sequences_shape(self):
        data = np.random.rand(100, 1)
        X, y = create_sequences(data, lookback=12)
        assert X.shape == (88, 12)
        assert y.shape == (88,)

    def test_create_sequences_values(self):
        data = np.arange(20).reshape(-1, 1).astype(float)
        X, y = create_sequences(data, lookback=3)
        np.testing.assert_array_equal(X[0], [0, 1, 2])
        assert y[0] == 3

    def test_create_sequences_empty(self):
        data = np.random.rand(5, 1)
        X, y = create_sequences(data, lookback=10)
        assert len(X) == 0
        assert len(y) == 0
