import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from datetime import datetime, timedelta
from data_processing import DataProcessor, ProcessedData

pytestmark = pytest.mark.unit

def _make_fx_df(n=100, start_price=1.25, seed=42):
    rng = np.random.default_rng(seed)
    base = datetime(2023, 6, 1, 9, 0)
    dates = [base + timedelta(minutes=i) for i in range(n)]
    mid = start_price + np.cumsum(rng.normal(0, 0.0001, n))
    spread = 0.0002
    return pd.DataFrame({
        "date": dates,
        "bid_price": mid - spread / 2,
        "ask_price": mid + spread / 2,
        "mid_price": mid,
    })

def _make_news_df(n=30, seed=42):
    rng = np.random.default_rng(seed)
    base = datetime(2023, 6, 1, 9, 5)
    dates = [base + timedelta(minutes=i * 3) for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "competitor_label": rng.choice([-1, 0, 1], size=n),
    })

class TestLoadFxData:
    def test_load_fx_data_sorts_by_date(self, fx_trading_config):
        df = _make_fx_df(50)
        shuffled = df.sample(frac=1, random_state=0)

        with patch("data_processing.data_processor.pd.read_csv", return_value=shuffled.copy()):
            dp = DataProcessor(fx_trading_config)
            train, val, test = dp.load_fx_data()

            for result_df in [train, val, test]:
                dates = result_df["date"].tolist()
                assert dates == sorted(dates)

    def test_load_fx_data_parses_dates(self, fx_trading_config):
        df = _make_fx_df(50)
        df["date"] = df["date"].astype(str)

        with patch("data_processing.data_processor.pd.read_csv", return_value=df.copy()):
            dp = DataProcessor(fx_trading_config)
            train, val, test = dp.load_fx_data()

            for result_df in [train, val, test]:
                assert pd.api.types.is_datetime64_any_dtype(result_df["date"])

class TestAggregateNewsByMinute:
    @pytest.fixture
    def processor(self, fx_trading_config):
        return DataProcessor(fx_trading_config)

    def test_aggregate_news_by_minute_majority_vote(self, processor):
        ts = datetime(2023, 6, 1, 10, 0, 0)
        df = pd.DataFrame({
            "date": [ts, ts + timedelta(seconds=10), ts + timedelta(seconds=20)],
            "competitor_label": [1, 1, -1],
        })
        result = processor._aggregate_news_by_minute(df, "competitor_label")
        assert len(result) == 1
        assert result["competitor_label"].iloc[0] == 1

    def test_aggregate_news_by_minute_tie_returns_zero(self, processor):
        ts = datetime(2023, 6, 1, 10, 0, 0)
        df = pd.DataFrame({
            "date": [ts, ts + timedelta(seconds=10)],
            "competitor_label": [1, -1],
        })
        result = processor._aggregate_news_by_minute(df, "competitor_label")
        assert len(result) == 1
        assert result["competitor_label"].iloc[0] == 0

    def test_aggregate_news_by_minute_empty(self, processor):
        df = pd.DataFrame(columns=["date", "competitor_label"])
        result = processor._aggregate_news_by_minute(df, "competitor_label")
        assert result.empty

    def test_aggregate_news_single_per_minute(self, processor):
        ts1 = datetime(2023, 6, 1, 10, 0)
        ts2 = datetime(2023, 6, 1, 10, 1)
        df = pd.DataFrame({
            "date": [ts1, ts2],
            "competitor_label": [1, -1],
        })
        result = processor._aggregate_news_by_minute(df, "competitor_label")
        assert len(result) == 2
        assert result["competitor_label"].iloc[0] == 1
        assert result["competitor_label"].iloc[1] == -1

class TestSplitAndScaleData:
    def test_split_and_scale_data_shapes(self, tmp_path, fx_trading_config):
        """Verify ProcessedData fields have expected shapes and types."""
        train_df = _make_fx_df(120, seed=1)
        val_df = _make_fx_df(40, seed=2)
        test_df = _make_fx_df(80, seed=3)
        news_train = _make_news_df(20, seed=4)
        news_test = _make_news_df(15, seed=5)

        train_df.to_csv(fx_trading_config.FX_DATA_PATH_TRAIN, index=False)
        val_df.to_csv(fx_trading_config.FX_DATA_PATH_VAL, index=False)
        test_df.to_csv(fx_trading_config.FX_DATA_PATH_TEST, index=False)
        news_train.to_csv(fx_trading_config.NEWS_DATA_PATH_TRAIN, index=False)
        news_test.to_csv(fx_trading_config.NEWS_DATA_PATH_TEST, index=False)

        dp = DataProcessor(fx_trading_config)
        result = dp.split_and_scale_data()

        assert isinstance(result, ProcessedData)
        assert isinstance(result.llm_train_scaled, np.ndarray)
        assert isinstance(result.llm_val_scaled, np.ndarray)
        assert isinstance(result.llm_test_scaled, np.ndarray)
        assert result.llm_train_scaled.shape == (120, 1)
        assert result.llm_test_scaled.shape == (80, 1)

    def test_split_and_scale_data_offset(self, tmp_path, fx_trading_config):
        """Verify test metadata is offset by INPUT_CHUNK_LENGTH."""
        fx_trading_config.INPUT_CHUNK_LENGTH = 10

        train_df = _make_fx_df(120, seed=1)
        val_df = _make_fx_df(40, seed=2)
        test_df = _make_fx_df(80, seed=3)
        news_train = _make_news_df(20, seed=4)
        news_test = _make_news_df(15, seed=5)

        train_df.to_csv(fx_trading_config.FX_DATA_PATH_TRAIN, index=False)
        val_df.to_csv(fx_trading_config.FX_DATA_PATH_VAL, index=False)
        test_df.to_csv(fx_trading_config.FX_DATA_PATH_TEST, index=False)
        news_train.to_csv(fx_trading_config.NEWS_DATA_PATH_TRAIN, index=False)
        news_test.to_csv(fx_trading_config.NEWS_DATA_PATH_TEST, index=False)

        dp = DataProcessor(fx_trading_config)
        result = dp.split_and_scale_data()

        expected_len = 80 - 10  # test rows minus INPUT_CHUNK_LENGTH
        assert len(result.test_fx_timestamps) == expected_len
        assert len(result.test_bid_prices) == expected_len
        assert len(result.test_ask_prices) == expected_len
        assert len(result.test_mid_prices) == expected_len

    def test_processed_data_llm_scaler_range(self, tmp_path, fx_trading_config):
        """After MinMaxScaler, llm_train_scaled values should be in [0, 1]."""
        train_df = _make_fx_df(120, seed=1)
        val_df = _make_fx_df(40, seed=2)
        test_df = _make_fx_df(40, seed=3)
        news_train = _make_news_df(20, seed=4)
        news_test = _make_news_df(10, seed=5)

        train_df.to_csv(fx_trading_config.FX_DATA_PATH_TRAIN, index=False)
        val_df.to_csv(fx_trading_config.FX_DATA_PATH_VAL, index=False)
        test_df.to_csv(fx_trading_config.FX_DATA_PATH_TEST, index=False)
        news_train.to_csv(fx_trading_config.NEWS_DATA_PATH_TRAIN, index=False)
        news_test.to_csv(fx_trading_config.NEWS_DATA_PATH_TEST, index=False)

        dp = DataProcessor(fx_trading_config)
        result = dp.split_and_scale_data()

        assert result.llm_train_scaled.min() >= -1e-9
        assert result.llm_train_scaled.max() <= 1.0 + 1e-9

        # Verify scaler round-trip
        original = train_df["mid_price"].values.reshape(-1, 1).astype(np.float32)
        reconstructed = result.llm_scaler.inverse_transform(result.llm_scaler.transform(original))
        np.testing.assert_allclose(reconstructed, original, atol=1e-5)