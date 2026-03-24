import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os
from utils import FXTradingConfig
from data_processing import DataProcessor
from strategies import TradingStrategy
from metrics import ModelEvalMetrics

pytestmark = pytest.mark.integration

def _mock_heavy_imports():
    """Mock all heavy ML imports so run_trading_strategy can be loaded."""
    mock_modules = {}
    for mod_name in [
        "torch", "torch.cuda",
        "models", "models.base_model", "models.darts_model",
        "models.chronos_model", "models.toto_model",
        "matplotlib", "matplotlib.pyplot",
    ]:
        mock_modules[mod_name] = MagicMock()
    return mock_modules

def _import_group_data_by_date():
    mocks = _mock_heavy_imports()
    with patch.dict("sys.modules", mocks):
        sys.modules.pop("run_trading_strategy", None)
        from run_trading_strategy import group_data_by_date
    return group_data_by_date

def _make_fx_data(n_rows, start_price=1.2550, seed=42, base_time=None):
    rng = np.random.default_rng(seed)
    if base_time is None:
        base_time = datetime(2023, 6, 1, 9, 0)
    dates = [base_time + timedelta(minutes=i) for i in range(n_rows)]
    mid = start_price + np.cumsum(rng.normal(0, 0.0001, n_rows))
    spread = 0.0002
    return pd.DataFrame({
        "date": dates,
        "bid_price": mid - spread / 2,
        "ask_price": mid + spread / 2,
        "mid_price": mid,
    })

def _make_news_data(n_rows, sentiments=None, seed=42, base_time=None):
    rng = np.random.default_rng(seed)
    if base_time is None:
        base_time = datetime(2023, 6, 1, 9, 5)
    dates = [base_time + timedelta(minutes=i * 4) for i in range(n_rows)]
    if sentiments is None:
        sentiments = rng.choice([-1, 0, 1], size=n_rows)
    return pd.DataFrame({
        "date": dates,
        "competitor_label": sentiments,
    })

def _setup_csv_files(tmp_path, config, n_train=120, n_val=40, n_test=100, n_news_train=20, n_news_test=15,
                     news_sentiments_test=None):
    """Write synthetic CSV files and update config paths."""
    os.makedirs(tmp_path, exist_ok=True)

    train_df = _make_fx_data(n_train, seed=1)
    val_df = _make_fx_data(n_val, seed=2, base_time=datetime(2023, 6, 2, 9, 0))
    test_df = _make_fx_data(n_test, seed=3, base_time=datetime(2023, 6, 3, 9, 0))
    news_train = _make_news_data(n_news_train, seed=4)
    news_test = _make_news_data(n_news_test, sentiments=news_sentiments_test, seed=5,
                                base_time=datetime(2023, 6, 3, 9, 5))

    paths = {
        "fx_train": str(tmp_path / "fx-train.csv"),
        "fx_val": str(tmp_path / "fx-val.csv"),
        "fx_test": str(tmp_path / "fx-test.csv"),
        "news_train": str(tmp_path / "news-train.csv"),
        "news_test": str(tmp_path / "news-test.csv"),
    }

    train_df.to_csv(paths["fx_train"], index=False)
    val_df.to_csv(paths["fx_val"], index=False)
    test_df.to_csv(paths["fx_test"], index=False)
    news_train.to_csv(paths["news_train"], index=False)
    news_test.to_csv(paths["news_test"], index=False)

    config.FX_DATA_PATH_TRAIN = paths["fx_train"]
    config.FX_DATA_PATH_VAL = paths["fx_val"]
    config.FX_DATA_PATH_TEST = paths["fx_test"]
    config.NEWS_DATA_PATH_TRAIN = paths["news_train"]
    config.NEWS_DATA_PATH_TEST = paths["news_test"]
    config.INPUT_CHUNK_LENGTH = 10

    return train_df, val_df, test_df, news_train, news_test

def _run_pipeline_with_list_predictions(config):
    """Run the pipeline with list-type predictions (single model)."""
    group_data_by_date = _import_group_data_by_date()

    dp = DataProcessor(config)
    processed = dp.split_and_scale_data()

    true_values = processed.test_mid_prices
    predicted_values = [p + 0.001 for p in true_values]

    fx_timestamps = processed.test_fx_timestamps
    bid_prices = processed.test_bid_prices
    ask_prices = processed.test_ask_prices
    news_timestamps = processed.test_news_timestamps
    news_sentiments = processed.test_news_sentiments

    chunked = group_data_by_date(
        fx_timestamps, news_timestamps, true_values,
        predicted_values, bid_prices, ask_prices, news_sentiments,
    )

    ts = TradingStrategy(
        config.WALLET_A, config.WALLET_B,
        config.NEWS_HOLD_MINUTES, config.BET_SIZING,
        config.ENABLE_TRANSACTION_COSTS, config.ALLOW_NEWS_OVERLAP,
    )

    for date_key, values in chunked.items():
        ts.simulate_trading_with_strategies(
            values["fx_timestamps"], values["true_values"],
            values["predicted_values"], values["bid_prices"],
            values["ask_prices"], values["news_timestamps"],
            values["news_sentiments"],
        )

    return ts, true_values, predicted_values

class TestFullPipelineFixedPredictions:
    def test_full_pipeline_fixed_predictions(self, tmp_path):
        config = FXTradingConfig()
        config.OUTPUT_DIR = str(tmp_path / "results")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        _setup_csv_files(tmp_path, config)

        ts, _, _ = _run_pipeline_with_list_predictions(config)

        for s in ["mean_reversion", "trend", "model_driven"]:
            assert len(ts.pnl[s]) > 0, f"PnL for {s} should be non-empty"
            assert ts.num_trades[s] > 0, f"num_trades for {s} should be > 0"

        changed = False
        for s in ["mean_reversion", "trend", "model_driven"]:
            if ts.wallet_a[s] != 10000.0 or ts.wallet_b[s] != 10000.0:
                changed = True
        assert changed, "At least one strategy should have different wallet values"

class TestFullPipelineEnsemblePredictions:
    def test_full_pipeline_ensemble_predictions(self, tmp_path):
        """Build multi-day data directly so day 2+ has prior training data."""
        group_data_by_date = _import_group_data_by_date()
        rng = np.random.default_rng(42)

        # Build 3 full market days of data (each day: several hours before
        # market open + full 9am-5pm window)
        all_timestamps, all_mids, all_bids, all_asks = [], [], [], []
        price = 1.2550
        for day_offset in range(3):
            base = datetime(2023, 6, 1 + day_offset, 7, 0)
            for minute in range(720):  # 7am to 7pm = 720 min
                ts = base + timedelta(minutes=minute)
                price += rng.normal(0, 0.0001)
                all_timestamps.append(ts)
                all_mids.append(price)
                all_bids.append(price - 0.0001)
                all_asks.append(price + 0.0001)

        pred_values = {
            "arima": [m + 0.0005 for m in all_mids],
            "nbeats": [m + 0.001 for m in all_mids],
            "chronos": [m - 0.0003 for m in all_mids],
        }

        news_ts = [datetime(2023, 6, 1, 10, 5), datetime(2023, 6, 2, 10, 5)]
        news_sents = [1, -1]

        chunked = group_data_by_date(
            all_timestamps, news_ts, all_mids, pred_values,
            all_bids, all_asks, news_sents,
        )

        ts = TradingStrategy(
            10000, 10000, 3, "fixed", False,
            optimize_ensemble=False,
        )

        for date_key, values in sorted(chunked.items()):
            ts.simulate_trading_with_ensemble_strategy(
                values["fx_timestamps"], values["true_values"],
                values["predicted_values"], values["bid_prices"],
                values["ask_prices"], seed=42,
            )

        assert ts.ensemble_model_trained is True
        assert ts.num_trades["ensemble"] > 0

class TestFullPipelineWithTransactionCosts:
    def test_full_pipeline_with_transaction_costs(self, tmp_path):
        config_no_cost = FXTradingConfig()
        config_no_cost.OUTPUT_DIR = str(tmp_path / "res_nc")
        os.makedirs(config_no_cost.OUTPUT_DIR, exist_ok=True)
        config_no_cost.ENABLE_TRANSACTION_COSTS = False
        _setup_csv_files(tmp_path / "nc", config_no_cost)

        config_cost = FXTradingConfig()
        config_cost.OUTPUT_DIR = str(tmp_path / "res_c")
        os.makedirs(config_cost.OUTPUT_DIR, exist_ok=True)
        config_cost.ENABLE_TRANSACTION_COSTS = True
        _setup_csv_files(tmp_path / "c", config_cost)

        ts_no_cost, _, _ = _run_pipeline_with_list_predictions(config_no_cost)
        ts_cost, _, _ = _run_pipeline_with_list_predictions(config_cost)

        for s in ["mean_reversion", "trend", "model_driven"]:
            profit_no_cost = sum(ts_no_cost.pnl[s])
            profit_cost = sum(ts_cost.pnl[s])
            assert profit_cost <= profit_no_cost + 1e-9, (
                f"Transaction costs should not increase profit for {s}: "
                f"no_cost={profit_no_cost:.4f}, with_cost={profit_cost:.4f}"
            )

class TestFullPipelineKellyBetSizing:
    def test_full_pipeline_kelly_bet_sizing(self, tmp_path):
        config = FXTradingConfig()
        config.OUTPUT_DIR = str(tmp_path / "results")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        config.BET_SIZING = "active_kelly"
        _setup_csv_files(tmp_path, config, n_test=200)

        group_data_by_date = _import_group_data_by_date()

        dp = DataProcessor(config)
        processed = dp.split_and_scale_data()

        true_values = processed.test_mid_prices
        predicted_values = [p + 0.001 for p in true_values]

        chunked = group_data_by_date(
            processed.test_fx_timestamps, processed.test_news_timestamps,
            true_values, predicted_values,
            processed.test_bid_prices, processed.test_ask_prices,
            processed.test_news_sentiments,
        )

        ts = TradingStrategy(
            config.WALLET_A, config.WALLET_B,
            config.NEWS_HOLD_MINUTES, config.BET_SIZING,
            config.ENABLE_TRANSACTION_COSTS, config.ALLOW_NEWS_OVERLAP,
        )

        for date_key, values in chunked.items():
            ts.simulate_trading_with_strategies(
                values["fx_timestamps"], values["true_values"],
                values["predicted_values"], values["bid_prices"],
                values["ask_prices"], values["news_timestamps"],
                values["news_sentiments"],
            )

        md_pnl = [p for p in ts.pnl["model_driven"] if p != 0.0]
        if len(md_pnl) > 2:
            assert len(set(abs(round(p, 8)) for p in md_pnl)) > 1, \
                "Kelly bet sizing should produce varying position sizes"

class TestEvalMetricsOnPipelineOutput:
    def test_eval_metrics_on_pipeline_output(self, tmp_path):
        config = FXTradingConfig()
        config.OUTPUT_DIR = str(tmp_path / "results")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        _setup_csv_files(tmp_path, config)

        _, true_values, predicted_values = _run_pipeline_with_list_predictions(config)

        metrics = ModelEvalMetrics()
        result = metrics.calculate_prediction_error(predicted_values, true_values)

        for key in ["RMSE", "MASE", "MAPE", "sMAPE"]:
            assert key in result
            assert np.isfinite(result[key])
            assert result[key] >= 0

class TestFullPipelineKellyRollingWindow:
    def test_kelly_rolling_window_differs_from_full_history(self, tmp_path):
        """Kelly with a rolling window should produce different sizing than full history."""
        group_data_by_date = _import_group_data_by_date()
        rng = np.random.default_rng(99)

        all_timestamps, all_mids, all_bids, all_asks = [], [], [], []
        price = 1.2550
        for day_offset in range(5):
            base = datetime(2023, 6, 1 + day_offset, 7, 0)
            for minute in range(720):
                ts = base + timedelta(minutes=minute)
                price += rng.normal(0, 0.0001)
                all_timestamps.append(ts)
                all_mids.append(price)
                all_bids.append(price - 0.0001)
                all_asks.append(price + 0.0001)

        predicted_values = [m + 0.0005 for m in all_mids]

        news_ts = [datetime(2023, 6, d, 10, 5) for d in range(1, 6)]
        news_sents = [1, -1, 1, -1, 1]

        chunked = group_data_by_date(
            all_timestamps, news_ts, all_mids, predicted_values,
            all_bids, all_asks, news_sents,
        )

        def _run_with_window(window):
            ts = TradingStrategy(
                10000, 10000, 3, "active_kelly", False,
                kelly_window_days=window,
            )
            for date_key, values in sorted(chunked.items()):
                ts.advance_kelly_day()
                ts.simulate_trading_with_strategies(
                    values["fx_timestamps"], values["true_values"],
                    values["predicted_values"], values["bid_prices"],
                    values["ask_prices"], values["news_timestamps"],
                    values["news_sentiments"],
                )
            return ts

        ts_full = _run_with_window(None)
        ts_windowed = _run_with_window(2)

        for s in ["mean_reversion", "trend", "model_driven"]:
            assert ts_full.num_trades[s] > 0
            assert ts_windowed.num_trades[s] > 0

        full_pnl = sum(ts_full.pnl["model_driven"])
        windowed_pnl = sum(ts_windowed.pnl["model_driven"])
        assert full_pnl != pytest.approx(windowed_pnl, abs=1e-12), \
            "Rolling window should produce different PnL than full history"


class TestNewsSentimentIntegration:
    def test_news_sentiment_integration(self, tmp_path):
        config = FXTradingConfig()
        config.OUTPUT_DIR = str(tmp_path / "results")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        n_news = 20
        sentiments = np.array([1] * (n_news // 2) + [-1] * (n_news // 2))
        _setup_csv_files(tmp_path, config, n_test=200, n_news_test=n_news,
                         news_sentiments_test=sentiments)

        group_data_by_date = _import_group_data_by_date()

        dp = DataProcessor(config)
        processed = dp.split_and_scale_data()

        true_values = processed.test_mid_prices
        predicted_values = [p + 0.001 for p in true_values]

        chunked = group_data_by_date(
            processed.test_fx_timestamps, processed.test_news_timestamps,
            true_values, predicted_values,
            processed.test_bid_prices, processed.test_ask_prices,
            processed.test_news_sentiments,
        )

        ts = TradingStrategy(
            config.WALLET_A, config.WALLET_B,
            config.NEWS_HOLD_MINUTES, config.BET_SIZING,
            config.ENABLE_TRANSACTION_COSTS, config.ALLOW_NEWS_OVERLAP,
        )

        for date_key, values in chunked.items():
            ts.simulate_trading_with_strategies(
                values["fx_timestamps"], values["true_values"],
                values["predicted_values"], values["bid_prices"],
                values["ask_prices"], values["news_timestamps"],
                values["news_sentiments"],
            )

        assert ts.num_trades["news_sentiment"] > 0