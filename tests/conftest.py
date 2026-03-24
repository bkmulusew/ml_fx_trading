import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import FXTradingConfig
from strategies import TradingStrategy

def make_fx_data(n_rows=200, start_price=1.2550, seed=42):
    """Generate synthetic minute-frequency FX data within market hours."""
    rng = np.random.default_rng(seed)
    base = datetime(2023, 6, 1, 9, 0)
    dates = [base + timedelta(minutes=i) for i in range(n_rows)]
    mid = start_price + np.cumsum(rng.normal(0, 0.0001, n_rows))
    spread = 0.0002
    return pd.DataFrame({
        "date": dates,
        "bid_price": mid - spread / 2,
        "ask_price": mid + spread / 2,
        "mid_price": mid,
    })

def make_news_data(n_rows=50, seed=42):
    """Generate synthetic news sentiment data."""
    rng = np.random.default_rng(seed)
    base = datetime(2023, 6, 1, 9, 5)
    dates = [base + timedelta(minutes=i * 4) for i in range(n_rows)]
    return pd.DataFrame({
        "date": dates,
        "competitor_label": rng.choice([-1, 0, 1], size=n_rows),
    })

@pytest.fixture
def fx_trading_config(tmp_path):
    """FXTradingConfig with paths pointing to temporary CSV files."""
    config = FXTradingConfig()
    config.FX_DATA_PATH_TRAIN = str(tmp_path / "fx-train.csv")
    config.FX_DATA_PATH_VAL = str(tmp_path / "fx-val.csv")
    config.FX_DATA_PATH_TEST = str(tmp_path / "fx-test.csv")
    config.NEWS_DATA_PATH_TRAIN = str(tmp_path / "news-train.csv")
    config.NEWS_DATA_PATH_TEST = str(tmp_path / "news-test.csv")
    config.OUTPUT_DIR = str(tmp_path / "results")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    return config

@pytest.fixture
def sample_fx_dataframe():
    """DataFrame with columns [date, bid_price, ask_price, mid_price], ~200 rows."""
    return make_fx_data(n_rows=200)

@pytest.fixture
def sample_news_dataframe():
    """DataFrame with columns [date, competitor_label], ~50 rows."""
    return make_news_data(n_rows=50)

@pytest.fixture
def sample_fx_csv_files(tmp_path, sample_fx_dataframe):
    """Write the sample FX dataframe into three CSV files (train/val/test splits)."""
    df = sample_fx_dataframe
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    paths = {
        "train": str(tmp_path / "fx-train.csv"),
        "val": str(tmp_path / "fx-val.csv"),
        "test": str(tmp_path / "fx-test.csv"),
    }
    df.iloc[:train_end].to_csv(paths["train"], index=False)
    df.iloc[train_end:val_end].to_csv(paths["val"], index=False)
    df.iloc[val_end:].to_csv(paths["test"], index=False)
    return paths

@pytest.fixture
def sample_news_csv_files(tmp_path, sample_news_dataframe):
    """Write the sample news dataframe into two CSV files (train/test splits)."""
    df = sample_news_dataframe
    n = len(df)
    split = int(n * 0.6)

    paths = {
        "train": str(tmp_path / "news-train.csv"),
        "test": str(tmp_path / "news-test.csv"),
    }
    df.iloc[:split].to_csv(paths["train"], index=False)
    df.iloc[split:].to_csv(paths["test"], index=False)
    return paths

@pytest.fixture
def trading_strategy_instance():
    """TradingStrategy with fixed bet sizing and no transaction costs."""
    return TradingStrategy(
        wallet_a=10000,
        wallet_b=10000,
        news_hold_minutes=3,
        bet_sizing="fixed",
        enable_transaction_costs=False,
    )

@pytest.fixture
def trading_strategy_with_costs():
    """TradingStrategy with fixed bet sizing and transaction costs enabled."""
    return TradingStrategy(
        wallet_a=10000,
        wallet_b=10000,
        news_hold_minutes=3,
        bet_sizing="fixed",
        enable_transaction_costs=True,
    )

@pytest.fixture
def trading_strategy_kelly_window():
    """TradingStrategy with active Kelly and a 5-day rolling window."""
    return TradingStrategy(
        wallet_a=10000,
        wallet_b=10000,
        news_hold_minutes=3,
        bet_sizing="active_kelly",
        enable_transaction_costs=False,
        kelly_window_days=5,
    )