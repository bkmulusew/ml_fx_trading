import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import List, Any

@dataclass
class ProcessedData:
    """Container for all processed and scaled data."""
    # Darts format data (for Darts models)
    darts_train_scaled: Any
    darts_val_scaled: Any
    darts_test_scaled: Any
    darts_scaler: Any  # Darts Scaler for inverse transform

    # llm format data (for Chronos/Toto models)
    llm_train_scaled: np.ndarray
    llm_val_scaled: np.ndarray
    llm_test_scaled: np.ndarray
    llm_scaler: Any  # MinMaxScaler for inverse transform

    # Test metadata (common to all models)
    test_fx_timestamps: List
    test_bid_prices: List[float]
    test_ask_prices: List[float]
    test_news_timestamps: List
    test_news_sentiments: List[float]
    test_mid_prices: List[float]  # Unscaled test mid prices for evaluation

class DataProcessor:
    def __init__(self, fx_trading_config):
        self.fx_trading_config = fx_trading_config

    def load_fx_data(self):
        """Loads the time series data."""
        fx_data_train = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_TRAIN)
        fx_data_val = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_VAL)
        fx_data_test = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_TEST)

        for df in [fx_data_train, fx_data_val, fx_data_test]:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return fx_data_train, fx_data_val, fx_data_test

    def load_news_data(self):
        """Loads the news data."""
        news_data_train = pd.read_csv(self.fx_trading_config.NEWS_DATA_PATH_TRAIN)
        news_data_test = pd.read_csv(self.fx_trading_config.NEWS_DATA_PATH_TEST)

        for df in [news_data_train, news_data_test]:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return news_data_train, news_data_test

    def _aggregate_news_by_minute(self, news_df, sentiment_col):
        """
        For timestamps that share the same year-month-day hour:minute (ignoring seconds),
        take the majority sentiment (+1, 0, -1). Returns one row per minute.
        """
        if news_df.empty:
            return news_df

        df = news_df.copy()

        # Collapse to minute precision (drops seconds)
        df["date_minute"] = df["date"].dt.floor("min")

        def pick_majority_with_tie_zero(x):
            counts = x.value_counts()
            max_count = counts.max()
            # Retrieve all sentiments tied for max count
            tied = counts[counts == max_count].index.tolist()

            if len(tied) == 1:
                return tied[0]  # Normal majority
            else:
                return 0

        # For each minute, take the majority sentiment
        agg = (
            df.groupby("date_minute", as_index=False)[sentiment_col]
              .agg(pick_majority_with_tie_zero)
              .rename(columns={"date_minute": "date"})
        )

        # Ensure sorted by time
        agg.sort_values("date", inplace=True)
        agg.reset_index(drop=True, inplace=True)

        return agg

    def split_and_scale_data(self):
        """Split and scale data for all models."""
        input_chunk_length = self.fx_trading_config.INPUT_CHUNK_LENGTH
        fx_data_train, fx_data_val, fx_data_test = self.load_fx_data()
        news_data_train, news_data_test = self.load_news_data()

        # Aggregate news by minute with majority sentiment
        sentiment_col = self.fx_trading_config.SENTIMENT_SOURCE
        news_data_train = self._aggregate_news_by_minute(news_data_train, sentiment_col)
        news_data_test = self._aggregate_news_by_minute(news_data_test, sentiment_col)

        # Extract test metadata
        fx_timestamps = fx_data_test["date"].tolist()
        bid_prices = fx_data_test["bid_price"].tolist()
        ask_prices = fx_data_test["ask_price"].tolist()
        news_timestamps = news_data_test["date"].tolist()
        news_sentiments = news_data_test[sentiment_col].tolist()

        # --- Prepare Darts format data ---
        darts_train = TimeSeries.from_dataframe(fx_data_train, value_cols=["mid_price"])
        darts_val = TimeSeries.from_dataframe(fx_data_val, value_cols=["mid_price"])
        darts_test = TimeSeries.from_dataframe(fx_data_test, value_cols=["mid_price"])

        # Scale Darts series
        darts_scaler = Scaler()
        darts_train_scaled = darts_scaler.fit_transform(darts_train)
        darts_val_scaled = darts_scaler.transform(darts_val)
        darts_test_scaled = darts_scaler.transform(darts_test)

        # --- Prepare llm format data ---
        llm_train = fx_data_train["mid_price"].values.reshape(-1, 1).astype(np.float32)
        llm_val = fx_data_val["mid_price"].values.reshape(-1, 1).astype(np.float32)
        llm_test = fx_data_test["mid_price"].values.reshape(-1, 1).astype(np.float32)

        # Scale llm arrays
        llm_scaler = MinMaxScaler(feature_range=(0, 1))
        llm_train_scaled = llm_scaler.fit_transform(llm_train)
        llm_val_scaled = llm_scaler.transform(llm_val)
        llm_test_scaled = llm_scaler.transform(llm_test)

        # --- Prepare test metadata with input_chunk_length offset ---
        test_fx_timestamps = fx_timestamps[input_chunk_length:]
        test_bid_prices = bid_prices[input_chunk_length:]
        test_ask_prices = ask_prices[input_chunk_length:]

        # Unscaled test mid prices for evaluation (with offset)
        test_mid_prices = fx_data_test["mid_price"].values[input_chunk_length:].tolist()

        return ProcessedData(
            darts_train_scaled=darts_train_scaled,
            darts_val_scaled=darts_val_scaled,
            darts_test_scaled=darts_test_scaled,
            darts_scaler=darts_scaler,
            llm_train_scaled=llm_train_scaled,
            llm_val_scaled=llm_val_scaled,
            llm_test_scaled=llm_test_scaled,
            llm_scaler=llm_scaler,
            test_fx_timestamps=test_fx_timestamps,
            test_bid_prices=test_bid_prices,
            test_ask_prices=test_ask_prices,
            test_news_timestamps=news_timestamps,
            test_news_sentiments=news_sentiments,
            test_mid_prices=test_mid_prices,
        )