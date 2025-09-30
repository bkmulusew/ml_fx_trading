import pandas as pd
from darts import TimeSeries

class DataProcessor:
    def __init__(self, model_config):
        self.model_config = model_config

    def load_fx_data(self):
        """Loads the time series data."""
        fx_data_train = pd.read_csv(self.model_config.FX_DATA_PATH_TRAIN)
        fx_data_val = pd.read_csv(self.model_config.FX_DATA_PATH_VAL)
        fx_data_test = pd.read_csv(self.model_config.FX_DATA_PATH_TEST)

        for df in [fx_data_train, fx_data_val, fx_data_test]:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return fx_data_train, fx_data_val, fx_data_test

    def load_news_data(self):
        """Loads the news data."""
        news_data_train = pd.read_csv(self.model_config.NEWS_DATA_PATH_TRAIN)
        news_data_val = pd.read_csv(self.model_config.NEWS_DATA_PATH_VAL)
        news_data_test = pd.read_csv(self.model_config.NEWS_DATA_PATH_TEST)

        for df in [news_data_train, news_data_val, news_data_test]:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return news_data_train, news_data_val, news_data_test

    def prepare_data(self):
        """Extracts price and news related data from the DataFrame."""
        fx_data_train, fx_data_val, fx_data_test = self.load_fx_data()
        news_data_train, news_data_val, news_data_test = self.load_news_data()
        
        # Extract ts data into respective lists
        fx_timestamps = fx_data_test["date"].tolist()
        bid_prices = fx_data_test["bid_price"].tolist()
        ask_prices = fx_data_test["ask_price"].tolist()

        # Extract news data into respective lists
        news_timestamps = news_data_test["date"].tolist()
        news_sentiments = news_data_test[self.model_config.SENTIMENT_SOURCE].tolist()
        
        if self.model_config.MODEL_NAME == 'toto' or self.model_config.MODEL_NAME == 'chronos':
            mid_price_series_train = fx_data_train["mid_price"].values
            mid_price_series_val = fx_data_val["mid_price"].values
            mid_price_series_test = fx_data_test["mid_price"].values
        else:
            mid_price_series_train = TimeSeries.from_dataframe(fx_data_train, value_cols=["mid_price"])
            mid_price_series_val = TimeSeries.from_dataframe(fx_data_val, value_cols=["mid_price"])
            mid_price_series_test = TimeSeries.from_dataframe(fx_data_test, value_cols=["mid_price"])

        return {
            "fx_timestamps": fx_timestamps,
            "news_timestamps": news_timestamps,
            "bid_prices": bid_prices,
            "ask_prices": ask_prices,
            "mid_price_series": {
                "train": mid_price_series_train,
                "val": mid_price_series_val,
                "test": mid_price_series_test,
            },
            "news_sentiments": news_sentiments,
        }