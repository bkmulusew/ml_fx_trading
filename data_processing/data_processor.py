import pandas as pd
from darts import TimeSeries

class DataProcessor:
    def __init__(self, model_config):
        self.model_config = model_config

    def load_data(self):
        """Loads the CSV file and calculates the ratio if not already done."""
        data_df_train = pd.read_csv(self.model_config.DATA_PATH_TRAIN)
        data_df_val = pd.read_csv(self.model_config.DATA_PATH_VAL)
        data_df_test = pd.read_csv(self.model_config.DATA_PATH_TEST)
        return data_df_train, data_df_val, data_df_test

    def prepare_fx_data(self):
        """Extracts price related data from the DataFrame."""
        data_df_train, data_df_val, data_df_test = self.load_data()
        
        # Extract data into respective lists
        dates = data_df_test["date"].tolist()
        bid_prices = data_df_test["bid_price"].tolist()
        ask_prices = data_df_test["ask_price"].tolist()
        news_sentiments = data_df_test[self.model_config.SENTIMENT_SOURCE].tolist()
        
        if self.model_config.MODEL_NAME == 'toto':
            mid_price_series_train = data_df_train["mid_price"].values
            mid_price_series_val = data_df_val["mid_price"].values
            mid_price_series_test = data_df_test["mid_price"].values
        else:
            mid_price_series_train = TimeSeries.from_dataframe(data_df_train, value_cols=["mid_price"])
            mid_price_series_val = TimeSeries.from_dataframe(data_df_val, value_cols=["mid_price"])
            mid_price_series_test = TimeSeries.from_dataframe(data_df_test, value_cols=["mid_price"])

        return {
            "dates": dates,
            "bid_prices": bid_prices,
            "ask_prices": ask_prices,
            "mid_price_series": {
                "train": mid_price_series_train,
                "val": mid_price_series_val,
                "test": mid_price_series_test,
            },
            "news_sentiments": news_sentiments,
        }