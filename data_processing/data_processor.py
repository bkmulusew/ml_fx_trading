import pandas as pd
from darts import TimeSeries

class DataProcessor:
    def __init__(self, model_config):
        self.model_config = model_config
        self.data_df = None

    def load_and_prepare_data(self):
        """Loads the CSV file and calculates the ratio if not already done."""
        if self.data_df is None:
            self.data_df = pd.read_csv(self.model_config.DATA_FILE_PATH)
        return self.data_df

    def extract_price_time_series(self):
        """Extracts price related data from the DataFrame."""
        
        df = self.load_and_prepare_data()
        
        # Extract data into respective lists
        dates = df["date"].tolist()
        bid_prices = df["bid_price"].tolist()
        ask_prices = df["ask_price"].tolist()
        news_sentiments = df["competitor_label"].tolist()

        if self.model_config.MODEL_NAME == 'toto' or self.model_config.MODEL_NAME == 'chronos':
            mid_price_series = df["mid_price"].values
        else:
            mid_price_series = TimeSeries.from_dataframe(df, value_cols=["mid_price"])

        return (
            dates,
            bid_prices,
            ask_prices,
            mid_price_series,
            news_sentiments
        )