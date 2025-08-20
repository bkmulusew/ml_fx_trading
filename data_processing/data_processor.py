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
        bid_prices = df["bid_price"].values
        ask_prices = df["ask_price"].values
        with_prompt_values = df["with_prompt"].tolist()
        without_prompt_values = df["without_prompt"].tolist()

        spread = (ask_prices - bid_prices)
        
        if self.model_config.MODEL_NAME == 'toto':
            mid_price_series = df["mid_price"].values
        else:
            mid_price_series = TimeSeries.from_dataframe(df, value_cols=["mid_price"])

        return (
            dates,
            bid_prices,
            ask_prices,
            spread,
            mid_price_series,
            with_prompt_values,
            without_prompt_values
        )