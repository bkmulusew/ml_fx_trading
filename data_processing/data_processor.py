import pandas as pd
from darts import TimeSeries
import numpy as np

class DataProcessor:
    def __init__(self, model_config):
        self.model_config = model_config
        self.data_df = None

    def load_and_prepare_data(self):
        """Loads the CSV file and calculates the ratio if not already done."""
        if self.data_df is None:
            self.data_df = pd.read_csv(self.model_config.DATA_FILE_PATH)
        return self.data_df

    def get_ratio_time_series(self):
        """Converts the ratio column to a time series object."""
        df = self.load_and_prepare_data()
        # Select the second column dynamically
        # date_column = df.columns[0]
        # fx_column = df.columns[1]
        # without_prompt_column = df.columns[2]
        # with_prompt_column = df.columns[3]
        # dates_list = df[date_column].tolist()
        # without_prompt_list = df[without_prompt_column].tolist()
        # with_prompt_list = df[with_prompt_column].tolist()
        # return dates_list, TimeSeries.from_dataframe(df, value_cols=[fx_column]), with_prompt_list, without_prompt_list

        # ------------

        date_column = df.columns[0]
        ask_price_column = df.columns[1]
        bid_price_column = df.columns[2]
        mid_price_column = df.columns[3]
        without_prompt_column = df.columns[5]
        with_prompt_column = df.columns[6]
        dates_list = df[date_column].tolist()
        bid_prices_list = df[bid_price_column].tolist()
        ask_prices_list = df[ask_price_column].tolist()
        without_prompt_list = df[without_prompt_column].tolist()
        with_prompt_list = df[with_prompt_column].tolist()
        return dates_list, bid_prices_list, ask_prices_list, TimeSeries.from_dataframe(df, value_cols=[mid_price_column]), with_prompt_list, without_prompt_list

    def get_test_columns(self, test_size):
        """Retrieves the test dataset's numerator and denominator columns."""
        df = self.load_and_prepare_data()
        # Select the second column dynamically
        second_column_name = df.columns[1]
        return ((df[second_column_name]).values)[-test_size:], np.ones(test_size)