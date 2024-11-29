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
        first_column_name = df.columns[0]
        second_column_name = df.columns[1]
        dates = df[first_column_name].tolist()
        return dates, TimeSeries.from_dataframe(df, value_cols=[second_column_name])

    def get_test_columns(self, test_size):
        """Retrieves the test dataset's numerator and denominator columns."""
        df = self.load_and_prepare_data()
        # Select the second column dynamically
        second_column_name = df.columns[1]
        return ((df[second_column_name]).values)[-test_size:], np.ones(test_size)