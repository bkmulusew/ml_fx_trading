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
    
        # Define column names for clarity
        columns = {
            'date': df.columns[0],
            'bid': df.columns[1],
            'ask': df.columns[2],
            'mid': df.columns[3],
            'without_prompt': df.columns[4],
            'with_prompt': df.columns[5]
        }
        
        # Extract data into respective lists
        dates = df[columns['date']].tolist()
        bid_prices = df[columns['bid']].tolist()
        ask_prices = df[columns['ask']].tolist()
        with_prompt_values = df[columns['with_prompt']].tolist()
        without_prompt_values = df[columns['without_prompt']].tolist()
        
        if self.model_config.MODEL_NAME == 'toto':
            mid_price_series = df[columns['mid']].values
        else:
            mid_price_series = TimeSeries.from_dataframe(df, value_cols=[columns['mid']])

        return (
            dates,
            bid_prices,
            ask_prices,
            mid_price_series,
            with_prompt_values,
            without_prompt_values
        )