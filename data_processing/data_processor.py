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
        with_prompt_values = df["with_prompt"].tolist()
        without_prompt_values = df["without_prompt"].tolist()
        
        if self.model_config.MODEL_NAME == 'toto':
            mid_price_series = df["mid_price"].values
        else:
            mid_price_series = TimeSeries.from_dataframe(df, value_cols=["mid_price"])

        LLM_data = pd.read_csv("dataset/Zero Shot News File_09-00-16-30.csv")
        LLM_dates = LLM_data["Time"].tolist()
        expert_prompt_label = LLM_data["Expert Prompt Label"].tolist()
        naive_prompt_label = LLM_data["Naive Prompt Label"].tolist()
        competitior_label = LLM_data["Competitor Label"].tolist()
        naive_converted_prompt_label = LLM_data["Naive + Converted Prompt Label"].tolist()

        date_to_label = dict(zip(LLM_dates, naive_converted_prompt_label))

        labels = [date_to_label.get(d, 0) for d in dates]

        llm_label = [0, 0, 0]

        for i in range(3, len(labels)):
            llm_label.append(labels[i-3])

        return (
            dates,
            bid_prices,
            ask_prices,
            mid_price_series,
            with_prompt_values,
            without_prompt_values,
            llm_label
        )