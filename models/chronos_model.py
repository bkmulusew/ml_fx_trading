from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from chronos import ChronosBoltPipeline
import pandas as pd
class ChronosFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Amazon's Chronos, a pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.scaler = MinMaxScaler((0, 1))
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_NAME = "autogluon/chronos-bolt-small"

    def initalize_model(self):
        """Load pre-trained predictor"""
        pipeline = ChronosBoltPipeline.from_pretrained(self.MODEL_NAME)
        return pipeline

    def split_and_scale_data(self):
        """Prepare data for zero-shot forecasting"""
        # Extract raw data
        dates, bid_prices, ask_prices, mid_price_series, with_prompt_values, without_prompt_values = self.data_processor.extract_price_time_series()
        # Scale the mid price series
        
        scaled_mid_price = self.scaler.fit_transform(
            torch.tensor(mid_price_series.univariate_values()).reshape(-1, 1)).flatten()

        # Use last 40% for testing (no training needed for zero-shot)
        test_size = int(len(mid_price_series) * 0.4)
        test_start = len(mid_price_series) - test_size

        # Store metadata for later use
        self.true_values = mid_price_series[test_start:].univariate_values()
        self.test_dates = dates[test_start:]
        self.test_bid_prices = bid_prices[test_start:]
        self.test_ask_prices = ask_prices[test_start:]
        self.test_with_prompt = with_prompt_values[test_start:]
        self.test_without_prompt = without_prompt_values[test_start:]

        # Return the scaled data including lookback window
        return scaled_mid_price[test_start - self.model_config.INPUT_CHUNK_LENGTH:]

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Model used in zero-shot mode - skipping training")
        return None

    def predict_future_values(self):
        print(f"Making predictions...")
        all_predictions = []


        return None

    def generate_predictions(self):
        """Generate predictions for the test set"""
        print("Preparing test data...")
        import pandas as pd
        
        # Get scaled test data
        test_data = self.split_and_scale_data()

        # pred_list = self.predict_future_values(test_data)

        df = pd.DataFrame({
            'timestamp':self.test_dates,
            'target': test_data[-len(self.test_dates):],
            'item_id': 'USD_CYN'
        }
        )
        test_data = TimeSeriesDataFrame.from_data_frame(df)
        predictor = self.initalize_model(df)

        predictions = []
        test_length = len(test_data)
        
        for i in range(self.model_config.INPUT_CHUNK_LENGTH, test_length):
            start_idx = i - self.model_config.INPUT_CHUNK_LENGTH
            end_idx = i
        
            context_window = test_data.slice_by_timestep(start_idx, end_idx)
        
            # Make prediction for this window
            prediction = predictor.predict(context_window, model='Chronos[bolt_base]')
            prediction = prediction.slice_by_timestep(0, 1)
            predictions.append(prediction)


        predictions = TimeSeriesDataFrame(pd.concat(predictions, ignore_index=False))
        # Make predictions
        scaled_predictions = predictions['mean'].values
        scaled_predictions = torch.tensor(scaled_predictions, device=self.device)
        print("Inverse transforming predictions...")
        # Inverse transform predictions to original scale
        predictions = self.scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)).flatten()

        # Ensure we have matching lengths
        min_length = min(len(predictions), len(self.true_values))
        print(len(predictions), len(self.true_values))
        
        predictions = predictions[:min_length]
        true_values = self.true_values[:min_length]

        print(f"Generated {len(predictions)} predictions")

        return {
            'predicted_values': predictions.tolist(),
            'true_values': true_values,
            'test_dates': self.test_dates[:min_length],
            'test_bid_prices': self.test_bid_prices[:min_length],
            'test_ask_prices': self.test_ask_prices[:min_length],
            'test_with_prompt': self.test_with_prompt[:min_length],
            'test_without_prompt': self.test_without_prompt[:min_length]
        }
