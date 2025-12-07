from models import FinancialForecastingModel
import numpy as np
import torch
from chronos import ChronosBoltPipeline

class ChronosFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Amazon's Chronos-bolt, a pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_NAME = "autogluon/chronos-bolt-base"
        self.forecaster = self.initialize_model()

    def initialize_model(self):
        try:
            print(f"\nLoading Chronos-Bolt model...")
            pipeline = ChronosBoltPipeline.from_pretrained(self.MODEL_NAME)
            print("Chronos-Bolt model loaded successfully!")
            return pipeline
        except Exception as e:
            print(f"Error initializing Chronos model: {e}")

    def split_and_scale_data(self, train_ratio=0.5, validation_ratio=0.1):
        data = self.data_processor.prepare_data()

        fx_timestamps = data["fx_timestamps"]
        news_timestamps = data["news_timestamps"]
        bid_prices = data["bid_prices"]
        ask_prices = data["ask_prices"]
        news_sentiments = data["news_sentiments"]

        mid_prices = data["mid_price_series"]
        test_mid_prices = mid_prices["test"]

        self.test_mid_prices = test_mid_prices[self.model_config.INPUT_CHUNK_LENGTH:].tolist()

        X_test = test_mid_prices.reshape(-1, 1).astype(np.float32)

        meta = self._align_test_targets(
            fx_timestamps=fx_timestamps,
            bid_prices=bid_prices,
            ask_prices=ask_prices,
        )

        return (X_test, *meta, news_timestamps, news_sentiments)

    def _align_test_targets(self, **test_series):
        """Process all test data series by applying the input chunk length offset."""
        return [
            series[self.model_config.INPUT_CHUNK_LENGTH:]
            for series in test_series.values()
        ]    

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Model used in zero-shot mode - skipping training")
        return None

    def predict_future_values(self, input_sequences):
        """Make prediction for a batch of input sequences"""

        # Convert to tensor (EVAL_BATCH_SIZE, INPUT_CHUNK_LENGTH)
        batch_array = np.array(input_sequences, dtype=np.float32).squeeze(axis=-1)
        inputs = torch.FloatTensor(batch_array).to(self.device)

        try:
            with torch.no_grad():
                quantiles, mean = self.forecaster.predict_quantiles(
                    inputs, 
                    prediction_length=self.model_config.OUTPUT_CHUNK_LENGTH
                )
                
            predictions = mean.cpu().numpy()
            return predictions

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return [seq[-1] for seq in input_sequences]

    def generate_predictions(self, data):
        """Generate predictions using sliding window with batching on raw data"""

        # data shape: (time_steps, 1)
        num_timesteps, _ = data.shape

        num_predictions = num_timesteps - self.model_config.INPUT_CHUNK_LENGTH

        if num_predictions <= 0:
            raise ValueError(f"Not enough data points. Need at least {self.model_config.INPUT_CHUNK_LENGTH + 1} timesteps, got {num_timesteps}")
        
        all_predictions = np.empty(num_predictions, dtype=np.float32)

        for batch_idx in range(0, num_predictions, self.model_config.EVAL_BATCH_SIZE):
            batch_start = batch_idx
            batch_end = min(batch_idx + self.model_config.EVAL_BATCH_SIZE, num_predictions)

            indices = np.arange(batch_start, batch_end)[:, None] + np.arange(self.model_config.INPUT_CHUNK_LENGTH)
            batch_input = data[indices]  # Shape: (EVAL_BATCH_SIZE, INPUT_CHUNK_LENGTH, 1)
            
            predictions = self.predict_future_values(batch_input)
            all_predictions[batch_start:batch_end] = predictions[:, 0]

        predicted_mid_prices = all_predictions.tolist()

        return predicted_mid_prices
