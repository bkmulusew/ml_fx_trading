from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from chronos import BaseChronosPipeline, Chronos2Pipeline
from datetime import datetime, timedelta

class Chronos2FinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Amazon's Chronos-bolt, a pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_NAME = "s3://autogluon/chronos-2/"
        self.forecaster = self.initialize_model()


    def initialize_model(self):
        """Load pre-trained predictor"""
        try:
            print(f"\nLoading Chronos-Bolt model...")
            pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(self.MODEL_NAME, device_map="cuda")
            print("Chronos-Bolt model loaded successfully!")
            return pipeline
        except Exception as e:
            print(f"Error initializing Chronos model: {e}")

    def split_and_scale_data(self, train_ratio=0.5, validation_ratio=0.1):
        """Prepare data with proper train/test split and no data leakage"""
        data = self.data_processor.prepare_data()

        fx_timestamps = data["fx_timestamps"]
        news_timestamps = data["news_timestamps"]
        bid_prices = data["bid_prices"]
        ask_prices = data["ask_prices"]
        news_sentiments = data["news_sentiments"]

        mid_prices = data["mid_price_series"]
        train_mid_prices = mid_prices["train"]
        val_mid_prices = mid_prices["val"]
        test_mid_prices = mid_prices["test"]

        self.test_mid_prices = test_mid_prices[self.model_config.INPUT_CHUNK_LENGTH:].tolist()

        # Reshape for scaling (T, 1)
        X_train = train_mid_prices.reshape(-1, 1).astype(np.float32)
        X_val = val_mid_prices.reshape(-1, 1).astype(np.float32)
        X_test = test_mid_prices.reshape(-1, 1).astype(np.float32)

        # Scale data - fit scaler on TRAIN ONLY
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Process test data
        meta = self._align_test_targets(
            fx_timestamps=fx_timestamps,
            bid_prices=bid_prices,
            ask_prices=ask_prices,
        )

        return (X_test_scaled, *meta, news_timestamps, news_sentiments)

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

            batch_array = np.array(input_sequences, dtype=np.float32)

            batch_array = batch_array.transpose(0, 2, 1)

            inputs = torch.FloatTensor(batch_array) # shape (EVAL_BATCH_SIZE, 1, INPUT_LEN)
            try:
                with torch.no_grad():
                    quantiles, mean = self.forecaster.predict_quantiles(
                        inputs,
                        prediction_length=self.model_config.OUTPUT_CHUNK_LENGTH
                    )
                    predictions = np.array(mean).squeeze(axis=1)
                    return predictions

            except Exception as e:
                print(f"Error during prediction: {e}")
                return np.array([seq[-1] for seq in input_sequences])

    def generate_predictions(self, data):
        """Generate predictions using sliding window with batching"""
        import time
        start_time = time.time()
        print("Starting prediction generation...")
        num_timesteps, _ = data.shape
        num_predictions = num_timesteps - self.model_config.INPUT_CHUNK_LENGTH

        if num_predictions <= 0:
            raise ValueError(f"Not enough data points. Need at least {self.model_config.INPUT_CHUNK_LENGTH + 1} timesteps, got {num_timesteps}")
        
        all_predictions = np.empty(num_predictions, dtype=np.float32)

        for batch_idx in range(0, num_predictions, self.model_config.EVAL_BATCH_SIZE):
            batch_start = batch_idx
            batch_end = min(batch_idx + self.model_config.EVAL_BATCH_SIZE, num_predictions)
            if batch_idx % self.model_config.EVAL_BATCH_SIZE*1000 == 0:
                elapsed = time.time() - start_time
                print(f"Processing batch {batch_idx} - {batch_end} / {num_predictions}. Elapsed time: {elapsed:.2f}s")
                
            indices = np.arange(batch_start, batch_end)[:, None] + np.arange(self.model_config.INPUT_CHUNK_LENGTH)
            batch_input = data[indices]
            predictions = self.predict_future_values(batch_input)

            all_predictions[batch_start:batch_end] = predictions[:, self.model_config.OUTPUT_CHUNK_LENGTH - 1]

        predictions_reshaped = all_predictions.reshape(-1, 1)
        predicted_mid_prices = self.scaler.inverse_transform(predictions_reshaped).ravel().tolist()

        return predicted_mid_prices