from models import FinancialForecastingModel
import numpy as np
import torch
from chronos import ChronosBoltPipeline

class ChronosFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Amazon's Chronos-bolt, a pre-trained transformer for zero-shot forecasting"""
    def __init__(self, fx_trading_config, scaler):
        self.fx_trading_config = fx_trading_config
        self.scaler = scaler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_NAME = "autogluon/chronos-bolt-base"
        self.PRESET_NAME = "bolt_base"
        self.forecaster = self.initialize_model()

    def initialize_model(self):
        """Load pre-trained predictor"""
        try:
            print(f"\nLoading Chronos-Bolt model...")
            pipeline = ChronosBoltPipeline.from_pretrained(self.MODEL_NAME)
            print("Chronos-Bolt model loaded successfully!")
            return pipeline
        except Exception as e:
            print(f"Error initializing Chronos model: {e}")

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
                quantiles, mean = self.forecaster.predict_quantiles(inputs, prediction_length=self.fx_trading_config.OUTPUT_CHUNK_LENGTH)

            predictions = mean.cpu().numpy()
            return predictions

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return [seq[-1] for seq in input_sequences]

    def generate_predictions(self, data):
        """Generate predictions using sliding window with batching"""

        # data shape: (time_steps, 1)
        num_timesteps, _ = data.shape

        # Calculate how many predictions we can make
        num_predictions = num_timesteps - self.fx_trading_config.INPUT_CHUNK_LENGTH

        if num_predictions <= 0:
            raise ValueError(f"Not enough data points. Need at least {self.fx_trading_config.INPUT_CHUNK_LENGTH + 1} timesteps, got {num_timesteps}")

        # Allocate storage for predictions
        all_predictions = np.empty(num_predictions, dtype=np.float32)

        for batch_idx in range(0, num_predictions, self.fx_trading_config.EVAL_BATCH_SIZE):
            # Calculate batch boundaries
            batch_start = batch_idx
            batch_end = min(batch_idx + self.fx_trading_config.EVAL_BATCH_SIZE, num_predictions)

            # Create batch of input sequences
            indices = np.arange(batch_start, batch_end)[:, None] + np.arange(self.fx_trading_config.INPUT_CHUNK_LENGTH)
            batch_input = data[indices]  # Shape: (EVAL_BATCH_SIZE, INPUT_CHUNK_LENGTH, 1)

            predictions = self.predict_future_values(batch_input)
            all_predictions[batch_start:batch_end] = predictions[:, 0]

        # Inverse transform
        predictions_reshaped = all_predictions.reshape(-1, 1)
        predicted_mid_prices = self.scaler.inverse_transform(predictions_reshaped).ravel().tolist()

        return predicted_mid_prices