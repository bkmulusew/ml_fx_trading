from models import FinancialForecastingModel
import numpy as np
import torch
from models.toto.toto.data.util.dataset import MaskedTimeseries
from models.toto.toto.inference.forecaster import TotoForecaster
from models.toto.toto.model.toto import Toto

NUM_SAMPLES: int = 64
TIME_INTERVAL_SECONDS: float = 60.0
class TotoFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Toto's pre-trained transformer for zero-shot forecasting"""
    def __init__(self, fx_trading_config, scaler):
        self.fx_trading_config = fx_trading_config
        self.scaler = scaler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.forecaster = self.initialize_model()

    def initialize_model(self):
        """Load pre-trained Toto model and compile it"""
        print("\nLoading Toto model...")

        try:
            toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
            toto.to(self.device)

            # Compile for better performance (optional)
            try:
                toto.compile()
                print("Model compiled successfully")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")

            # Initialize forecaster with the underlying model
            forecaster = TotoForecaster(toto.model)

            print("Toto model loaded successfully!")

            return forecaster

        except Exception as e:
            print(f"Error loading Toto model: {e}")
            raise

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Toto model used in zero-shot mode - skipping training")

    def predict_future_values(self, input_sequences):
        """Make prediction for a batch of input sequences"""

        # input_sequences shape: (EVAL_BATCH_SIZE, INPUT_CHUNK_LENGTH, 1)
        batch_size, _, num_features = input_sequences.shape

        # Toto expects (batch, variates, time_steps)
        # Transpose from (batch, time_steps, features) to (batch, features, time_steps)
        batch_tensor = torch.FloatTensor(input_sequences).transpose(1, 2).to(self.device)

        # Create timestamp and time interval tensors
        timestamp_seconds = torch.zeros_like(batch_tensor).to(self.device)
        time_interval_seconds = torch.full(
            (batch_size, num_features),  # Only one feature now
            TIME_INTERVAL_SECONDS,
            dtype=torch.float32
        ).to(self.device)

        # Create MaskedTimeseries with univariate input
        inputs = MaskedTimeseries(
            series=batch_tensor,
            padding_mask=torch.full_like(batch_tensor, True, dtype=torch.bool),
            id_mask=torch.zeros_like(batch_tensor, dtype=torch.long),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        with torch.no_grad():
            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                num_samples=NUM_SAMPLES,
                samples_per_batch=NUM_SAMPLES,
            )

        # Extract predictions using median for robustness
        # forecast.median shape: (batch, variates, prediction_length)
        predictions = forecast.median.cpu().numpy()

        # Extract mid price predictions (only variate now)
        mid_predictions = predictions[:, 0, 0]  # Shape: (batch_size,)

        return mid_predictions

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
            all_predictions[batch_start:batch_end] = predictions

        # Inverse transform
        predictions_reshaped = all_predictions.reshape(-1, 1)
        predicted_mid_prices = self.scaler.inverse_transform(predictions_reshaped).ravel().tolist()

        return predicted_mid_prices