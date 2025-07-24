from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from models.toto.toto.data.util.dataset import MaskedTimeseries
from models.toto.toto.inference.forecaster import TotoForecaster
from models.toto.toto.model.toto import Toto


class TotoFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Toto's pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.scaler = MinMaxScaler((0, 1))
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading Toto model...")
        # Initialize model
        self.toto = self.initalize_model()
        self.model = self.toto.model  # Extract the underlying model

        # Initialize forecaster with the underlying model
        self.forecaster = TotoForecaster(self.model)
        print("Toto model loaded successfully!")

    def initalize_model(self):
        """Load pre-trained Toto model and compile it"""
        toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
        toto.to(self.device)

        # Compile for better performance (optional)
        try:
            toto.compile()
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")

        return toto

    def split_and_scale_data(self):
        """Prepare data for zero-shot forecasting"""
        # Extract raw data
        dates, bid_prices, ask_prices, mid_price_series, with_prompt_values, without_prompt_values = self.data_processor.extract_price_time_series()

        # Scale the mid price series
        scaled_mid_price = self.scaler.fit_transform(
            mid_price_series.reshape(-1, 1)).flatten()

        # Use last 40% for testing (no training needed for zero-shot)
        test_size = int(len(mid_price_series) * 0.4)
        test_start = len(mid_price_series) - test_size

        # Store metadata for later use
        self.true_values = mid_price_series[test_start:].tolist()
        self.test_dates = dates[test_start:]
        self.test_bid_prices = bid_prices[test_start:]
        self.test_ask_prices = ask_prices[test_start:]
        self.test_with_prompt = with_prompt_values[test_start:]
        self.test_without_prompt = without_prompt_values[test_start:]

        # Return the scaled data including lookback window
        return scaled_mid_price[test_start - self.model_config.INPUT_CHUNK_LENGTH:]

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Toto model used in zero-shot mode - skipping training")
        return None

    def predict_future_values(self, input_sequences):
        """Make predictions using the pre-trained model - batch processing like in notebook"""
        print(f"Making predictions for {len(input_sequences)} sequences...")
        all_predictions = []

        # Convert to tensor
        full_input = torch.FloatTensor(input_sequences).to(self.device)
        batch_size = 1024 if self.device.type == 'cuda' else 256
        num_batches = (full_input.size(0) + batch_size - 1) // batch_size

        print(f"Processing {num_batches} batches of size {batch_size}")

        for batch_idx in range(0, full_input.size(0), batch_size):
            if batch_idx % (batch_size * 20) == 0:  # Progress update every 20 batches
                progress = (batch_idx // batch_size + 1)
                print(f"Processing batch {progress}/{num_batches}")

            # shape: (batch_size, sequence_length)
            batch_series = full_input[batch_idx:batch_idx+batch_size]

            # Optional metadata
            timestamp_seconds = torch.zeros_like(batch_series).to(self.device)
            time_interval_seconds = torch.full(
                (batch_series.size(0),), 60).to(self.device)  # 1-minute intervals

            # Wrap in MaskedTimeseries
            inputs = MaskedTimeseries(
                series=batch_series,
                padding_mask=torch.full_like(
                    batch_series, True, dtype=torch.bool),
                id_mask=torch.zeros_like(batch_series),
                timestamp_seconds=timestamp_seconds,
                time_interval_seconds=time_interval_seconds,
            )

            # Generate forecast
            with torch.no_grad():
                forecast = self.forecaster.forecast(
                    inputs,
                    prediction_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                    num_samples=self.model_config.NUM_SAMPLES,
                    samples_per_batch=self.model_config.NUM_SAMPLES,
                )

            # Extract predictions and add to list
            batch_predictions = forecast.median.view(-1).tolist()
            all_predictions.extend(batch_predictions)

        print("Prediction completed!")
        return np.array(all_predictions)

    def generate_predictions(self):
        """Generate predictions for the test set"""
        print("Preparing test data...")
        # Get scaled test data
        test_data = self.split_and_scale_data()

        # Prepare sliding windows for prediction
        input_sequences = []
        for i in range(self.model_config.INPUT_CHUNK_LENGTH, len(test_data)):
            sequence = test_data[i - self.model_config.INPUT_CHUNK_LENGTH:i]
            input_sequences.append(sequence)

        if not input_sequences:
            raise ValueError(
                "No valid input sequences found. Check INPUT_CHUNK_LENGTH and test data size.")

        input_sequences = np.array(input_sequences)
        print(f"Created {len(input_sequences)} input sequences")

        # Make predictions
        scaled_predictions = self.predict_future_values(input_sequences)

        print("Inverse transforming predictions...")
        # Inverse transform predictions to original scale
        predictions = self.scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)).flatten()

        # Ensure we have matching lengths
        min_length = min(len(predictions), len(self.true_values))
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
