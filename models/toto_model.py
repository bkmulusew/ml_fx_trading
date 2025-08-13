from types import NoneType
from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from models.toto.toto.data.util.dataset import MaskedTimeseries
from models.toto.toto.inference.forecaster import TotoForecaster
from models.toto.toto.model.toto import Toto

NUM_SAMPLES: int = 512
TIME_INTERVAL_SECONDS: float = 60.0

class TotoFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Toto's pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.forecaster = self.initialize_model()

    def initialize_model(self):
        """Load pre-trained Toto model and compile it"""
        print("Loading Toto model...")

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

    def split_and_scale_data(self, train_ratio=0.5, validation_ratio=0.1):
        """Prepare data with proper train/test split and no data leakage"""
        # Extract raw data
        dates, bid_prices, ask_prices, mid_prices, with_prompt_values, without_prompt_values = self.data_processor.extract_price_time_series()

        # Calculate indices for splitting
        num_observations = len(mid_prices)
        train_end_index = int(num_observations * train_ratio)
        validation_end_index = int(num_observations * (train_ratio + validation_ratio))

        # Split mid price series into train/validation/test
        train_series, val_series, test_series = self._split_mid_price_series(mid_prices, train_end_index, validation_end_index)

        # Fit scaler ONLY on training data to prevent data leakage
        self.scaler.fit(train_data.reshape(-1, 1))

        params = {
            "min_": self.scaler.min_.tolist(),
            "scale_": self.scaler.scale_.tolist(),
            "data_min_": self.scaler.data_min_.tolist(),
            "data_max_": self.scaler.data_max_.tolist(),
            "data_range_": self.scaler.data_range_.tolist(),
            "feature_range": self.scaler.feature_range
        }

        # Process test data
        test_data = self._process_test_data(
            dates=dates[validation_end_index:],
            bid_prices=bid_prices[validation_end_index:],
            ask_prices=ask_prices[validation_end_index:],
            without_prompt=without_prompt_values[validation_end_index:],
            with_prompt=with_prompt_values[validation_end_index:]
        )

        return (*scaled_series, *test_data)
    
    def _split_mid_price_series(self, mid_price, train_end, validation_end):
        """Split the mid price series into train, validation, and test sets."""
        return (
            mid_price[:train_end],
            mid_price[train_end:validation_end],
            mid_price[validation_end:]
        )
    
    def _process_test_data(self, **test_series):
        """Process all test data series by applying the input chunk length offset."""
        return [
            series[self.model_config.INPUT_CHUNK_LENGTH:]
            for series in test_series.values()
        ]
    
    def _scale_series_data(self, train_series, val_series, test_series):
        """Scale the series data using the Scaler."""
        return (
            self.scaler.fit_transform(train_series.reshape(-1, 1)).flatten(),
            self.scaler.transform(val_series.reshape(-1, 1)).flatten(),
            self.scaler.transform(test_series.reshape(-1, 1)).flatten()
        )

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Toto model used in zero-shot mode - skipping training")

    def predict_future_values(self, input_sequences):
        """Make prediction for a batch of input sequences"""

        # Convert to tensor (BATCH_SIZE, INPUT_CHUNK_LENGTH)
        batch_array = np.array(input_sequences, dtype=np.float32)
        batch_tensor = torch.FloatTensor(batch_array).to(self.device)
        timestamp_seconds = torch.zeros_like(batch_tensor).to(self.device)

        time_interval_seconds = torch.full((batch_tensor.size(0),), TIME_INTERVAL_SECONDS, dtype=torch.float32).to(self.device)

        # Wrap in MaskedTimeseries
        inputs = MaskedTimeseries(
            series=batch_tensor,
            padding_mask=torch.full_like(batch_tensor, True, dtype=torch.bool),
            id_mask=torch.zeros_like(batch_tensor, dtype=torch.long),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        # Generate forecast
        with torch.no_grad():
            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                num_samples=NUM_SAMPLES,
                samples_per_batch=NUM_SAMPLES,
            )

        # Extract predictions using median for robustness
        predictions = forecast.median.cpu().numpy().flatten()
        return predictions.tolist()

    def generate_predictions(self, test_series):
        """Generate predictions using sliding window with batching"""

        # Calculate how many predictions we can make
        num_predictions = len(test_series) - self.model_config.INPUT_CHUNK_LENGTH
        
        # Generate predictions sequentially
        scaled_predictions = []

        # Get all valid prediction indices - offset by INPUT_CHUNK_LENGTH to prevent data leakage
        valid_indices = list(range(test_start + self.input_chunk_length, total_points))

        for batch_idx in range(0, len(valid_indices), self.batch_size):
            batch_slice = valid_indices[batch_idx : batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            total_batches = (len(valid_indices) + self.batch_size - 1) // self.batch_size
            
            print(f"Batch {batch_num}/{total_batches} = predicting for the following indices {batch_slice[0]} - {batch_slice[-1]}")

            # Create batch of input sequences
            batch_sequences = []

            for pred_idx in range(batch_start, batch_end):
                # Get the sequence for this prediction
                start_idx = pred_idx
                end_idx = pred_idx + self.model_config.INPUT_CHUNK_LENGTH
                sequence = test_series[start_idx:end_idx]
                batch_sequences.append(sequence)
            
            print(f"Batch {batch_num} = {len(batch_sequences)} sequences of length {len(batch_sequences[0])}")

            batch_predictions = self.predict_future_values(batch_sequences)
            print(f"Batch {batch_num} = {batch_predictions} batch predictions")

            # Store results and log predictions during collection
            for i, global_idx in enumerate(batch_slice):
                prediction = batch_predictions[i]
                true_value = scaled_data[global_idx]

                predictions_scaled.append(prediction)
                prediction_indices.append(global_idx)

                # Inline progress log
                # if global_idx % 100 == 0 or global_idx == test_start:
                print(f"Index {global_idx}: Predicted {prediction:.6f}, True {true_value:.6f}")

        # Inverse transform and align metadata
        predicted_values = self.scaler.inverse_transform(
            np.array(scaled_predictions, dtype=np.float32).reshape(-1, 1)
        ).flatten()

        return predicted_values.tolist()
    
    def get_true_values(self, test_series):
        """Retrieves true values from the test series after scaling back."""
        true_values = self.scaler.inverse_transform(test_series.reshape(-1, 1)).flatten()
        return true_values[self.model_config.INPUT_CHUNK_LENGTH:].tolist()