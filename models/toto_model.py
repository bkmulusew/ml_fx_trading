from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from models.toto.toto.data.util.dataset import MaskedTimeseries
from models.toto.toto.inference.forecaster import TotoForecaster
from models.toto.toto.model.toto import Toto
from datetime import datetime


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
        self.toto = self.initialize_model()
        self.model = self.toto.model  # Extract the underlying model

        # Initialize forecaster with the underlying model
        self.forecaster = TotoForecaster(self.model)
        print("Toto model loaded successfully!")

    def initialize_model(self):
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
        """Prepare data with proper train/test split and no data leakage"""
        # Extract raw data
        dates, bid_prices, ask_prices, mid_price_series, with_prompt_values, without_prompt_values = self.data_processor.extract_price_time_series()

        # Split data first, then scale to avoid data leakage
        train_size = int(len(mid_price_series) * 0.6)

        print(f"Total data points: {len(mid_price_series)}")
        print(f"Training data points: {train_size}")
        print(f"Test data points: {len(mid_price_series) - train_size}")

        # Split the data
        train_data = mid_price_series[:train_size-128]
        test_data = mid_price_series[train_size - 128:]

        # Fit scaler ONLY on training data
        self.scaler.fit(train_data.reshape(-1, 1))

        # Scale training and test data separately
        train_scaled = self.scaler.transform(
            train_data.reshape(-1, 1)).flatten()
        test_scaled = self.scaler.transform(test_data.reshape(-1, 1)).flatten()

        # Combine for easier indexing during prediction
        self.full_scaled_data = np.concatenate([train_scaled, test_scaled])

        # Store metadata
        self.train_size = train_size
        self.test_start_idx = train_size
        self.original_mid_prices = mid_price_series

        # Test metadata (already properly indexed)
        self.test_dates = dates[train_size:]
        self.test_bid_prices = bid_prices[train_size:]
        self.test_ask_prices = ask_prices[train_size:]
        self.test_with_prompt = with_prompt_values[train_size:]
        self.test_without_prompt = without_prompt_values[train_size:]
        self.test_true_values = test_data  # Original scale

        print(
            f"Scaler fitted on training data - Min: {train_data.min():.6f}, Max: {train_data.max():.6f}")
        print(
            f"Test data range - Min: {test_data.min():.6f}, Max: {test_data.max():.6f}")

        return self.full_scaled_data

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Toto model used in zero-shot mode - skipping training")
        return None

    def create_realistic_timestamps(self, batch_size, sequence_length, start_time=None):
        """Create realistic timestamps for 1-minute interval data"""
        if start_time is None:
            start_time = int(datetime(2023, 1, 1).timestamp())

        timestamps = torch.zeros(batch_size, sequence_length).to(self.device)

        for i in range(batch_size):
            for j in range(sequence_length):
                timestamps[i, j] = start_time + (i * sequence_length + j) * 60

        return timestamps

    def predict_future_values(self, input_sequences):
        """Make predictions using the pre-trained model - wrapper for abstract method compliance"""
        print("+++++++++++++++++++++++")
        if isinstance(input_sequences, (list, np.ndarray)):
            input_sequences = np.array(input_sequences)

            if input_sequences.ndim == 1:
                input_sequences = input_sequences.reshape(1, -1)

            return self.predict_batch(input_sequences)
        else:
            raise ValueError("Input sequences must be a list or numpy array")

    def predict_batch(self, input_sequences):
        """Make predictions for a batch of input sequences"""
        batch_size = input_sequences.shape[0]
        sequence_length = input_sequences.shape[1]

        print(
            f"Predicting batch of size {batch_size} with sequence length {sequence_length}")

        # Convert to tensor and move to device
        batch_tensor = torch.FloatTensor(input_sequences).to(self.device)

        # Create realistic timestamps
        timestamp_seconds = self.create_realistic_timestamps(
            batch_size, sequence_length)
        time_interval_seconds = torch.full((batch_size,), 60.0).to(self.device)

        # Wrap in MaskedTimeseries
        inputs = MaskedTimeseries(
            series=batch_tensor,
            padding_mask=torch.full_like(batch_tensor, True, dtype=torch.bool),
            id_mask=torch.zeros_like(batch_tensor),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        try:
            # Generate forecast
            with torch.no_grad():
                forecast = self.forecaster.forecast(
                    inputs,
                    prediction_length=1,
                    num_samples=16,
                    samples_per_batch=16,
                )

            # Extract predictions using median for robustness
            if hasattr(forecast, 'median'):
                predictions = forecast.median.cpu().numpy()
            else:
                predictions = forecast.samples.median(dim=0)[0].cpu().numpy()

            print(f"Generated predictions shape: {predictions.shape}")
            return predictions.flatten()

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return zeros as fallback
            return np.zeros(batch_size * 1)

    def generate_predictions(self):
        """Generate predictions for the test set with careful alignment"""
        print("Starting zero-shot forecasting...")

        # Get scaled data
        scaled_data = self.split_and_scale_data()

        input_chunk_length = 128
        batch_size = 512

        # Calculate valid prediction range
        # We need input_chunk_length points before each prediction
        # First valid prediction is at test_start_idx (predicting this point using previous input_chunk_length points)
        first_prediction_idx = self.test_start_idx
        # Can predict up to the last point
        last_prediction_idx = len(scaled_data) - 1

        total_possible_predictions = last_prediction_idx - first_prediction_idx + 1

        print(f"Input chunk length: {input_chunk_length}")
        print(f"Test starts at index: {self.test_start_idx}")
        print(f"First prediction index: {first_prediction_idx}")
        print(f"Last prediction index: {last_prediction_idx}")
        print(f"Total possible predictions: {total_possible_predictions}")

        if total_possible_predictions <= 0:
            raise ValueError(
                "No valid predictions possible with current data split")

        all_predictions_scaled = []
        prediction_test_indices = []  # Track which test indices these correspond to

        # Process in batches
        for batch_start in range(0, total_possible_predictions, batch_size):
            batch_end = min(batch_start + batch_size,
                            total_possible_predictions)
            current_batch_size = batch_end - batch_start

            print(
                f"Processing batch {batch_start//batch_size + 1}: predictions {batch_start} to {batch_end-1}")

            batch_sequences = []
            batch_test_indices = []

            for i in range(batch_start, batch_end):
                prediction_idx = first_prediction_idx + i

                # Get input sequence: the input_chunk_length points before prediction_idx
                start_idx = prediction_idx - input_chunk_length
                end_idx = prediction_idx

                if start_idx >= 0:
                    sequence = scaled_data[start_idx:end_idx]
                    batch_sequences.append(sequence)
                    # This prediction corresponds to test index i
                    batch_test_indices.append(i)

            if not batch_sequences:
                continue

            batch_sequences = np.array(batch_sequences)

            try:
                # Make predictions (in scaled space)
                batch_predictions_scaled = self.predict_batch(batch_sequences)

                # Handle output shape
                if batch_predictions_scaled.ndim > 1:
                    batch_predictions_scaled = batch_predictions_scaled.flatten()

                # If we got multiple predictions per sequence, take the first
                expected_length = len(batch_sequences)
                if len(batch_predictions_scaled) > expected_length:
                    batch_predictions_scaled = batch_predictions_scaled[:expected_length]

                all_predictions_scaled.extend(batch_predictions_scaled)
                prediction_test_indices.extend(batch_test_indices)

                print(
                    f"Batch generated {len(batch_predictions_scaled)} predictions")

            except Exception as e:
                print(f"Error in batch: {e}")
                # Use fallback - predict no change (current value)
                fallback_predictions = [scaled_data[first_prediction_idx + batch_start + j]
                                        for j in range(current_batch_size)]
                all_predictions_scaled.extend(fallback_predictions)
                prediction_test_indices.extend(batch_test_indices)

        if not all_predictions_scaled:
            raise ValueError("No predictions generated!")

        print(f"Total predictions generated: {len(all_predictions_scaled)}")

        # Convert scaled predictions back to original scale
        predictions_array = np.array(all_predictions_scaled).reshape(-1, 1)
        predictions_original = self.scaler.inverse_transform(
            predictions_array).flatten()

        # Align with test data
        aligned_predictions = []
        aligned_true_values = []
        aligned_dates = []
        aligned_bid_prices = []
        aligned_ask_prices = []
        aligned_with_prompt = []
        aligned_without_prompt = []

        for i, test_idx in enumerate(prediction_test_indices):
            if test_idx < len(self.test_true_values) and i < len(predictions_original):
                aligned_predictions.append(predictions_original[i])
                aligned_true_values.append(self.test_true_values[test_idx])
                aligned_dates.append(self.test_dates[test_idx])
                aligned_bid_prices.append(self.test_bid_prices[test_idx])
                aligned_ask_prices.append(self.test_ask_prices[test_idx])
                aligned_with_prompt.append(self.test_with_prompt[test_idx])
                aligned_without_prompt.append(
                    self.test_without_prompt[test_idx])

        print(f"Final aligned predictions: {len(aligned_predictions)}")

        # Validation checks
        if aligned_predictions and aligned_true_values:
            # Calculate basic metrics
            mse = np.mean((np.array(aligned_predictions) -
                          np.array(aligned_true_values)) ** 2)
            mae = np.mean(np.abs(np.array(aligned_predictions) -
                          np.array(aligned_true_values)))

            # Check for reasonable ranges
            pred_mean = np.mean(aligned_predictions)
            true_mean = np.mean(aligned_true_values)
            pred_std = np.std(aligned_predictions)
            true_std = np.std(aligned_true_values)

            print(f"=== VALIDATION METRICS ===")
            print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
            print(
                f"Prediction stats - Mean: {pred_mean:.6f}, Std: {pred_std:.6f}")
            print(
                f"True value stats - Mean: {true_mean:.6f}, Std: {true_std:.6f}")
            print(f"Mean difference: {abs(pred_mean - true_mean):.6f}")

            # Check if predictions are reasonable
            if pred_std == 0:
                print("WARNING: All predictions are identical!")
            if abs(pred_mean - true_mean) > true_std * 2:
                print("WARNING: Prediction mean significantly different from true mean!")

        return {
            'predicted_values': aligned_predictions,
            'true_values': aligned_true_values,
            'test_dates': aligned_dates,
            'test_bid_prices': aligned_bid_prices,
            'test_ask_prices': aligned_ask_prices,
            'test_with_prompt': aligned_with_prompt,
            'test_without_prompt': aligned_without_prompt
        }
