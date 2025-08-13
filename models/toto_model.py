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
PREDICTION_LENGTH: int = 1

class TotoFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Toto's pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model components
        self.toto = None
        self.model = None
        self.forecaster = None
        
        # Data storage
        self.full_scaled_data = None
        self.train_size = None
        self.test_start_idx = None
        self.original_mid_prices = None
        
        # Test metadata
        self.test_dates = None
        self.test_bid_prices = None
        self.test_ask_prices = None
        self.test_with_prompt = None
        self.test_without_prompt = None
        self.test_true_values = None
        
        print("Initializing Toto Financial Forecasting Model...")
        self.initialize_model()

    def initialize_model(self):
        """Load pre-trained Toto model and compile it"""
        print("Loading Toto model...")

        try:
            self.toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
            self.toto.to(self.device)
            self.model = self.toto.model
            
            # Initialize forecaster with the underlying model
            self.forecaster = TotoForecaster(self.model)
            
            # Compile for better performance (optional)
            try:
                self.toto.compile()
                print("Model compiled successfully")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")

            print("Toto model loaded successfully!")

        except Exception as e:
            print(f"Error loading Toto model: {e}")
            raise

    def split_and_scale_data(self):
        """Prepare data with proper train/test split and no data leakage"""
        # Extract raw data
        try:
            dates, bid_prices, ask_prices, mid_price_series, with_prompt_values, without_prompt_values = (
                self.data_processor.extract_price_time_series()
            )
        except Exception as e:
            print(f"Error extracting time series data: {e}")
            raise

        # Calculate split indices to prevent data leakage
        total_length = len(mid_price_series)
        self.train_size = int(total_length * TRAIN_RATIO)

        print(f"Total data points: {total_length}")
        print(f"Training data points: {self.train_size}")
        print(f"Test data points: {total_length - self.train_size}")

        # Split the data - No overlap between train and test
        train_data = mid_price_series[:self.train_size]
        test_data = mid_price_series[self.train_size:]

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

        # Scale training and test data separately
        train_scaled = self.scaler.transform(train_data.reshape(-1, 1)).flatten()
        test_scaled = self.scaler.transform(test_data.reshape(-1, 1)).flatten()

        # Store scaled data
        self.full_scaled_data = np.concatenate([train_scaled, test_scaled])

        # Store metadata
        self.test_start_idx = self.train_size
        self.original_mid_prices = mid_price_series

        # Test metadata - aligned with test data
        self.test_dates = dates[self.train_size:]
        self.test_bid_prices = bid_prices[self.train_size:]
        self.test_ask_prices = ask_prices[self.train_size:]
        self.test_with_prompt = with_prompt_values[self.train_size:]
        self.test_without_prompt = without_prompt_values[self.train_size:]
        self.test_true_values = test_data  # Original scale

        print(f"Scaler fitted on training data - Min: {train_data.min():.6f}, Max: {train_data.max():.6f}")
        print(f"Test data range - Min: {test_data.min():.6f}, Max: {test_data.max():.6f}")

        return self.full_scaled_data

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Toto model used in zero-shot mode - skipping training")
        return None

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

        try:
            # Generate forecast
            with torch.no_grad():
                forecast = self.forecaster.forecast(
                    inputs,
                    prediction_length=PREDICTION_LENGTH,
                    num_samples=NUM_SAMPLES,
                    samples_per_batch=NUM_SAMPLES,
                )

            # Extract predictions using median for robustness
            predictions = forecast.median.cpu().numpy().flatten()
            return predictions.tolist()

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return [seq[-1] for seq in input_sequences]

    def generate_predictions(self):
        """Generate predictions using sliding window with batching"""
        
        # Prepare data and metadata
        scaled_data = self.split_and_scale_data()
        total_points = len(scaled_data)
        test_start = self.test_start_idx
        
        # Generate predictions sequentially
        predictions_scaled = []
        prediction_indices = []

        # Get all valid prediction indices
        valid_indices = list(range(test_start, total_points))

        for batch_idx in range(0, len(valid_indices), BATCH_SIZE):
            batch_slice = valid_indices[batch_idx : batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1
            total_batches = (len(valid_indices) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"Batch {batch_num}/{total_batches} = predicting for the following indices {batch_slice[0]} - {batch_slice[-1]}")

            # Create batch sequences
            batch_sequences = []
            for i in batch_slice:
                sequence = scaled_data[i - INPUT_CHUNK_LENGTH : i]
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
        predictions_original = self.scaler.inverse_transform(
            np.array(predictions_scaled).reshape(-1, 1)
        ).flatten()
        
        # Align with test metadata
        aligned_results = {
            'predicted_values': predictions_original.tolist(),
            'true_values': [],
            'test_dates': [],
            'test_bid_prices': [],
            'test_ask_prices': [],
            'test_with_prompt': [],
            'test_without_prompt': []
        }
        
        # Populate true values and metadata
        for i, global_idx in enumerate(prediction_indices):
            test_idx = global_idx - test_start
            aligned_results['true_values'].append(self.test_true_values[test_idx])
            aligned_results['test_dates'].append(self.test_dates[test_idx])
            aligned_results['test_bid_prices'].append(self.test_bid_prices[test_idx])
            aligned_results['test_ask_prices'].append(self.test_ask_prices[test_idx])
            aligned_results['test_with_prompt'].append(self.test_with_prompt[test_idx])
            aligned_results['test_without_prompt'].append(self.test_without_prompt[test_idx])
        
        print(f"Completed forecasting. Generated {len(predictions_original)} predictions")
        return aligned_results