from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from chronos import ChronosBoltPipeline
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

class ChronosFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Amazon's Chronos-bolt, a pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.MODEL_NAME = "autogluon/chronos-bolt-base"
        self.PRESET_NAME = "bolt_base"
        print(f"Initializing Chronos-Bolt with context length {self.model_config.INPUT_CHUNK_LENGTH},\n"
              f"batch size {self.model_config.BATCH_SIZE} and prediction length {self.model_config.OUTPUT_CHUNK_LENGTH}")
        self.forecaster = self.initialize_model()


    def initialize_model(self):
        """Load pre-trained predictor"""
        try:
            pipeline = ChronosBoltPipeline.from_pretrained(self.MODEL_NAME)
            return pipeline
        except Exception as e:
            print(f"Error initializing Chronos model: {e}")

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

        # Scale data
        scaled_series = self._scale_series_data(train_series, val_series, test_series)

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
        print("Model used in zero-shot mode - skipping training")
        return None

    def predict_future_values(self, input_sequences):
        """Make prediction for a batch of input sequences"""

        # Convert to tensor (BATCH_SIZE, INPUT_CHUNK_LENGTH)
        batch_array = np.array(input_sequences, dtype=np.float32)
        inputs = torch.FloatTensor(batch_array).to(self.device)

        try:
            with torch.no_grad():
                quantiles, mean = self.forecaster.predict_quantiles(inputs, prediction_length=self.model_config.OUTPUT_CHUNK_LENGTH)
                median_percentile = quantiles[:, :, 4]
                
            # predictions = median_percentile.cpu().numpy().flatten()
            predictions = mean.cpu().numpy().flatten()
            return predictions.tolist()

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return [seq[-1] for seq in input_sequences]


    def generate_predictions(self, test_series):
        """Generate predictions using sliding window with batching"""

        # Calculate how many predictions we can make
        num_predictions = len(test_series) - self.model_config.INPUT_CHUNK_LENGTH
        
        # Generate predictions sequentially
        scaled_predictions = []

        for batch_idx in range(0, num_predictions, self.model_config.BATCH_SIZE):
            # Calculate batch boundaries
            batch_start = batch_idx
            batch_end = min(batch_idx + self.model_config.BATCH_SIZE, num_predictions)
            batch_size_actual = batch_end - batch_start

            # Create batch of input sequences
            batch_sequences = []

            for pred_idx in range(batch_start, batch_end):
                # Get the sequence for this prediction
                start_idx = pred_idx
                end_idx = pred_idx + self.model_config.INPUT_CHUNK_LENGTH
                sequence = test_series[start_idx:end_idx]
                batch_sequences.append(sequence)

            # Convert to numpy array for batch processing
            batch_input = np.array(batch_sequences, dtype=np.float32)
            
            # Get predictions for this batch
            batch_predictions = self.predict_future_values(batch_input)
            
            # Validate prediction count matches expected batch size
            if len(batch_predictions) != batch_size_actual:
                print(f"Warning: Expected {batch_size_actual} predictions, got {len(batch_predictions)}")
            
            scaled_predictions.extend(batch_predictions)

        # Inverse transform and align metadata
        predicted_values = self.scaler.inverse_transform(
            np.array(scaled_predictions, dtype=np.float32).reshape(-1, 1)
        ).flatten()

        return predicted_values.tolist()
    
    def get_true_values(self, test_series):
        """Retrieves true values from the test series after scaling back."""
        true_values = self.scaler.inverse_transform(test_series.reshape(-1, 1)).flatten()
        return true_values[self.model_config.INPUT_CHUNK_LENGTH:].tolist()