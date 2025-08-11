from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from chronos import ChronosBoltPipeline

class ChronosFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Amazon's Chronos, a pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_chunk_length = model_config.INPUT_CHUNK_LENGTH
        self.batch_size = model_config.BATCH_SIZE
        self.train_ratio = model_config.TRAIN_RATIO
        self.prediction_length = model_config.OUTPUT_CHUNK_LENGTH

        self.forecaster = None
        
        self.full_scaled_data = None
        self.train_size = None
        self.test_start_idx = None
        self.original_mid_prices = None
        
        self.test_dates = None
        self.test_bid_prices = None
        self.test_ask_prices = None
        self.test_with_prompt = None
        self.test_without_prompt = None
        self.test_true_values = None

        print(f"Initializing Chronos Model with context length {self.input_chunk_length},\n"
              f"               batch size {self.batch_size} and prediction length {self.prediction_length}")
        self.MODEL_NAME = "autogluon/chronos-bolt-small"
        self.initialize_model()

    def initialize_model(self):
        """Load pre-trained predictor"""
        try:
            pipeline = ChronosBoltPipeline.from_pretrained(self.MODEL_NAME)
            self.forecaster = pipeline
        except Exception as e:
            print(f"Error initializing Chronos model: {e}")

    def split_and_scale_data(self):
        # Extract raw data
        try:
            dates, bid_prices, ask_prices, mid_price_series, with_prompt_values, without_prompt_values = (
                self.data_processor.extract_price_time_series()
            )
        except Exception as e:
            print(f"Error extracting time series data: {e}")
            raise

        total_length = len(mid_price_series)
        self.train_size = int(total_length * self.train_ratio)

        print(f"Total data points: {total_length}")
        print(f"Training data points: {self.train_size}")
        print(f"Test data points: {total_length - self.train_size}")

        train_data = mid_price_series[:self.train_size]
        test_data = mid_price_series[self.train_size:]

        self.scaler.fit(train_data.reshape(-1, 1))

        params = {
            "min_": self.scaler.min_.tolist(),
            "scale_": self.scaler.scale_.tolist(),
            "data_min_": self.scaler.data_min_.tolist(),
            "data_max_": self.scaler.data_max_.tolist(),
            "data_range_": self.scaler.data_range_.tolist(),
            "feature_range": self.scaler.feature_range
        }

        train_scaled = self.scaler.transform(train_data.reshape(-1, 1)).flatten()
        test_scaled = self.scaler.transform(test_data.reshape(-1, 1)).flatten()

        self.full_scaled_data = np.concatenate([train_scaled, test_scaled])

        self.test_start_idx = self.train_size
        self.original_mid_prices = mid_price_series

        self.test_dates = dates[self.train_size + self.input_chunk_length:]
        self.test_bid_prices = bid_prices[self.train_size + self.input_chunk_length:]
        self.test_ask_prices = ask_prices[self.train_size + self.input_chunk_length:]
        self.test_with_prompt = with_prompt_values[self.train_size + self.input_chunk_length:]
        self.test_without_prompt = without_prompt_values[self.train_size + self.input_chunk_length:]
        self.test_true_values = test_data[self.input_chunk_length:]

        print(f"Scaler fitted on training data - Min: {train_data.min():.6f}, Max: {train_data.max():.6f}")
        print(f"Test data range - Min: {test_data.min():.6f}, Max: {test_data.max():.6f}")

        return self.full_scaled_data
    

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
                quantiles, mean = self.forecaster.predict_quantiles(inputs, prediction_length=self.prediction_length)
                median_percentile = quantiles[:, :, 4]
                
            # predictions = median_percentile.cpu().numpy().flatten()
            predictions = mean.cpu().numpy().flatten()
            return predictions.tolist()

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return [seq[-1] for seq in input_sequences]


    def generate_predictions(self):
        scaled_data = self.split_and_scale_data()
        total_points = len(scaled_data)
        test_start = self.test_start_idx
        
        # Generate predictions sequentially
        predictions_scaled = []
        prediction_indices = []

        # Get all valid prediction indices - offset by INPUT_CHUNK_LENGTH to prevent data leakage
        valid_indices = list(range(test_start + self.input_chunk_length, total_points))

        for batch_idx in range(0, len(valid_indices), self.batch_size):
            batch_slice = valid_indices[batch_idx : batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            total_batches = (len(valid_indices) + self.batch_size - 1) // self.batch_size
            
            print(f"Batch {batch_num}/{total_batches} = predicting for the following indices {batch_slice[0]} - {batch_slice[-1]}")

            # Create batch sequences
            batch_sequences = []
            for i in batch_slice:
                sequence = scaled_data[i - self.input_chunk_length : i]
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
            test_idx = global_idx - (test_start + self.input_chunk_length)
            aligned_results['true_values'].append(self.test_true_values[test_idx])
            aligned_results['test_dates'].append(self.test_dates[test_idx])
            aligned_results['test_bid_prices'].append(self.test_bid_prices[test_idx])
            aligned_results['test_ask_prices'].append(self.test_ask_prices[test_idx])
            aligned_results['test_with_prompt'].append(self.test_with_prompt[test_idx])
            aligned_results['test_without_prompt'].append(self.test_without_prompt[test_idx])
        
        print(f"Completed forecasting. Generated {len(predictions_original)} predictions")
        return aligned_results