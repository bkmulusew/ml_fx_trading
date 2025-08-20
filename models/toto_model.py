from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from models.toto.toto.data.util.dataset import MaskedTimeseries
from models.toto.toto.inference.forecaster import TotoForecaster
from models.toto.toto.model.toto import Toto

NUM_SAMPLES: int = 32
TIME_INTERVAL_SECONDS: float = 60.0

class TotoFinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Toto's pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.feature_order = ['mid', 'bid', 'ask', 'spread']
        self.scalers = {name: MinMaxScaler(feature_range=(0, 1)) for name in self.feature_order}
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
        dates, bid_prices, ask_prices, spread, mid_prices, with_prompt_values, without_prompt_values = self.data_processor.extract_price_time_series()

        # Calculate indices for splitting
        n = len(mid_prices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))

        # Split each series (keep unscaled copies for ground-truth later)
        self.train_mid_prices, self.val_mid_prices, self.test_mid_prices = mid_prices[:train_end], mid_prices[train_end:val_end], mid_prices[val_end:]
        self.train_bid_prices, self.val_bid_prices, self.test_bid_prices = bid_prices[:train_end], bid_prices[train_end:val_end], bid_prices[val_end:]
        self.train_ask_prices, self.val_ask_prices, self.test_ask_prices = ask_prices[:train_end], ask_prices[train_end:val_end], ask_prices[val_end:]
        self.train_spread, self.val_spread, self.test_spread = spread[:train_end], spread[train_end:val_end], spread[val_end:]

        # Build matrices (T, C) in a fixed feature order
        X_train = np.column_stack([
            self.train_mid_prices, self.train_bid_prices, self.train_ask_prices, self.train_spread
        ])
        X_val = np.column_stack([
            self.val_mid_prices, self.val_bid_prices, self.val_ask_prices, self.val_spread
        ])
        X_test = np.column_stack([
            self.test_mid_prices, self.test_bid_prices, self.test_ask_prices, self.test_spread
        ])

        # Scale each column with its own scaler fit on TRAIN ONLY
        X_train_scaled = X_train.copy().astype(np.float32) # (T_train, 4)
        X_val_scaled   = X_val.copy().astype(np.float32)
        X_test_scaled  = X_test.copy().astype(np.float32)

        for j, name in enumerate(self.feature_order):
            s = self.scalers[name]
            X_train_scaled[:, j:j+1] = s.fit_transform(X_train[:, j:j+1])
            X_val_scaled[:, j:j+1] = s.transform(X_val[:,j:j+1])
            X_test_scaled[:, j:j+1] = s.transform(X_test[:, j:j+1])

        # Process test data
        meta = self._align_test_targets(
            dates=dates[val_end:],
            without_prompt=without_prompt_values[val_end:],
            with_prompt=with_prompt_values[val_end:]
        )

        return (X_test_scaled, *meta)
    
    def _align_test_targets(self, **test_series):
        """Process all test data series by applying the input chunk length offset."""
        return [
            series[self.model_config.INPUT_CHUNK_LENGTH:]
            for series in test_series.values()
        ]

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Toto model used in zero-shot mode - skipping training")

    def predict_future_values(self, input_sequences):
        """Make prediction for a batch of input sequences"""

        # input_sequences shape: (BATCH_SIZE, INPUT_CHUNK_LENGTH, 4)
        batch_size, _, num_features = input_sequences.shape

        # Toto expects (batch, variates, time_steps)
        # Transpose from (batch, time_steps, features) to (batch, features, time_steps)
        batch_tensor = torch.FloatTensor(input_sequences).transpose(1, 2).to(self.device)

        # Create timestamp and time interval tensors
        timestamp_seconds = torch.zeros_like(batch_tensor).to(self.device)
        time_interval_seconds = torch.full(
            (batch_size, num_features), 
            TIME_INTERVAL_SECONDS, 
            dtype=torch.float32
        ).to(self.device)

        # Create MaskedTimeseries with multivariate input
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
                prediction_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                num_samples=NUM_SAMPLES,
                samples_per_batch=NUM_SAMPLES,
            )

        # Extract predictions using median for robustness
        # forecast.median shape: (batch, variates, prediction_length)
        predictions = forecast.median.cpu().numpy()
        
        # Collect predictions in feature_order
        out = [predictions[:, self.feature_order.index(name), 0] for name in self.feature_order]
    
        return tuple(out) # (mid_preds, bid_preds, ask_preds, spread_preds)

    def generate_predictions(self, data):
        """Generate predictions using sliding window with batching"""

        # data shape: (time_steps, 4)
        num_timesteps, _ = data.shape

        # Calculate how many predictions we can make
        num_predictions = num_timesteps - self.model_config.INPUT_CHUNK_LENGTH

        if num_predictions <= 0:
            raise ValueError(f"Not enough data points. Need at least {self.model_config.INPUT_CHUNK_LENGTH + 1} timesteps, got {num_timesteps}")
        
        # Allocate storage for each feature
        all_preds = {name: np.empty(num_predictions, dtype=np.float32) for name in self.feature_order}

        for batch_idx in range(0, num_predictions, self.model_config.BATCH_SIZE):
            # Calculate batch boundaries
            batch_start = batch_idx
            batch_end = min(batch_idx + self.model_config.BATCH_SIZE, num_predictions)

            # Create batch of input sequences
            indices = np.arange(batch_start, batch_end)[:, None] + np.arange(self.model_config.INPUT_CHUNK_LENGTH)
            batch_input = data[indices]  # Shape: (actual_batch_size, INPUT_CHUNK_LENGTH, num_features)
            
            preds = self.predict_future_values(batch_input)
            
            for name, arr in zip(self.feature_order, preds):
                all_preds[name][batch_start:batch_end] = arr

        # Inverse transform
        predicted_values = {}
        for name in self.feature_order:
            arr = all_preds[name].reshape(-1, 1)
            predicted_values[name] = self.scalers[name].inverse_transform(arr).ravel().tolist()

        return tuple(predicted_values[name] for name in self.feature_order)