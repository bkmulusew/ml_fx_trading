from models import FinancialForecastingModel
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel, NBEATSModel, TCNModel, TransformerModel

class DartsFinancialForecastingModel(FinancialForecastingModel):
    """A financial forecasting model based on the Darts library."""
    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.scaler = None
        self.model_config = model_config
        self.model = self.initialize_model(model_config.MODEL_NAME)

    def initialize_model(self, model_name):
        """Creates the model."""
        if model_name == "nbeats":
            return NBEATSModel(
                input_chunk_length=self.model_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                num_layers=5,
                num_blocks=1,
                num_stacks=2,
                layer_widths=512,
                dropout=0.2,
                n_epochs=self.model_config.N_EPOCHS,
                batch_size=self.model_config.BATCH_SIZE,
                model_name="nbeats",
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [0]
                },
            )

        elif model_name == "nhits":
            return NHiTSModel(
                input_chunk_length=self.model_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                num_layers=5,
                num_blocks=1,
                num_stacks=2,
                layer_widths=512,
                dropout=0.2,
                n_epochs=self.model_config.N_EPOCHS,
                batch_size=self.model_config.BATCH_SIZE,
                model_name="nhits",
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [0]
                },
            )

        elif model_name == "transformer":
            return TransformerModel(
                input_chunk_length=self.model_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                n_epochs=self.model_config.N_EPOCHS,
                batch_size=self.model_config.BATCH_SIZE,
                nhead=4,
                d_model=256,
                num_encoder_layers = 3,
                num_decoder_layers = 3,
                dim_feedforward = 16,
                norm_type = "LayerNormNoBias",
                dropout=0.2,
                model_name="transformer",
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [0]
                },
            )

        elif model_name == "tcn":
            return TCNModel(
                input_chunk_length=self.model_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.model_config.OUTPUT_CHUNK_LENGTH,
                kernel_size = 3,
                num_filters = 64,
                num_layers = 8,
                dilation_base = 2,
                n_epochs=self.model_config.N_EPOCHS,
                batch_size=self.model_config.BATCH_SIZE,
                weight_norm = True,
                model_name="tcn",
                dropout = 0.2,
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs = {
                    "accelerator": "gpu",
                    "devices": [0]
                }
            )

        else:
            raise ValueError("Invalid model name.")


    def split_and_scale_data(self, train_ratio=0.5, validation_ratio=0.1):
        """Splits the data into training, validation, and test sets and applies scaling."""
        dates, bid_prices, ask_prices, mid_price, with_prompt_values, without_prompt_values, label_values = self.data_processor.extract_price_time_series()

        # Calculate indices for splitting
        num_observations = len(mid_price)
        train_end_index = int(num_observations * train_ratio)
        validation_end_index = int(num_observations * (train_ratio + validation_ratio))

        # Split mid price series into train/validation/test
        train_series, val_series, test_series = self._split_mid_price_series(mid_price, train_end_index, validation_end_index)

        # Scale the series data
        scaled_series = self._scale_series_data(train_series, val_series, test_series)
        
        # Process test data
        test_data = self._process_test_data(
            dates=dates[validation_end_index:],
            bid_prices=bid_prices[validation_end_index:],
            ask_prices=ask_prices[validation_end_index:],
            without_prompt=without_prompt_values[validation_end_index:],
            with_prompt=with_prompt_values[validation_end_index:],
            label=label_values[validation_end_index:]
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
        self.scaler = Scaler()
        return (
            self.scaler.fit_transform(train_series),
            self.scaler.transform(val_series),
            self.scaler.transform(test_series)
        )

    def train(self, train_series, validation_series):
        """Trains the model."""
        self.model.fit(train_series, val_series=validation_series, verbose=True)

    def predict_future_values(self, test_series):
        """Makes future value predictions based on the test series."""
        return self.model.predict(self.model_config.OUTPUT_CHUNK_LENGTH, series=test_series)

    def generate_predictions(self, test_series):
        """Generates predictions for each window of the test series."""
        transformed_series = []

        for i in range(len(test_series) - self.model_config.INPUT_CHUNK_LENGTH):
            transformed_series.append(test_series[i: i + self.model_config.INPUT_CHUNK_LENGTH])

        pred_series = self.predict_future_values(transformed_series)

        predicted_values = []
        for pred in pred_series:
            predicted_values.append(pred.values()[0][0])

        predicted_df = pd.DataFrame(predicted_values, columns=['predicted'])
        tseries_predicted = TimeSeries.from_dataframe(predicted_df, value_cols='predicted')
        tseries_predicted = self.scaler.inverse_transform(tseries_predicted).pd_series(copy=True)
        predicted_values = tseries_predicted.values.tolist()

        return predicted_values

    def get_true_values(self, test_series):
        """Retrieves true values from the test series after scaling back."""
        test_series_inverse = self.scaler.inverse_transform(test_series)
        true_df = test_series_inverse.pd_series(copy=True)
        true_values = true_df[self.model_config.INPUT_CHUNK_LENGTH:].values.tolist()
        return true_values