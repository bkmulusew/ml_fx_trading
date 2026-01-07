from models import FinancialForecastingModel
import pandas as pd
from darts import TimeSeries
from darts.models import NHiTSModel, NBEATSModel, TCNModel
from typing import Dict

class DartsFinancialForecastingModel(FinancialForecastingModel):
    """A financial forecasting model based on the Darts library."""
    def __init__(self, fx_trading_config, scaler):
        self.scaler = scaler
        self.fx_trading_config = fx_trading_config
        self.model = self.initialize_model(fx_trading_config.MODEL_NAME)

    def initialize_model(self, model_name):
        """Creates the model."""
        print(f"\nLoading {model_name} model...")
        if model_name == "nbeats":
            model = NBEATSModel(
                input_chunk_length=self.fx_trading_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                num_layers=5,
                num_blocks=1,
                num_stacks=2,
                layer_widths=512,
                dropout=0.2,
                n_epochs=self.fx_trading_config.N_EPOCHS,
                batch_size=self.fx_trading_config.TRAIN_BATCH_SIZE,
                model_name="nbeats",
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                    "accelerator": "gpu",
                    "devices": [0]
                },
            )
        elif model_name == "nhits":
            model = NHiTSModel(
                input_chunk_length=self.fx_trading_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                num_layers=5,
                num_blocks=1,
                num_stacks=2,
                layer_widths=512,
                dropout=0.2,
                n_epochs=self.fx_trading_config.N_EPOCHS,
                batch_size=self.fx_trading_config.TRAIN_BATCH_SIZE,
                model_name="nhits",
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                    "accelerator": "gpu",
                    "devices": [0]
                },
            )
        elif model_name == "tcn":
            model = TCNModel(
                input_chunk_length=self.fx_trading_config.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                kernel_size = 3,
                num_filters = 64,
                num_layers = 8,
                dilation_base = 2,
                n_epochs=self.fx_trading_config.N_EPOCHS,
                batch_size=self.fx_trading_config.TRAIN_BATCH_SIZE,
                weight_norm = True,
                model_name="tcn",
                dropout = 0.2,
                optimizer_kwargs={"lr": 0.0001},
                pl_trainer_kwargs={
                    "accelerator": "gpu",
                    "devices": [0]
                },
            )
        elif model_name == "ensemble":
            model = {
                "nbeats": NBEATSModel(
                    input_chunk_length=self.fx_trading_config.INPUT_CHUNK_LENGTH,
                    output_chunk_length=self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                    num_layers=5,
                    num_blocks=1,
                    num_stacks=2,
                    layer_widths=512,
                    dropout=0.2,
                    n_epochs=self.fx_trading_config.N_EPOCHS,
                    batch_size=self.fx_trading_config.TRAIN_BATCH_SIZE,
                    model_name="nbeats",
                    optimizer_kwargs={"lr": 0.0001},
                    pl_trainer_kwargs={
                        "accelerator": "gpu",
                        "devices": [0]
                    },
                ),
                "nhits": NHiTSModel(
                    input_chunk_length=self.fx_trading_config.INPUT_CHUNK_LENGTH,
                    output_chunk_length=self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                    num_layers=5,
                    num_blocks=1,
                    num_stacks=2,
                    layer_widths=512,
                    dropout=0.2,
                    n_epochs=self.fx_trading_config.N_EPOCHS,
                    batch_size=self.fx_trading_config.TRAIN_BATCH_SIZE,
                    model_name="nhits",
                    optimizer_kwargs={"lr": 0.0001},
                    pl_trainer_kwargs={
                        "accelerator": "gpu",
                        "devices": [0]
                    },
                ),
                "tcn": TCNModel(
                    input_chunk_length=self.fx_trading_config.INPUT_CHUNK_LENGTH,
                    output_chunk_length=self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                    kernel_size = 3,
                    num_filters = 64,
                    num_layers = 8,
                    dilation_base = 2,
                    n_epochs=self.fx_trading_config.N_EPOCHS,
                    batch_size=self.fx_trading_config.TRAIN_BATCH_SIZE,
                    weight_norm = True,
                    model_name="tcn",
                    dropout = 0.2,
                    optimizer_kwargs={"lr": 0.0001},
                    pl_trainer_kwargs={
                        "accelerator": "gpu",
                        "devices": [0]
                    },
                )
            }
        else:
            raise ValueError("Invalid model name.")

        print(f"{model_name} model loaded successfully!")
        return model

    def train(self, train_series, validation_series=None):
        """Trains the model."""
        if isinstance(self.model, dict):
            # Enable training of all the models in ensemble
            for name, model in self.model.items():
                print(f"Training {name} model...")
                if validation_series is not None:
                    model.fit(train_series, val_series=validation_series, verbose=False)
                else:
                    model.fit(train_series, verbose=False)
        else:
            if validation_series is not None:
                self.model.fit(train_series, val_series=validation_series, verbose=False)
            else:
                self.model.fit(train_series, verbose=False)

    def predict_future_values(self, test_series):
        """Makes future value predictions based on the test series."""
        if isinstance(self.model, dict):
            # Enable prediction of all the models in ensemble
            preds = {}
            for name, model in self.model.items():
                preds[name] = model.predict(
                    self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                    series=test_series,
                )
            return preds
        else:
            return self.model.predict(
                self.fx_trading_config.OUTPUT_CHUNK_LENGTH,
                series=test_series,
            )

    def generate_predictions(self, test_series):
        """Generates predictions for each window of the test series."""
        transformed_series = []

        for i in range(len(test_series) - self.fx_trading_config.INPUT_CHUNK_LENGTH):
            transformed_series.append(test_series[i: i + self.fx_trading_config.INPUT_CHUNK_LENGTH])

        pred_series = self.predict_future_values(transformed_series)

        # Ensemble branch
        if isinstance(self.model, dict):
            all_predictions: Dict[str, list] = {}

            for name, series_list in pred_series.items():
                predicted_values = []
                for pred in series_list:
                    predicted_values.append(pred.values()[0][0])

                predicted_df = pd.DataFrame(predicted_values, columns=["predicted"])
                tseries_predicted = TimeSeries.from_dataframe(
                    predicted_df, value_cols="predicted"
                )
                tseries_predicted = self.scaler.inverse_transform(
                    tseries_predicted
                ).pd_series(copy=True)
                all_predictions[name] = tseries_predicted.values.tolist()

            return all_predictions

        predicted_values = []
        for pred in pred_series:
            predicted_values.append(pred.values()[0][0])

        predicted_df = pd.DataFrame(predicted_values, columns=['predicted'])
        tseries_predicted = TimeSeries.from_dataframe(predicted_df, value_cols='predicted')
        tseries_predicted = self.scaler.inverse_transform(tseries_predicted).pd_series(copy=True)
        predicted_values = tseries_predicted.values.tolist()

        return predicted_values