from utils import ModelConfig
from data_processing import DataProcessor
from models import DartsFinancialForecastingModel
from models import PytorchFinancialForecastingModel
from metrics import ModelEvaluationMetrics
from matplotlib import pyplot as plt
import numpy as np
import argparse
from strategies import TradingStrategy

def run_sl_based_trading_strategy(model_name, model_config, trade_thresholds):
    """Run a trading strategy based on a supervised learning model, including model training, prediction,
    and evaluation of trading performance."""
    eval_metrics = ModelEvaluationMetrics()

    # Initialize a DataProcessor instance to preprocess and manage the dataset.
    dataProcessor = DataProcessor(model_config)

    # Instantiate a financial forecasting model, train, evaluate, and predict.
    if model_name == 'bilstm':
        predictor = PytorchFinancialForecastingModel(model_name, dataProcessor, model_config)
        processed_data = predictor.split_and_scale_data()
        predictor.train(processed_data['x_train'], processed_data['y_train'], processed_data['x_valid'], processed_data['y_valid'])
        generated_values = predictor.generate_predictions(processed_data['x_test'], processed_data['y_test'])
        predicted_values = generated_values['predicted_values']
        true_values = generated_values['true_values']
    else:
        predictor = DartsFinancialForecastingModel(model_name, dataProcessor, model_config)
        train_series, valid_series, test_series = predictor.split_and_scale_data()
        predictor.train(train_series, valid_series)
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.get_true_values(test_series)

    # Plot the actual and predicted values using matplotlib to visualize the model's performance.
    plt.plot(true_values, color = 'blue', label = 'True')
    plt.plot(predicted_values, color = 'red', label = 'Prediction')
    plt.title('True and Predicted Values')
    plt.xlabel('Observations')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()

    # Calculate and print the prediction error using the actual and predicted values.
    prediction_error = eval_metrics.calculate_prediction_error(predicted_values, true_values)
    print(f"Prediction Error: {prediction_error}")
    print (f"\n")

    print(f"Model: {model_name}")

    # Simulate trading using each of the specified trade thresholds.
    for trade_thresold in trade_thresholds:
        print(f"Trade Threshold: {trade_thresold}")
        trading_strategy = TradingStrategy(model_config.WALLET_A, model_config.WALLET_B, model_config.FRAC_KELLY, trade_thresold)
        trading_strategy.simulate_trading_with_strategies(true_values, predicted_values)

def run(args):
    """Parse command-line arguments and configure the trading model, then run the trading strategy."""
    model_config = ModelConfig()
    model_config.INPUT_CHUNK_LENGTH = args.input_chunk_length
    model_config.OUTPUT_CHUNK_LENGTH = args.output_chunk_length
    model_config.N_EPOCHS = args.n_epochs
    model_config.BATCH_SIZE = args.batch_size
    model_config.TRAIN_RATIO = args.train_ratio
    model_config.DATA_FILE_PATH = args.data_path
    model_config.WALLET_A = args.wallet_a
    model_config.WALLET_B = args.wallet_b
    model_config.FRAC_KELLY = args.frac_kelly

    model_name = args.model
    trade_thresholds = [float(threshold) for threshold in args.trade_thresholds.split(',')]

    # Run the trading strategy based on the specified model and configuration.
    run_sl_based_trading_strategy(model_name, model_config, trade_thresholds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet_a", type=float, default=54870.0, help="Amount of money in wallet A (currency A).")
    parser.add_argument("--wallet_b", type=float, default=10000.0, help="Amount of money in wallet B (currency B).")
    parser.add_argument(
        "--model",
        type=str,
        default="tcn",
        help="Specify the supervised learning model to use. Supported models include 'bilstm' for Bidirectional LSTM with attention, \
            'nbeats' for NBEATS, 'nhits' for NHiTS, 'transformer' for Transformer, and 'tcn' for Temporal Convolutional Network. \
            Default is 'tcn'."
    )
    parser.add_argument("--input_chunk_length", type=int, default=50, help="Length of the input sequences.")
    parser.add_argument("--output_chunk_length", type=int, default=1, help="Length of the output sequences.")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Ratio of training data used in the train/test split.")
    parser.add_argument("--data_path", type=str, default="", help="Path to the training data. Currency rates should be provided as 1 A / 1 B, where A and B are the respective currencies.", required=True)
    parser.add_argument("--frac_kelly", type=bool, default=True, help="Enable fractional kelly to size bets.")
    parser.add_argument(
        "--trade_thresholds", 
        type=str, 
        default="0", 
        # default="0,0.000025,0.00005,0.0001", 
        help="List of threshold values to determine the trade direction and wether to trade based on changes in currency ratio. \
            Provide the values as a comma-separated list of 4 values. For example, '--trade_thresholds 0,0.00025,0.0005,0.001' sets thresholds for trade actions."
    )

    args = parser.parse_args()
    run(args)