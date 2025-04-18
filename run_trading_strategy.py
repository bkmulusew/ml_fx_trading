from utils import ModelConfig
from data_processing import DataProcessor
from models import DartsFinancialForecastingModel
from models import PytorchFinancialForecastingModel
from metrics import ModelEvaluationMetrics
from matplotlib import pyplot as plt
import numpy as np
import argparse
from strategies import TradingStrategy
from collections import defaultdict
from datetime import datetime
import pandas as pd

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
        train_series, valid_series, test_series, test_dates, test_bid_prices, test_ask_prices, test_with_prompt, test_without_prompt = predictor.split_and_scale_data()
        predictor.train(train_series, valid_series)
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.get_true_values(test_series)

    # Plot the actual and predicted values using matplotlib to visualize the model's performance.
    plt.plot(true_values, color = 'blue', label = 'True')
    plt.plot(predicted_values, color = 'red', label = 'Prediction')
    plt.title('True and Predicted Values (Model: ' + model_name + ' )')
    plt.xlabel('Observations')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()
    plt.savefig('true_vs_pridicted_' + model_name + '.png', dpi=300, bbox_inches='tight')

    plt.clf()

    # Calculate and print the prediction error using the actual and predicted values.
    prediction_error = eval_metrics.calculate_prediction_error(predicted_values, true_values)
    print(f"Prediction Error: {prediction_error}")
    print (f"\n")

    print(f"Model: {model_name}")

    # Parse test_dates into datetime objects
    # parsed_dates = [datetime.strptime(date, "%d/%m/%Y %H:%M:%S") for date in test_dates]

    # parsed_dates = [datetime.strptime(date, "%m/%d/%y %H:%M") for date in test_dates]
    parsed_dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in test_dates]

    # Create a dictionary to group values by date
    chunked_values = defaultdict(lambda: {"true_values": [], "predicted_values": [], "bid_price":[], "ask_price":[], "with_prompt": [], "without_prompt": []})

    # Chunk true_values and predicted_values by date
    for date, true_val, pred_val, bid_price, ask_price, with_prompt, without_prompt in zip(parsed_dates, true_values, predicted_values, test_bid_prices, test_ask_prices, test_with_prompt, test_without_prompt):
        date_key = date.date()  # Use only the date part as the key
        chunked_values[date_key]["true_values"].append(true_val)
        chunked_values[date_key]["predicted_values"].append(pred_val)
        chunked_values[date_key]["bid_price"].append(bid_price)
        chunked_values[date_key]["ask_price"].append(ask_price)
        chunked_values[date_key]["with_prompt"].append(with_prompt)
        chunked_values[date_key]["without_prompt"].append(without_prompt)

    # Use Kelly

    print("--------------------------------")
    print("Using Kelly")
    print("--------------------------------")
    mean_reversion_profit = []
    trend_profit = []
    forecasting_profit = []
    ensemble_with_llm_trend_profit = []
    ensemble_with_llm_mean_reversion_profit = []

    mean_reversion_profit_per_trade = []
    trend_profit_per_trade = []
    forecasting_profit_per_trade = []
    ensemble_with_llm_trend_profit_per_trade = []
    ensemble_with_llm_mean_reversion_profit_per_trade = []

    mean_reversion_coeffs = []
    trend_coeffs = []
    forecasting_coeffs = []
    llm_sentiment_coeffs = []

    mean_reversion_num_trades = []
    trend_num_trades = []
    forecasting_num_trades = []
    ensemble_with_llm_trend_num_trades = []
    ensemble_with_llm_mean_reversion_num_trades = []

    # Simulate trading using each of the specified trade thresholds.
    for trade_thresold in trade_thresholds:
        print(f"Trade Threshold: {trade_thresold}")
        # Print the results
        for date_key, values in chunked_values.items():
            trading_strategy = TradingStrategy(model_config.WALLET_A, model_config.WALLET_B, model_config.FRAC_KELLY, trade_thresold)
            trading_strategy.simulate_trading_with_strategies(values['true_values'], values['predicted_values'], values['bid_price'], values['ask_price'], values["with_prompt"])
            mean_reversion_profit.append(trading_strategy.total_profit_or_loss["mean_reversion"])
            trend_profit.append(trading_strategy.total_profit_or_loss["trend"])
            forecasting_profit.append(trading_strategy.total_profit_or_loss["pure_forcasting"])
            ensemble_with_llm_trend_profit.append(trading_strategy.total_profit_or_loss["ensemble_with_llm_trend"])
            ensemble_with_llm_mean_reversion_profit.append(trading_strategy.total_profit_or_loss["ensemble_with_llm_mean_reversion"])
            mean_reversion_profit_per_trade_val = trading_strategy.total_profit_or_loss["mean_reversion"] / trading_strategy.num_trades["mean_reversion"]
            trend_profit_per_trade_val = trading_strategy.total_profit_or_loss["trend"] / trading_strategy.num_trades["trend"]
            forecasting_profit_per_trade_val = trading_strategy.total_profit_or_loss["pure_forcasting"] / trading_strategy.num_trades["pure_forcasting"]
            ensemble_with_llm_trend_profit_per_trade_val = trading_strategy.total_profit_or_loss["ensemble_with_llm_trend"] / trading_strategy.num_trades["ensemble_with_llm_trend"]
            ensemble_with_llm_mean_reversion_profit_per_trade_val = trading_strategy.total_profit_or_loss["ensemble_with_llm_mean_reversion"] / trading_strategy.num_trades["ensemble_with_llm_mean_reversion"]
            mean_reversion_profit_per_trade.append(mean_reversion_profit_per_trade_val)
            trend_profit_per_trade.append(trend_profit_per_trade_val)
            forecasting_profit_per_trade.append(forecasting_profit_per_trade_val)
            ensemble_with_llm_trend_profit_per_trade.append(ensemble_with_llm_trend_profit_per_trade_val)
            ensemble_with_llm_mean_reversion_profit_per_trade.append(ensemble_with_llm_mean_reversion_profit_per_trade_val)
            
            mean_reversion_coeffs.append(trading_strategy.mean_reversion_coeff_llm)
            trend_coeffs.append(trading_strategy.trend_coeff_llm)
            forecasting_coeffs.append(trading_strategy.forecasting_coeff_llm)
            llm_sentiment_coeffs.append(trading_strategy.llm_sentiment_coeff)

            mean_reversion_num_trades.append(trading_strategy.num_trades["mean_reversion"])
            trend_num_trades.append(trading_strategy.num_trades["trend"])
            forecasting_num_trades.append(trading_strategy.num_trades["pure_forcasting"])
            ensemble_with_llm_trend_num_trades.append(trading_strategy.num_trades["ensemble_with_llm_trend"])
            ensemble_with_llm_mean_reversion_num_trades.append(trading_strategy.num_trades["ensemble_with_llm_mean_reversion"])
            
    cumulative_mean_reversion_profit = np.cumsum(mean_reversion_profit)
    cumulative_trend_profit = np.cumsum(trend_profit)
    cumulative_forecasting_profit = np.cumsum(forecasting_profit)
    cumulative_ensemble_with_llm_trend_profit = np.cumsum(ensemble_with_llm_trend_profit)
    cumulative_ensemble_with_llm_mean_reversion_profit = np.cumsum(ensemble_with_llm_mean_reversion_profit)

    cumulative_mean_reversion_profit_per_trade = [
        np.sum(mean_reversion_profit[:i+1]) / np.sum(mean_reversion_num_trades[:i+1]) for i in range(len(mean_reversion_profit))
    ]
    cumulative_trend_profit_per_trade = [
        np.sum(trend_profit[:i+1]) / np.sum(trend_num_trades[:i+1]) for i in range(len(trend_profit))
    ]
    cumulative_forecasting_profit_per_trade = [
        np.sum(forecasting_profit[:i+1]) / np.sum(forecasting_num_trades[:i+1]) for i in range(len(forecasting_profit))
    ]
    cumulative_ensemble_with_llm_trend_profit_per_trade = [
        np.sum(ensemble_with_llm_trend_profit[:i+1]) / np.sum(ensemble_with_llm_trend_num_trades[:i+1]) for i in range(len(ensemble_with_llm_trend_profit))
    ]
    cumulative_ensemble_with_llm_mean_reversion_profit_per_trade = [
        np.sum(ensemble_with_llm_mean_reversion_profit[:i+1]) / np.sum(ensemble_with_llm_mean_reversion_num_trades[:i+1]) for i in range(len(ensemble_with_llm_mean_reversion_profit))
    ]

    plt.plot(cumulative_mean_reversion_profit_per_trade, color='purple', label='Cumulative Mean Reversion Profit Per Trade')
    plt.plot(cumulative_trend_profit_per_trade, color='blue', label='Cumulative Trend Profit Per Trade')
    plt.plot(cumulative_forecasting_profit_per_trade, color='red', label='Cumulative Forecasting Profit Per Trade')
    plt.plot(cumulative_ensemble_with_llm_trend_profit_per_trade, color='green', label='Cumulative Ensemble With LLM Trend Profit Per Trade')
    plt.plot(cumulative_ensemble_with_llm_mean_reversion_profit_per_trade, color='orange', label='Cumulative Ensemble With LLM Mean Reversion Profit Per Trade')
    plt.title('Cumulative Trend and Forecasting Profits Per Trade Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Profit Per Trade')
    plt.legend()
    plt.savefig('cumulative_trend_vs_forecasting_profits_per_trade_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_coeffs, color='purple', label='Mean Reversion Coefficient')
    plt.plot(trend_coeffs, color='blue', label='Trend Coefficient')
    plt.plot(forecasting_coeffs, color='red', label='Forecasting Coefficient')
    plt.plot(llm_sentiment_coeffs, color='green', label='LLM Coefficient')
    plt.title('Trend, Forecasting, and LLM Coefficients Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.savefig('trend_vs_forecasting_coeffs_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(cumulative_mean_reversion_profit, color='purple', label='Cumulative Mean Reversion Profit')
    plt.plot(cumulative_trend_profit, color='blue', label='Cumulative Trend Profit')
    plt.plot(cumulative_forecasting_profit, color='red', label='Cumulative Forecasting Profit')
    plt.plot(cumulative_ensemble_with_llm_trend_profit, color='green', label='Cumulative Ensemble With LLM Trend Profit')
    plt.plot(cumulative_ensemble_with_llm_mean_reversion_profit, color='orange', label='Cumulative Ensemble With LLM Mean Reversion Profit')
    plt.title('Cumulative Trend and Forecasting Profits Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.savefig('cumulative_trend_vs_forecasting_profits_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_profit, color='purple', label='Mean Reversion Profit')
    plt.plot(trend_profit, color = 'blue', label = 'Trend Profit')
    plt.plot(forecasting_profit, color = 'red', label = 'Forcasting Profit')
    plt.plot(ensemble_with_llm_trend_profit, color='green', label='Ensemble With LLM Trend Profit')
    plt.plot(ensemble_with_llm_mean_reversion_profit, color='orange', label='Ensemble With LLM Mean Reversion Profit')
    plt.title('Trend and Forcasting Profits Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Profit')
    plt.legend()
    plt.savefig('trend_vs_forcasting_profits_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_profit_per_trade, color='purple', label='Mean Reversion Profit Per Trade')
    plt.plot(trend_profit_per_trade, color = 'blue', label = 'Trend Profit Per Trade')
    plt.plot(forecasting_profit_per_trade, color = 'red', label = 'Forcasting Profit Per Trade')
    plt.plot(ensemble_with_llm_trend_profit_per_trade, color='green', label='Ensemble With LLM Trend Profit Per Trade')
    plt.plot(ensemble_with_llm_mean_reversion_profit_per_trade, color='orange', label='Ensemble With LLM Mean Reversion Profit Per Trade')
    plt.title('Trend and Forcasting Profit Per Trade Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Profit Per Trade')
    plt.legend()
    plt.savefig('trend_vs_forcasting_profit_per_trade_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    # Use Fixed Position
    
    print("--------------------------------")
    print("Using Fixed Position")
    print("--------------------------------")

    mean_reversion_profit = []
    trend_profit = []
    forecasting_profit = []
    ensemble_with_llm_trend_profit = []
    ensemble_with_llm_mean_reversion_profit = []

    mean_reversion_profit_per_trade = []
    trend_profit_per_trade = []
    forecasting_profit_per_trade = []
    ensemble_with_llm_trend_profit_per_trade = []
    ensemble_with_llm_mean_reversion_profit_per_trade = []

    mean_reversion_coeffs = []
    trend_coeffs = []
    forecasting_coeffs = []
    llm_sentiment_coeffs = []

    mean_reversion_num_trades = []
    trend_num_trades = []
    forecasting_num_trades = []
    ensemble_with_llm_trend_num_trades = []
    ensemble_with_llm_mean_reversion_num_trades = []

    # Simulate trading using each of the specified trade thresholds.
    for trade_thresold in trade_thresholds:
        print(f"Trade Threshold: {trade_thresold}")
        # Print the results
        for date_key, values in chunked_values.items():
            trading_strategy = TradingStrategy(model_config.WALLET_A, model_config.WALLET_B, model_config.FRAC_KELLY, trade_thresold)
            trading_strategy.simulate_trading_with_strategies(values['true_values'], values['predicted_values'], values['bid_price'], values['ask_price'], values["with_prompt"], use_kelly=False)
            mean_reversion_profit.append(trading_strategy.total_profit_or_loss["mean_reversion"])
            trend_profit.append(trading_strategy.total_profit_or_loss["trend"])
            forecasting_profit.append(trading_strategy.total_profit_or_loss["pure_forcasting"])
            ensemble_with_llm_trend_profit.append(trading_strategy.total_profit_or_loss["ensemble_with_llm_trend"])
            ensemble_with_llm_mean_reversion_profit.append(trading_strategy.total_profit_or_loss["ensemble_with_llm_mean_reversion"])
            mean_reversion_profit_per_trade_val = trading_strategy.total_profit_or_loss["mean_reversion"] / trading_strategy.num_trades["mean_reversion"]
            trend_profit_per_trade_val = trading_strategy.total_profit_or_loss["trend"] / trading_strategy.num_trades["trend"]
            forecasting_profit_per_trade_val = trading_strategy.total_profit_or_loss["pure_forcasting"] / trading_strategy.num_trades["pure_forcasting"]
            ensemble_with_llm_trend_profit_per_trade_val = trading_strategy.total_profit_or_loss["ensemble_with_llm_trend"] / trading_strategy.num_trades["ensemble_with_llm_trend"]
            ensemble_with_llm_mean_reversion_profit_per_trade_val = trading_strategy.total_profit_or_loss["ensemble_with_llm_mean_reversion"] / trading_strategy.num_trades["ensemble_with_llm_mean_reversion"]
            mean_reversion_profit_per_trade.append(mean_reversion_profit_per_trade_val)
            trend_profit_per_trade.append(trend_profit_per_trade_val)
            forecasting_profit_per_trade.append(forecasting_profit_per_trade_val)
            ensemble_with_llm_trend_profit_per_trade.append(ensemble_with_llm_trend_profit_per_trade_val)
            ensemble_with_llm_mean_reversion_profit_per_trade.append(ensemble_with_llm_mean_reversion_profit_per_trade_val)

            mean_reversion_coeffs.append(trading_strategy.mean_reversion_coeff_llm)
            trend_coeffs.append(trading_strategy.trend_coeff_llm)
            forecasting_coeffs.append(trading_strategy.forecasting_coeff_llm)
            llm_sentiment_coeffs.append(trading_strategy.llm_sentiment_coeff)

            mean_reversion_num_trades.append(trading_strategy.num_trades["mean_reversion"])
            trend_num_trades.append(trading_strategy.num_trades["trend"])
            forecasting_num_trades.append(trading_strategy.num_trades["pure_forcasting"])
            ensemble_with_llm_trend_num_trades.append(trading_strategy.num_trades["ensemble_with_llm_trend"])
            ensemble_with_llm_mean_reversion_num_trades.append(trading_strategy.num_trades["ensemble_with_llm_mean_reversion"])
            
    cumulative_mean_reversion_profit = np.cumsum(mean_reversion_profit)
    cumulative_trend_profit = np.cumsum(trend_profit)
    cumulative_forecasting_profit = np.cumsum(forecasting_profit)
    cumulative_ensemble_with_llm_trend_profit = np.cumsum(ensemble_with_llm_trend_profit)
    cumulative_ensemble_with_llm_mean_reversion_profit = np.cumsum(ensemble_with_llm_mean_reversion_profit)

    cumulative_mean_reversion_profit_per_trade = [
        np.sum(mean_reversion_profit[:i+1]) / np.sum(mean_reversion_num_trades[:i+1]) for i in range(len(mean_reversion_profit))
    ]
    cumulative_trend_profit_per_trade = [
        np.sum(trend_profit[:i+1]) / np.sum(trend_num_trades[:i+1]) for i in range(len(trend_profit))
    ]
    cumulative_forecasting_profit_per_trade = [
        np.sum(forecasting_profit[:i+1]) / np.sum(forecasting_num_trades[:i+1]) for i in range(len(forecasting_profit))
    ]
    cumulative_ensemble_with_llm_trend_profit_per_trade = [
        np.sum(ensemble_with_llm_trend_profit[:i+1]) / np.sum(ensemble_with_llm_trend_num_trades[:i+1]) for i in range(len(ensemble_with_llm_trend_profit))
    ]
    cumulative_ensemble_with_llm_mean_reversion_profit_per_trade = [
        np.sum(ensemble_with_llm_mean_reversion_profit[:i+1]) / np.sum(ensemble_with_llm_mean_reversion_num_trades[:i+1]) for i in range(len(ensemble_with_llm_mean_reversion_profit))
    ]

    plt.plot(cumulative_mean_reversion_profit_per_trade, color='purple', label='Cumulative Mean Reversion Profit Per Trade')
    plt.plot(cumulative_trend_profit_per_trade, color='blue', label='Cumulative Trend Profit Per Trade')
    plt.plot(cumulative_forecasting_profit_per_trade, color='red', label='Cumulative Forecasting Profit Per Trade')
    plt.plot(cumulative_ensemble_with_llm_trend_profit_per_trade, color='green', label='Cumulative Ensemble With LLM Trend Profit Per Trade')
    plt.plot(cumulative_ensemble_with_llm_mean_reversion_profit_per_trade, color='orange', label='Cumulative Ensemble With LLM Mean Reversion Profit Per Trade')
    plt.title('Cumulative Trend and Forecasting Profits Per Trade Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Profit Per Trade')
    plt.legend()
    plt.savefig('cumulative_trend_vs_forecasting_profits_per_trade_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_coeffs, color='purple', label='Mean Reversion Coefficient')
    plt.plot(trend_coeffs, color='blue', label='Trend Coefficient')
    plt.plot(forecasting_coeffs, color='red', label='Forecasting Coefficient')
    plt.plot(llm_sentiment_coeffs, color='green', label='LLM Coefficient')
    plt.title('Trend, Forecasting, and LLM Coefficients Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.savefig('trend_vs_forecasting_coeffs_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(cumulative_mean_reversion_profit, color='purple', label='Cumulative Mean Reversion Profit')
    plt.plot(cumulative_trend_profit, color='blue', label='Cumulative Trend Profit')
    plt.plot(cumulative_forecasting_profit, color='red', label='Cumulative Forecasting Profit')
    plt.plot(cumulative_ensemble_with_llm_trend_profit, color='green', label='Cumulative Ensemble With LLM Trend Profit')
    plt.plot(cumulative_ensemble_with_llm_mean_reversion_profit, color='orange', label='Cumulative Ensemble With LLM Mean Reversion Profit')
    plt.title('Cumulative Trend and Forecasting Profits Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.savefig('cumulative_trend_vs_forecasting_profits_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_profit, color='purple', label='Mean Reversion Profit')
    plt.plot(trend_profit, color = 'blue', label = 'Trend Profit')
    plt.plot(forecasting_profit, color = 'red', label = 'Forcasting Profit')
    plt.plot(ensemble_with_llm_trend_profit, color='green', label='Ensemble With LLM Trend Profit')
    plt.plot(ensemble_with_llm_mean_reversion_profit, color='orange', label='Ensemble With LLM Mean Reversion Profit')
    plt.title('Trend and Forcasting Profits Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Profit')
    plt.legend()
    plt.savefig('trend_vs_forcasting_profits_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_profit_per_trade, color='purple', label='Mean Reversion Profit Per Trade')
    plt.plot(trend_profit_per_trade, color = 'blue', label = 'Trend Profit Per Trade')
    plt.plot(forecasting_profit_per_trade, color = 'red', label = 'Forcasting Profit Per Trade')
    plt.plot(ensemble_with_llm_trend_profit_per_trade, color='green', label='Ensemble With LLM Trend Profit Per Trade')
    plt.plot(ensemble_with_llm_mean_reversion_profit_per_trade, color='orange', label='Ensemble With LLM Mean Reversion Profit Per Trade')
    plt.title('Trend and Forcasting Profit Per Trade Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Profit Per Trade')
    plt.legend()
    plt.savefig('trend_vs_forcasting_profit_per_trade_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

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