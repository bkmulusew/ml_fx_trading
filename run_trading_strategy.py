# importing to set up reproducibility
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from utils import ModelConfig
from data_processing import DataProcessor
from models import DartsFinancialForecastingModel, ChronosFinancialForecastingModel, TotoFinancialForecastingModel
from metrics import ModelEvaluationMetrics
import numpy as np
import argparse
import os
from strategies import TradingStrategy
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import random
from collections import defaultdict
from typing import List, Dict, Union

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_prediction_comparison(true_values, predicted_values, model_config):
    """Plot true vs predicted values and save the figure."""
    plt.plot(true_values, color='blue', label='True')
    plt.plot(predicted_values, color='red', label=f'{model_config.MODEL_NAME} Prediction')
    plt.title(f'True and Predicted Values (Model: {model_config.MODEL_NAME})')
    plt.xlabel('Observations')
    plt.ylabel('Ratio')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/true_vs_predicted_{model_config.MODEL_NAME}.png', dpi=300, bbox_inches='tight')
    plt.clf()

def group_data_by_date(
    fx_timestamps,
    news_timestamps,
    true_values: List[float],
    predicted_values: Union[List[float], Dict[str, List[float]]],
    bid_prices: List[float],
    ask_prices: List[float],
    news_sentiments: List[float],
):
    """Group all test data by date for trading simulation.

    predicted_values can be:
        - list of floats
        - dict[str, list[float]] (per-model predictions)
    """

    chunked_values = defaultdict(lambda: {
        "fx_timestamps": [],
        "news_timestamps": [],
        "true_values": [],
        "predicted_values": [] if isinstance(predicted_values, list) else {},
        "bid_prices": [],
        "ask_prices": [],
        "news_sentiments": [],
    })

    n = len(true_values)
    # Sanity check (optional)
    # assert len(fx_timestamps) == len(bid_prices) == len(ask_prices) == n

    for i in range(n):
        fx_timestamp = fx_timestamps[i]
        true_val = true_values[i]
        bid_price = bid_prices[i]
        ask_price = ask_prices[i]

        date_key = fx_timestamp.date()

        day_bucket = chunked_values[date_key]
        day_bucket["fx_timestamps"].append(fx_timestamp)
        day_bucket["true_values"].append(true_val)
        day_bucket["bid_prices"].append(bid_price)
        day_bucket["ask_prices"].append(ask_price)

        # Handle predicted_values as list or dict
        if isinstance(predicted_values, list):
            pred_val = predicted_values[i]
            day_bucket["predicted_values"].append(pred_val)
        elif isinstance(predicted_values, dict):
            # Ensure dict structure exists per date
            if not isinstance(day_bucket["predicted_values"], dict):
                day_bucket["predicted_values"] = {}

            for model_name, preds in predicted_values.items():
                # Initialize list for this model on this date if needed
                if model_name not in day_bucket["predicted_values"]:
                    day_bucket["predicted_values"][model_name] = []
                day_bucket["predicted_values"][model_name].append(preds[i])

    # Attach news by date
    for news_timestamp, news_sentiment in zip(news_timestamps, news_sentiments):
        date_key = news_timestamp.date()
        if date_key not in chunked_values:
            print(f"News Date key not found: {date_key}")
        chunked_values[date_key]["news_timestamps"].append(news_timestamp)
        chunked_values[date_key]["news_sentiments"].append(news_sentiment)

    return chunked_values

# def p_value_null_greater(simulated, observed):
#     """
#     One-sided non-parametric p-value.
#     Tests H0: null distribution generates values >= observed.
#     """
#     sims = np.array(simulated)
#     N = len(sims)

#     if N == 0:
#         raise ValueError("simulated list must not be empty")

#     count = np.sum(sims >= observed)
    
#     # Add 1 to avoid p = 0
#     return (count + 1) / (N + 1)

def run_ml_based_trading_strategies(model_config):
    """Run a trading strategy based on a supervised learning model, including model training, prediction,
    and evaluation of trading performance."""
    set_seed(model_config.SEED)
    torch.set_float32_matmul_precision("high")

    # Initialize metrics evaluator
    eval_metrics = ModelEvaluationMetrics()

    # Initialize data processor
    dataProcessor = DataProcessor(model_config)

    # Train model and get predictions
    if model_config.MODEL_NAME == 'toto':
        predictor = TotoFinancialForecastingModel(dataProcessor, model_config)
        test_series, test_fx_timestamps, test_bid_prices, test_ask_prices, test_news_timestamps, test_news_sentiments = predictor.split_and_scale_data()
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.test_mid_prices
    elif model_config.MODEL_NAME == 'chronos':
        predictor = ChronosFinancialForecastingModel(dataProcessor, model_config)
        test_series, test_fx_timestamps, test_bid_prices, test_ask_prices, test_news_timestamps, test_news_sentiments = predictor.split_and_scale_data()
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.test_mid_prices
    elif model_config.MODEL_NAME == 'ensemble':
        predictor1 = DartsFinancialForecastingModel(dataProcessor, model_config)
        train_series, valid_series, test_series, test_fx_timestamps, test_bid_prices, test_ask_prices, test_news_timestamps, test_news_sentiments = predictor1.split_and_scale_data()
        predictor1.train(train_series)
        predicted_values1 = predictor1.generate_predictions(test_series)
        model_config.MODEL_NAME = "chronos"
        predictor2 = ChronosFinancialForecastingModel(dataProcessor, model_config)
        test_series2, _, _, _, _, _ = predictor2.split_and_scale_data()
        predicted_values2 = predictor2.generate_predictions(test_series2)
        model_config.MODEL_NAME = "toto"
        predictor3 = TotoFinancialForecastingModel(dataProcessor, model_config)
        test_series3, _, _, _, _, _ = predictor3.split_and_scale_data()
        predicted_values3 = predictor3.generate_predictions(test_series3)
        predicted_values = {
            **predicted_values1,
            "chronos": predicted_values2,
            "toto": predicted_values3
        }
        for key, value in predicted_values.items():
            print(f"length of predicted values {key}: {len(value)}")
        true_values = predictor1.test_mid_prices
        print(f"length of true values: {len(true_values)}")
    else:
        predictor = DartsFinancialForecastingModel(dataProcessor, model_config)
        train_series, valid_series, test_series, test_fx_timestamps, test_bid_prices, test_ask_prices, test_news_timestamps, test_news_sentiments = predictor.split_and_scale_data()
        predictor.train(train_series, valid_series)
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.test_mid_prices
        # noise_std = 0.001

        # noisy_values = [
        #     v + random.gauss(0, noise_std)
        #     for v in true_values
        # ]

        # predicted_values = [max(v, 0) for v in noisy_values]

    # Calculate and print the prediction error.
    # prediction_error_model = eval_metrics.calculate_prediction_error(predicted_values, true_values)
    # print(f"\nPrediction Error for {model_config.MODEL_NAME}: {prediction_error_model}")
    # print (f"\n")

    # plot_prediction_comparison(true_values, predicted_values, model_config)

    # Group data by date for trading simulation
    chunked_values = group_data_by_date(
        test_fx_timestamps,
        test_news_timestamps,
        true_values,
        predicted_values,
        test_bid_prices,
        test_ask_prices,
        test_news_sentiments
    )

    # Use Kelly
    print("--------------------------------")
    print("Using Kelly")
    print("--------------------------------")
    print("\n")

    # total_profits = []

    # actual_profit = -20401.40

    # for i in range(10000):

    mean_reversion_profit = []
    trend_profit = []
    forecast_based_profit = []
    news_sentiment_profit = []
    ensemble_profit = []

    mean_reversion_num_trades = []
    trend_num_trades = []
    forecast_based_num_trades = []
    news_sentiment_num_trades = []
    ensemble_num_trades = []

    for _, values in chunked_values.items():
        if (len(values['true_values']) < 7):
            continue

        trading_strategy = TradingStrategy(model_config.WALLET_A, model_config.WALLET_B, model_config.NEWS_HOLD_MINUTES, True, model_config.ENABLE_TRANSACTION_COSTS, model_config.ALLOW_NEWS_OVERLAP)
        trading_strategy.simulate_trading_with_strategies(values['fx_timestamps'], values['true_values'], values['predicted_values'], values['bid_prices'], values['ask_prices'], values['news_timestamps'], values['news_sentiments'])
        # trading_strategy.simulate_trading_with_ensemble_strategy(values['fx_timestamps'], values['true_values'], values['predicted_values'], values['bid_prices'], values['ask_prices'], values['news_timestamps'], values['news_sentiments'])
        mean_reversion_profit.append(trading_strategy.total_profit_or_loss["mean_reversion"])
        trend_profit.append(trading_strategy.total_profit_or_loss["trend"])
        forecast_based_profit.append(trading_strategy.total_profit_or_loss["forecast_based"])
        news_sentiment_profit.append(trading_strategy.total_profit_or_loss["news_sentiment"])
        ensemble_profit.append(trading_strategy.total_profit_or_loss["ensemble"])

        mean_reversion_num_trades.append(trading_strategy.num_trades["mean_reversion"])
        trend_num_trades.append(trading_strategy.num_trades["trend"])
        forecast_based_num_trades.append(trading_strategy.num_trades["forecast_based"])
        news_sentiment_num_trades.append(trading_strategy.num_trades["news_sentiment"])
        ensemble_num_trades.append(trading_strategy.num_trades["ensemble"])

        # total_profits.append(sum(news_sentiment_profit))

    cumulative_mean_reversion_profit = np.cumsum(mean_reversion_profit)
    cumulative_trend_profit = np.cumsum(trend_profit)
    cumulative_forecast_based_profit = np.cumsum(forecast_based_profit)
    cumulative_news_sentiment_profit = np.cumsum(news_sentiment_profit)
    cumulative_ensemble_profit = np.cumsum(ensemble_profit)

    # p_value = p_value_null_greater(total_profits, actual_profit)
    # print(f"P-value: {p_value:.4f}")

    plt.plot(cumulative_mean_reversion_profit, color='purple', label='Mean Reversion Strategy')
    plt.plot(cumulative_trend_profit, color='blue', label='Trend Strategy')
    plt.plot(cumulative_forecast_based_profit, color='red', label=f'Forecast-Based Strategy')
    # plt.plot(cumulative_news_sentiment_profit, color='pink', label='News Sentiment Strategy')
    # plt.plot(cumulative_ensemble_profit, color='orange', label='Ensemble Strategy')
    plt.title('Cumulative Profits Using Kelly-Based Position Sizing')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Profit (USD)')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/cumulative_profits_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    print("Kelly")
    print(f"Model: {model_config.MODEL_NAME}")
    print(f"Input Chunk Length: {model_config.INPUT_CHUNK_LENGTH}")
    print(f"Sentiment Source: {model_config.SENTIMENT_SOURCE}")
    print(f"News Hold Minutes: {model_config.NEWS_HOLD_MINUTES}")
    print(f"Cummulative Mean Reversion Profit: {cumulative_mean_reversion_profit[-1]}")
    print(f"Cummulative Trend Profit: {cumulative_trend_profit[-1]}")
    print(f"Cummulative Forecast-Based Profit: {cumulative_forecast_based_profit[-1]}")
    print(f"Cummulative News Sentiment Profit: {cumulative_news_sentiment_profit[-1]:.2f}")
    print(f"Cummulative Ensemble Profit: {cumulative_ensemble_profit[-1]}")
    # print(f"Profit per Trade: {cumulative_news_sentiment_profit[-1] / sum(news_sentiment_num_trades):.3f}")
    # print(f"Profit per Trade: {cumulative_ensemble_profit[-1] / sum(ensemble_num_trades):.3f}")

    # Use Fixed Position
    # print('\n')
    # print("--------------------------------")
    # print("Using Fixed Position")
    # print("--------------------------------")
    # print("\n")
    # mean_reversion_profit = []
    # trend_profit = []
    # forecast_based_profit = []
    # news_sentiment_profit = []
    # # ensemble_profit = []

    # mean_reversion_num_trades = []
    # trend_num_trades = []
    # forecast_based_num_trades = []
    # news_sentiment_num_trades = []
    # # ensemble_num_trades = []

    # for _, values in chunked_values.items():
    #     if (len(values['true_values']) < 7):
    #         continue

    #     trading_strategy = TradingStrategy(model_config.WALLET_A, model_config.WALLET_B, model_config.NEWS_HOLD_MINUTES, False, model_config.ENABLE_TRANSACTION_COSTS, model_config.ALLOW_NEWS_OVERLAP)
    #     trading_strategy.simulate_trading_with_strategies(values['fx_timestamps'], values['true_values'], values['predicted_values'], values['bid_prices'], values['ask_prices'], values["news_timestamps"], values["news_sentiments"])
    #     mean_reversion_profit.append(trading_strategy.total_profit_or_loss["mean_reversion"])
    #     trend_profit.append(trading_strategy.total_profit_or_loss["trend"])
    #     forecast_based_profit.append(trading_strategy.total_profit_or_loss["forecast_based"])
    #     news_sentiment_profit.append(trading_strategy.total_profit_or_loss["news_sentiment"])
    #     # ensemble_profit.append(trading_strategy.total_profit_or_loss["ensemble"])

    #     mean_reversion_num_trades.append(trading_strategy.num_trades["mean_reversion"])
    #     trend_num_trades.append(trading_strategy.num_trades["trend"])
    #     forecast_based_num_trades.append(trading_strategy.num_trades["forecast_based"])
    #     news_sentiment_num_trades.append(trading_strategy.num_trades["news_sentiment"])
    #     # ensemble_num_trades.append(trading_strategy.num_trades["ensemble"])

    # cumulative_mean_reversion_profit = np.cumsum(mean_reversion_profit)
    # cumulative_trend_profit = np.cumsum(trend_profit)
    # cumulative_forecast_based_profit = np.cumsum(forecast_based_profit)
    # cumulative_news_sentiment_profit = np.cumsum(news_sentiment_profit)
    # # cumulative_ensemble_profit = np.cumsum(ensemble_profit)

    # plt.plot(cumulative_mean_reversion_profit, color='purple', label='Mean Reversion Strategy')
    # plt.plot(cumulative_trend_profit, color='blue', label='Trend Strategy')
    # plt.plot(cumulative_forecast_based_profit, color='red', label=f'Forecast-Based Strategy')
    # # plt.plot(cumulative_news_sentiment_profit, color='pink', label='News Sentiment Strategy')
    # # plt.plot(cumulative_ensemble_profit, color='orange', label='Ensemble Strategy')
    # plt.title('Cumulative Profits Using Fixed Postion Size')
    # plt.xlabel('Days')
    # plt.ylabel('Cumulative Profit (USD)')
    # plt.legend()
    # plt.savefig(f'{model_config.OUTPUT_DIR}/cumulative_profits_fixed.png', dpi=300, bbox_inches='tight')

    # plt.clf()

    # print("Fixed Position")
    # print(f"Model: {model_config.MODEL_NAME}")
    # print(f"Input Chunk Length: {model_config.INPUT_CHUNK_LENGTH}")
    # print(f"Cummulative Forecast-Based Profit: {cumulative_forecast_based_profit[-1]}")
    # print(f"Cummulative News Sentiment Profit: {cumulative_news_sentiment_profit[-1]}")

def run(args):
    """Parse command-line arguments and configure the trading model, then run the trading strategy."""
    model_config = ModelConfig()
    model_config.INPUT_CHUNK_LENGTH = args.input_chunk_length
    model_config.OUTPUT_CHUNK_LENGTH = args.output_chunk_length
    model_config.N_EPOCHS = args.n_epochs
    model_config.TRAIN_BATCH_SIZE = args.train_batch_size
    model_config.EVAL_BATCH_SIZE = args.eval_batch_size
    model_config.FX_DATA_PATH_TRAIN = args.fx_data_path_train
    model_config.FX_DATA_PATH_VAL = args.fx_data_path_val
    model_config.FX_DATA_PATH_TEST = args.fx_data_path_test
    model_config.NEWS_DATA_PATH_TRAIN = args.news_data_path_train
    model_config.NEWS_DATA_PATH_TEST = args.news_data_path_test
    model_config.WALLET_A = args.wallet_a
    model_config.WALLET_B = args.wallet_b
    model_config.USE_FRAC_KELLY = args.use_frac_kelly
    model_config.ENABLE_TRANSACTION_COSTS = args.enable_transaction_costs
    model_config.NEWS_HOLD_MINUTES = args.news_hold_minutes
    model_config.ALLOW_NEWS_OVERLAP = args.allow_news_overlap
    model_config.SENTIMENT_SOURCE = args.sentiment_source
    model_config.SEED = args.seed

    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_config.OUTPUT_DIR = os.path.join(root_dir, args.output_dir)
    os.makedirs(model_config.OUTPUT_DIR, exist_ok=True)

    model_config.MODEL_NAME = args.model_name

    print_model_config(model_config)

    # Run the trading strategy based on the specified model and configuration.
    run_ml_based_trading_strategies(model_config)

def print_model_config(config):
    print("\n🔧 Model Configuration:")
    print(f"  Input Chunk Length        : {config.INPUT_CHUNK_LENGTH}")
    print(f"  Output Chunk Length       : {config.OUTPUT_CHUNK_LENGTH}")
    print(f"  Number of Epochs          : {config.N_EPOCHS}")
    print(f"  Train Batch Size          : {config.TRAIN_BATCH_SIZE}")
    print(f"  Eval Batch Size           : {config.EVAL_BATCH_SIZE}")
    print(f"  FX Data Path Train        : {config.FX_DATA_PATH_TRAIN}")
    print(f"  FX Data Path Val          : {config.FX_DATA_PATH_VAL}")
    print(f"  FX Data Path Test         : {config.FX_DATA_PATH_TEST}")
    print(f"  News Data Path Train      : {config.NEWS_DATA_PATH_TRAIN}")
    print(f"  News Data Path Test       : {config.NEWS_DATA_PATH_TEST}")
    print(f"  Wallet A Initial Amount   : {config.WALLET_A}")
    print(f"  Wallet B Initial Amount   : {config.WALLET_B}")
    print(f"  Fractional Kelly Enabled  : {config.USE_FRAC_KELLY}")
    print(f"  Transaction Costs Enabled : {config.ENABLE_TRANSACTION_COSTS}")
    print(f"  Output Directory          : {config.OUTPUT_DIR}")
    print(f"  News Hold Minutes         : {config.NEWS_HOLD_MINUTES}")
    print(f"  Allow News Overlap        : {config.ALLOW_NEWS_OVERLAP}")
    print(f"  Sentiment Source          : {config.SENTIMENT_SOURCE}")
    print(f"  Seed                      : {config.SEED}")
    print(f"  Model Name                : {config.MODEL_NAME}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--wallet_a", type=float, default=1000000.0, help="Amount of money in wallet A (currency A).")
    parser.add_argument("--wallet_b", type=float, default=1000000.0, help="Amount of money in wallet B (currency B).")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "nbeats",
            "nhits",
            "tcn",
            "toto",
            "chronos",
            "ensemble"
        ],
        default="tcn",
        help="Specify the model to use. Default is 'tcn'."
    )
    parser.add_argument("--input_chunk_length", type=int, default=64, help="Length of the input sequences.")
    parser.add_argument("--output_chunk_length", type=int, default=1, help="Length of the output sequences.")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument(
        "--fx_data_path_train",
        type=str,
        default="",
        help="Path to the fx training data. Currency rates should be provided as 1 A / 1 B, where A and B are the respective currencies.",
        required=True
    )
    parser.add_argument(
        "--fx_data_path_val",
        type=str,
        default="",
        help="Path to the fx validation data. Currency rates should be provided as 1 A / 1 B, where A and B are the respective currencies.",
        required=True
    )
    parser.add_argument(
        "--fx_data_path_test",
        type=str,
        default="",
        help="Path to the fx test data. Currency rates should be provided as 1 A / 1 B, where A and B are the respective currencies.",
        required=True
    )
    parser.add_argument("--news_data_path_train", type=str, default="", help="Path to the news training data.", required=True)
    parser.add_argument("--news_data_path_test", type=str, default="", help="Path to the news test data.", required=True)
    parser.add_argument("--use_frac_kelly", action="store_true", help="Use fractional Kelly to size bets. Default is False.")
    parser.add_argument("--enable_transaction_costs", action="store_true", help="Enable transaction costs. Default is False.")
    parser.add_argument("--output_dir", type=str, default="results/usd-cny-2023", help="Directory to save all outputs.")
    parser.add_argument(
        "--news_hold_minutes",
        type=int,
        default=-1,
        help="Number of minutes to hold a position before allowing exit for news sentiment strategy.",
    )
    parser.add_argument(
        "--sentiment_source",
        type=str,
        default="competitor_label",
        help="Choose which sentiment label column to use for trading."
    )
    parser.add_argument(
        "--allow_news_overlap",
        action="store_true",
        help="Enable overlapping news sentiment trades. When set, multiple news-driven positions may be open at the same time. Default: disabled.")

    args = parser.parse_args()

    run(args)