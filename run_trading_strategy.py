# importing to set up reproducibility
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from utils import FXTradingConfig
from data_processing import DataProcessor
from models import DartsFinancialForecastingModel, ChronosFinancialForecastingModel, TotoFinancialForecastingModel
import numpy as np
import argparse
from strategies import TradingStrategy
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import random
from typing import List, Dict, Union

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def run_ml_based_trading_strategies(fx_trading_config):
    """Run a trading strategy based on a supervised learning model, including model training, prediction,
    and evaluation of trading performance."""
    set_seed(fx_trading_config.SEED)
    torch.set_float32_matmul_precision("high")

    # Initialize data processor and split/scale data ONCE
    dataProcessor = DataProcessor(fx_trading_config)
    processed_data = dataProcessor.split_and_scale_data()

    # Extract test metadata from processed data
    test_fx_timestamps = processed_data.test_fx_timestamps
    test_bid_prices = processed_data.test_bid_prices
    test_ask_prices = processed_data.test_ask_prices
    test_news_timestamps = processed_data.test_news_timestamps
    test_news_sentiments = processed_data.test_news_sentiments
    true_values = processed_data.test_mid_prices

    # Train model and get predictions
    if fx_trading_config.MODEL_NAME == 'toto':
        predictor = TotoFinancialForecastingModel(fx_trading_config, processed_data.llm_scaler)
        predicted_values = predictor.generate_predictions(processed_data.llm_test_scaled)
    elif fx_trading_config.MODEL_NAME == 'chronos':
        predictor = ChronosFinancialForecastingModel(fx_trading_config, processed_data.llm_scaler)
        predicted_values = predictor.generate_predictions(processed_data.llm_test_scaled)
    elif fx_trading_config.MODEL_NAME == 'ensemble':
        # Darts models
        predictor1 = DartsFinancialForecastingModel(fx_trading_config, processed_data.darts_scaler)
        predictor1.train(processed_data.darts_train_scaled, processed_data.darts_val_scaled)
        predicted_values1 = predictor1.generate_predictions(processed_data.darts_test_scaled)

        # Chronos model
        predictor2 = ChronosFinancialForecastingModel(fx_trading_config, processed_data.llm_scaler)
        predicted_values2 = predictor2.generate_predictions(processed_data.llm_test_scaled)

        # Toto model
        predictor3 = TotoFinancialForecastingModel(fx_trading_config, processed_data.llm_scaler)
        predicted_values3 = predictor3.generate_predictions(processed_data.llm_test_scaled)

        predicted_values = {
            **predicted_values1,
            "chronos": predicted_values2,
            "toto": predicted_values3
        }
    else:
        predictor = DartsFinancialForecastingModel(fx_trading_config, processed_data.darts_scaler)
        predictor.train(processed_data.darts_train_scaled, processed_data.darts_val_scaled)
        predicted_values = predictor.generate_predictions(processed_data.darts_test_scaled)

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

    mean_reversion_profit = []
    trend_profit = []
    model_driven_profit = []
    news_sentiment_profit = []
    ensemble_profit = []

    mean_reversion_num_trades = []
    trend_num_trades = []
    model_driven_num_trades = []
    news_sentiment_num_trades = []
    ensemble_num_trades = []

    # True if we are running the ensemble meta-model
    is_ensemble_model = fx_trading_config.MODEL_NAME == 'ensemble'

    trading_strategy = TradingStrategy(
        fx_trading_config.WALLET_A,
        fx_trading_config.WALLET_B,
        fx_trading_config.NEWS_HOLD_MINUTES,
        fx_trading_config.BET_SIZING,
        fx_trading_config.ENABLE_TRANSACTION_COSTS,
        fx_trading_config.ALLOW_NEWS_OVERLAP,
    )

    for date_key, values in chunked_values.items():
        prev_mean_reversion_profit = sum(trading_strategy.pnl["mean_reversion"])
        prev_trend_profit = sum(trading_strategy.pnl["trend"])
        prev_model_driven_profit = sum(trading_strategy.pnl["model_driven"])
        prev_news_sentiment_profit = sum(trading_strategy.pnl["news_sentiment"])
        prev_ensemble_profit = sum(trading_strategy.pnl["ensemble"])

        if is_ensemble_model:
            # Run mean reversion, trend, and news sentiment using only actual rates
            # (place-holder predictions so that strategies not using them still work)
            trading_strategy.simulate_trading_with_strategies(
                values['fx_timestamps'],
                values['true_values'],
                values['true_values'],
                values['bid_prices'],
                values['ask_prices'],
                values['news_timestamps'],
                values['news_sentiments'],
                strategy_names=['mean_reversion', 'trend'],
            )

            # Run the ensemble meta-model strategy using per-model predictions
            trading_strategy.simulate_trading_with_ensemble_strategy(
                values['fx_timestamps'],
                values['true_values'],
                values['predicted_values'],
                values['bid_prices'],
                values['ask_prices'],
                seed=fx_trading_config.SEED,
            )
        else:
            # Non-ensemble models: run mean reversion, trend, model-driven, and news sentiment
            trading_strategy.simulate_trading_with_strategies(
                values['fx_timestamps'],
                values['true_values'],
                values['predicted_values'],
                values['bid_prices'],
                values['ask_prices'],
                values['news_timestamps'],
                values['news_sentiments'],
            )

        current_mean_reversion_profit = sum(trading_strategy.pnl["mean_reversion"])
        current_trend_profit = sum(trading_strategy.pnl["trend"])
        current_model_driven_profit = sum(trading_strategy.pnl["model_driven"])
        current_news_sentiment_profit = sum(trading_strategy.pnl["news_sentiment"])
        current_ensemble_profit = sum(trading_strategy.pnl["ensemble"])

        mean_reversion_profit.append(current_mean_reversion_profit - prev_mean_reversion_profit)
        trend_profit.append(current_trend_profit - prev_trend_profit)
        news_sentiment_profit.append(current_news_sentiment_profit - prev_news_sentiment_profit)

        if is_ensemble_model:
            ensemble_profit.append(current_ensemble_profit - prev_ensemble_profit)
        else:
            model_driven_profit.append(current_model_driven_profit - prev_model_driven_profit)

    mean_reversion_num_trades.append(trading_strategy.num_trades["mean_reversion"])
    trend_num_trades.append(trading_strategy.num_trades["trend"])
    news_sentiment_num_trades.append(trading_strategy.num_trades["news_sentiment"])

    if is_ensemble_model:
        ensemble_num_trades.append(trading_strategy.num_trades["ensemble"])
    else:
        model_driven_num_trades.append(trading_strategy.num_trades["model_driven"])

    cumulative_mean_reversion_profit = np.cumsum(mean_reversion_profit)
    cumulative_trend_profit = np.cumsum(trend_profit)
    cumulative_news_sentiment_profit = np.cumsum(news_sentiment_profit)

    if is_ensemble_model:
        cumulative_ensemble_profit = np.cumsum(ensemble_profit)
    else:
        cumulative_model_driven_profit = np.cumsum(model_driven_profit)

    plt.plot(cumulative_mean_reversion_profit, color='purple', label='Mean Reversion Strategy')
    plt.plot(cumulative_trend_profit, color='blue', label='Trend Strategy')
    plt.plot(cumulative_news_sentiment_profit, color='black', label=f'News-LLM Strategy ({fx_trading_config.SENTIMENT_SOURCE})')

    if is_ensemble_model:
        plt.plot(cumulative_ensemble_profit, color='orange', label='Ensemble Strategy')
    else:
        plt.plot(cumulative_model_driven_profit, color='red', label=f'Model-Driven Strategy ({fx_trading_config.MODEL_NAME})')
    plt.title('Cumulative Profits')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Profit (USD)')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(f'{fx_trading_config.OUTPUT_DIR}/cumulative_profits.png', dpi=300, bbox_inches='tight')

    plt.clf()

    print(f"Cummulative Mean Reversion Profit: {cumulative_mean_reversion_profit[-1]:.2f}")
    print(f"Cummulative Trend Profit: {cumulative_trend_profit[-1]:.2f}")
    print(f"Cummulative News Sentiment Profit: {cumulative_news_sentiment_profit[-1]:.2f}")
    if is_ensemble_model:
        print(f"Cummulative Ensemble Profit: {cumulative_ensemble_profit[-1]:.2f}")
    else:
        print(f"Cummulative Model-Driven Profit: {cumulative_model_driven_profit[-1]:.2f}")
    
def run(args):
    """Parse command-line arguments and configure the trading model, then run the trading strategy."""
    fx_trading_config = FXTradingConfig()
    fx_trading_config.INPUT_CHUNK_LENGTH = args.input_chunk_length
    fx_trading_config.OUTPUT_CHUNK_LENGTH = args.output_chunk_length
    fx_trading_config.N_EPOCHS = args.n_epochs
    fx_trading_config.TRAIN_BATCH_SIZE = args.train_batch_size
    fx_trading_config.EVAL_BATCH_SIZE = args.eval_batch_size
    fx_trading_config.FX_DATA_PATH_TRAIN = args.fx_data_path_train
    fx_trading_config.FX_DATA_PATH_VAL = args.fx_data_path_val
    fx_trading_config.FX_DATA_PATH_TEST = args.fx_data_path_test
    fx_trading_config.NEWS_DATA_PATH_TRAIN = args.news_data_path_train
    fx_trading_config.NEWS_DATA_PATH_TEST = args.news_data_path_test
    fx_trading_config.WALLET_A = args.wallet_a
    fx_trading_config.WALLET_B = args.wallet_b
    fx_trading_config.BET_SIZING = args.bet_sizing
    fx_trading_config.ENABLE_TRANSACTION_COSTS = args.enable_transaction_costs
    fx_trading_config.NEWS_HOLD_MINUTES = args.news_hold_minutes
    fx_trading_config.ALLOW_NEWS_OVERLAP = args.allow_news_overlap
    fx_trading_config.SENTIMENT_SOURCE = args.sentiment_source
    fx_trading_config.SEED = args.seed

    root_dir = os.path.dirname(os.path.abspath(__file__))
    fx_trading_config.OUTPUT_DIR = os.path.join(root_dir, args.output_dir)
    os.makedirs(fx_trading_config.OUTPUT_DIR, exist_ok=True)

    fx_trading_config.MODEL_NAME = args.model_name

    print_fx_trading_config(fx_trading_config)

    # Run the trading strategy based on the specified model and configuration.
    run_ml_based_trading_strategies(fx_trading_config)

def print_fx_trading_config(config):
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
    print(f"  Bet Sizing                : {config.BET_SIZING}")
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
            "arima",
            "nbeats",
            "nhits",
            "tcn",
            "toto",
            "chronos",
            "ensemble",
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
    parser.add_argument(
        "--bet_sizing",
        type=str,
        choices=["active_kelly", "passive_kelly", "fixed"],
        default="fixed",
        help="Bet sizing strategy: active_kelly, passive_kelly, or fixed. Default is fixed."
    )
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