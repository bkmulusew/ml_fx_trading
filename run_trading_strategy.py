from utils import ModelConfig
from data_processing import DataProcessor
from models import DartsFinancialForecastingModel, ChronosFinancialForecastingModel
from metrics import ModelEvaluationMetrics
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from strategies import TradingStrategy
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

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

def group_data_by_date(dates, true_values, predicted_values, bid_prices, ask_prices, with_prompt_values, without_prompt_values):
    """Group all test data by date for trading simulation."""
    # Parse test_dates into datetime objects
    
    # Create a dictionary to group values by date
    chunked_values = defaultdict(lambda: {
        "true_values": [], 
        "predicted_values": [],
        "bid_price": [], 
        "ask_price": [], 
        "with_prompt": [], 
        "without_prompt": []
    })
    
    # Chunk data by date
    for date, true_val, pred_val, bid_price, ask_price, with_prompt_val, without_prompt_val in zip(
            dates, true_values, predicted_values, bid_prices, ask_prices, with_prompt_values, without_prompt_values
        ):
        date_key = date.date()  # Use only the date part as the key
        chunked_values[date_key]["true_values"].append(true_val)
        chunked_values[date_key]["predicted_values"].append(pred_val)
        chunked_values[date_key]["bid_price"].append(bid_price)
        chunked_values[date_key]["ask_price"].append(ask_price)
        chunked_values[date_key]["with_prompt"].append(with_prompt_val)
        chunked_values[date_key]["without_prompt"].append(without_prompt_val)
    
    return chunked_values

def run_sl_based_trading_strategy(model_config):
    """Run a trading strategy based on a supervised learning model, including model training, prediction,
    and evaluation of trading performance."""
    # Initialize metrics evaluator
    eval_metrics = ModelEvaluationMetrics()

    # Initialize data processor
    dataProcessor = DataProcessor(model_config)

    # Train model and get predictions
    if model_config.MODEL_NAME == 'bilstm':
        predictor = PytorchFinancialForecastingModel(dataProcessor, model_config)
        processed_data = predictor.split_and_scale_data()
        predictor.train(processed_data['x_train'], processed_data['y_train'], processed_data['x_valid'], processed_data['y_valid'])
        generated_values = predictor.generate_predictions(processed_data['x_test'], processed_data['y_test'])
        predicted_values = generated_values['predicted_values']
        true_values = generated_values['true_values']
    elif model_config.MODEL_NAME == 'toto':
        predictor = TotoFinancialForecastingModel(dataProcessor, model_config)
        _, _, test_series, test_dates, test_bid_prices, test_ask_prices, test_with_prompt, test_without_prompt = predictor.split_and_scale_data()
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.get_true_values(test_series)
    elif model_config.MODEL_NAME == 'chronos':
        predictor = ChronosFinancialForecastingModel(dataProcessor, model_config)
        _, _, test_series, test_dates, test_bid_prices, test_ask_prices, test_with_prompt, test_without_prompt = predictor.split_and_scale_data()
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.get_true_values(test_series)
    else:
        predictor = DartsFinancialForecastingModel(dataProcessor, model_config)
        train_series, valid_series, test_series, test_dates, test_bid_prices, test_ask_prices, test_with_prompt, test_without_prompt = predictor.split_and_scale_data()
        predictor.train(train_series, valid_series)
        predicted_values = predictor.generate_predictions(test_series)
        true_values = predictor.get_true_values(test_series)

    # Calculate and print the prediction error.
    print(f"Model: {model_config.MODEL_NAME}")
    prediction_error_model = eval_metrics.calculate_prediction_error(predicted_values, true_values)
    print(f"Prediction Error for {model_config.MODEL_NAME}: {prediction_error_model}")
    print (f"\n")

    plot_prediction_comparison(true_values, predicted_values, model_config)

    parsed_test_dates = [datetime.strptime(date, "%m/%d/%Y %H:%M") for date in test_dates]

    # Group data by date for trading simulation
    chunked_values = group_data_by_date(
        parsed_test_dates,
        true_values,
        predicted_values,
        test_bid_prices,
        test_ask_prices,
        test_with_prompt,
        test_without_prompt
    )

    # Use Kelly

    print("--------------------------------")
    print("Using Kelly")
    print("--------------------------------")
    mean_reversion_profit = []
    trend_profit = []
    forecasting_profit = []
    hybrid_mean_reversion_profit = []
    hybrid_trend_profit = []
    ensemble_profit = []

    mean_reversion_profit_per_trade = []
    trend_profit_per_trade = []
    forecasting_profit_per_trade = []
    hybrid_mean_reversion_profit_per_trade = []
    hybrid_trend_profit_per_trade = []
    ensemble_profit_per_trade = []

    mean_reversion_num_trades = []
    trend_num_trades = []
    forecasting_num_trades = []
    hybrid_mean_reversion_num_trades = []
    hybrid_trend_num_trades = []
    ensemble_num_trades = []

    mean_reversion_sharpe_ratios = []
    trend_sharpe_ratios = []
    forecasting_sharpe_ratios = []
    hybrid_mean_reversion_sharpe_ratios = []
    hybrid_trend_sharpe_ratios = []
    ensemble_sharpe_ratios = []

    for _, values in chunked_values.items():
        if (len(values['true_values']) < 7):
            continue

        trading_strategy = TradingStrategy(model_config.WALLET_A, model_config.WALLET_B)
        trading_strategy.simulate_trading_with_strategies(values['true_values'], values['predicted_values'], values['bid_price'], values['ask_price'], values["with_prompt"], enable_transaction_costs=model_config.ENABLE_TRANSACTION_COSTS, hold_position=model_config.HOLD_POSITION)
        mean_reversion_profit.append(trading_strategy.total_profit_or_loss["mean_reversion"])
        trend_profit.append(trading_strategy.total_profit_or_loss["trend"])
        forecasting_profit.append(trading_strategy.total_profit_or_loss["pure_forcasting"])
        hybrid_mean_reversion_profit.append(trading_strategy.total_profit_or_loss["hybrid_mean_reversion"])
        hybrid_trend_profit.append(trading_strategy.total_profit_or_loss["hybrid_trend"])
        ensemble_profit.append(trading_strategy.total_profit_or_loss["ensemble"])
        mean_reversion_profit_per_trade_val = trading_strategy.total_profit_or_loss["mean_reversion"] / trading_strategy.num_trades["mean_reversion"]
        trend_profit_per_trade_val = trading_strategy.total_profit_or_loss["trend"] / trading_strategy.num_trades["trend"]
        forecasting_profit_per_trade_val = trading_strategy.total_profit_or_loss["pure_forcasting"] / trading_strategy.num_trades["pure_forcasting"]
        hybrid_mean_reversion_profit_per_trade_val = trading_strategy.total_profit_or_loss["hybrid_mean_reversion"] / trading_strategy.num_trades["hybrid_mean_reversion"]
        hybrid_trend_profit_per_trade_val = trading_strategy.total_profit_or_loss["hybrid_trend"] / trading_strategy.num_trades["hybrid_trend"]
        ensemble_profit_per_trade_val = trading_strategy.total_profit_or_loss["ensemble"] / trading_strategy.num_trades["ensemble"]
        mean_reversion_profit_per_trade.append(mean_reversion_profit_per_trade_val)
        trend_profit_per_trade.append(trend_profit_per_trade_val)
        forecasting_profit_per_trade.append(forecasting_profit_per_trade_val)
        hybrid_mean_reversion_profit_per_trade.append(hybrid_mean_reversion_profit_per_trade_val)
        hybrid_trend_profit_per_trade.append(hybrid_trend_profit_per_trade_val)
        ensemble_profit_per_trade.append(ensemble_profit_per_trade_val)

        mean_reversion_num_trades.append(trading_strategy.num_trades["mean_reversion"])
        trend_num_trades.append(trading_strategy.num_trades["trend"])
        forecasting_num_trades.append(trading_strategy.num_trades["pure_forcasting"])
        hybrid_mean_reversion_num_trades.append(trading_strategy.num_trades["hybrid_mean_reversion"])
        hybrid_trend_num_trades.append(trading_strategy.num_trades["hybrid_trend"])
        ensemble_num_trades.append(trading_strategy.num_trades["ensemble"])

        mean_reversion_sharpe_ratios.append(trading_strategy.sharpe_ratios["mean_reversion"])
        trend_sharpe_ratios.append(trading_strategy.sharpe_ratios["trend"])
        forecasting_sharpe_ratios.append(trading_strategy.sharpe_ratios["pure_forcasting"])
        hybrid_mean_reversion_sharpe_ratios.append(trading_strategy.sharpe_ratios["hybrid_mean_reversion"])
        hybrid_trend_sharpe_ratios.append(trading_strategy.sharpe_ratios["hybrid_trend"])
        ensemble_sharpe_ratios.append(trading_strategy.sharpe_ratios["ensemble"])
            
    cumulative_mean_reversion_profit = np.cumsum(mean_reversion_profit)
    cumulative_trend_profit = np.cumsum(trend_profit)
    cumulative_forecasting_profit = np.cumsum(forecasting_profit)
    cumulative_hybrid_mean_reversion_profit = np.cumsum(hybrid_mean_reversion_profit)
    cumulative_hybrid_trend_profit = np.cumsum(hybrid_trend_profit)
    cumulative_ensemble_profit = np.cumsum(ensemble_profit)

    cumulative_mean_reversion_profit_per_trade = [
        np.sum(mean_reversion_profit[:i+1]) / np.sum(mean_reversion_num_trades[:i+1]) for i in range(len(mean_reversion_profit))
    ]
    cumulative_trend_profit_per_trade = [
        np.sum(trend_profit[:i+1]) / np.sum(trend_num_trades[:i+1]) for i in range(len(trend_profit))
    ]
    cumulative_forecasting_profit_per_trade = [
        np.sum(forecasting_profit[:i+1]) / np.sum(forecasting_num_trades[:i+1]) for i in range(len(forecasting_profit))
    ]
    cumulative_hybrid_mean_reversion_profit_per_trade = [
        np.sum(hybrid_mean_reversion_profit[:i+1]) / np.sum(hybrid_mean_reversion_num_trades[:i+1]) for i in range(len(hybrid_mean_reversion_profit))
    ]
    cumulative_hybrid_trend_profit_per_trade = [
        np.sum(hybrid_trend_profit[:i+1]) / np.sum(hybrid_trend_num_trades[:i+1]) for i in range(len(hybrid_trend_profit))
    ]
    cumulative_ensemble_profit_per_trade = [
        np.sum(ensemble_profit[:i+1]) / np.sum(ensemble_num_trades[:i+1]) for i in range(len(ensemble_profit))
    ]

    plt.plot(cumulative_mean_reversion_profit_per_trade, color='purple', label='Cumulative Mean Reversion Profit Per Trade')
    plt.plot(cumulative_trend_profit_per_trade, color='blue', label='Cumulative Trend Profit Per Trade')
    plt.plot(cumulative_forecasting_profit_per_trade, color='red', label=f'Cumulative Forecasting Profit Per Trade {model_config.MODEL_NAME}')
    plt.plot(cumulative_hybrid_mean_reversion_profit_per_trade, color='green', label='Cumulative Hybrid Mean Reversion Profit Per Trade')
    plt.plot(cumulative_hybrid_trend_profit_per_trade, color='brown', label='Cumulative Hybrid Trend Profit Per Trade')
    plt.plot(cumulative_ensemble_profit_per_trade, color='orange', label='Cumulative Ensemble Profit Per Trade')
    plt.title('Cumulative Profits Per Trade Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Profit Per Trade')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/cumulative_profits_per_trade_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(cumulative_mean_reversion_profit, color='purple', label='Cumulative Mean Reversion Profit')
    plt.plot(cumulative_trend_profit, color='blue', label='Cumulative Trend Profit')
    plt.plot(cumulative_forecasting_profit, color='red', label=f'Cumulative Forecasting Profit {model_config.MODEL_NAME}')
    plt.plot(cumulative_hybrid_mean_reversion_profit, color='green', label='Cumulative Hybrid Mean Reversion Profit')
    plt.plot(cumulative_hybrid_trend_profit, color='brown', label='Cumulative Hybrid Trend Profit')
    plt.plot(cumulative_ensemble_profit, color='orange', label='Cumulative Ensemble Profit')
    plt.title('Cumulative Profits Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/cumulative_profits_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_profit, color='purple', label='Mean Reversion Profit')
    plt.plot(trend_profit, color='blue', label='Trend Profit')
    plt.plot(forecasting_profit, color='red', label=f'Forcasting Profit {model_config.MODEL_NAME}')
    plt.plot(hybrid_mean_reversion_profit, color='green', label='Hybrid Mean Reversion Profit')
    plt.plot(hybrid_trend_profit, color='brown', label='Hybrid Trend Profit')
    plt.plot(ensemble_profit, color='orange', label='Ensemble Profit')
    plt.title('Profits Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Profit')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/profits_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_profit_per_trade, color='purple', label='Mean Reversion Profit Per Trade')
    plt.plot(trend_profit_per_trade, color='blue', label='Trend Profit Per Trade')
    plt.plot(forecasting_profit_per_trade, color = 'red', label = f'Forcasting Profit Per Trade {model_config.MODEL_NAME}')
    plt.plot(hybrid_mean_reversion_profit_per_trade, color='green', label='Hybrid Mean Reversion Profit Per Trade')
    plt.plot(hybrid_trend_profit_per_trade, color='brown', label='Hybrid Trend Profit Per Trade')
    plt.plot(ensemble_profit_per_trade, color='orange', label='Ensemble Profit Per Trade')
    plt.title('Profit Per Trade Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Profit Per Trade')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/profit_per_trade_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_sharpe_ratios, color='purple', label='Mean Reversion Sharpe Ratio')
    plt.plot(trend_sharpe_ratios, color='blue', label='Trend Sharpe Ratio')
    plt.plot(forecasting_sharpe_ratios, color='red', label=f'Forecasting Sharpe Ratio {model_config.MODEL_NAME}')
    plt.plot(hybrid_mean_reversion_sharpe_ratios, color='green', label='Hybrid Mean Reversion Sharpe Ratio')
    plt.plot(hybrid_trend_sharpe_ratios, color='brown', label='Hybrid Trend Sharpe Ratio')
    plt.plot(ensemble_sharpe_ratios, color='orange', label='Ensemble Sharpe Ratio')
    plt.title('Sharpe Ratios Using Kelly')
    plt.xlabel('Observation')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/sharpe_ratios_kelly.png', dpi=300, bbox_inches='tight')

    plt.clf()

    # Use Fixed Position
    
    print('\n')
    print("--------------------------------")
    print("Using Fixed Position")
    print("--------------------------------")

    mean_reversion_profit = []
    trend_profit = []
    forecasting_profit = []
    hybrid_mean_reversion_profit = []
    hybrid_trend_profit = []
    ensemble_profit = []

    mean_reversion_profit_per_trade = []
    trend_profit_per_trade = []
    forecasting_profit_per_trade = []
    hybrid_mean_reversion_profit_per_trade = []
    hybrid_trend_profit_per_trade = []
    ensemble_profit_per_trade = []

    mean_reversion_num_trades = []
    trend_num_trades = []
    forecasting_num_trades = []
    hybrid_mean_reversion_num_trades = []
    hybrid_trend_num_trades = []
    ensemble_num_trades = []

    mean_reversion_sharpe_ratios = []
    trend_sharpe_ratios = []
    forecasting_sharpe_ratios = []
    hybrid_mean_reversion_sharpe_ratios = []
    hybrid_trend_sharpe_ratios = []
    ensemble_sharpe_ratios = []

    for _, values in chunked_values.items():
        if (len(values['true_values']) < 7):
            continue

        trading_strategy = TradingStrategy(model_config.WALLET_A, model_config.WALLET_B)
        trading_strategy.simulate_trading_with_strategies(values['true_values'], values['predicted_values'], values['bid_price'], values['ask_price'], values["with_prompt"], use_kelly=False, enable_transaction_costs=model_config.ENABLE_TRANSACTION_COSTS, hold_position=model_config.HOLD_POSITION)
        mean_reversion_profit.append(trading_strategy.total_profit_or_loss["mean_reversion"])
        trend_profit.append(trading_strategy.total_profit_or_loss["trend"])
        forecasting_profit.append(trading_strategy.total_profit_or_loss["pure_forcasting"])
        hybrid_mean_reversion_profit.append(trading_strategy.total_profit_or_loss["hybrid_mean_reversion"])
        hybrid_trend_profit.append(trading_strategy.total_profit_or_loss["hybrid_trend"])
        ensemble_profit.append(trading_strategy.total_profit_or_loss["ensemble"])
        mean_reversion_profit_per_trade_val = trading_strategy.total_profit_or_loss["mean_reversion"] / trading_strategy.num_trades["mean_reversion"]
        trend_profit_per_trade_val = trading_strategy.total_profit_or_loss["trend"] / trading_strategy.num_trades["trend"]
        forecasting_profit_per_trade_val = trading_strategy.total_profit_or_loss["pure_forcasting"] / trading_strategy.num_trades["pure_forcasting"]
        hybrid_mean_reversion_profit_per_trade_val = trading_strategy.total_profit_or_loss["hybrid_mean_reversion"] / trading_strategy.num_trades["hybrid_mean_reversion"]
        hybrid_trend_profit_per_trade_val = trading_strategy.total_profit_or_loss["hybrid_trend"] / trading_strategy.num_trades["hybrid_trend"]
        ensemble_profit_per_trade_val = trading_strategy.total_profit_or_loss["ensemble"] / trading_strategy.num_trades["ensemble"]
        mean_reversion_profit_per_trade.append(mean_reversion_profit_per_trade_val)
        trend_profit_per_trade.append(trend_profit_per_trade_val)
        forecasting_profit_per_trade.append(forecasting_profit_per_trade_val)
        hybrid_mean_reversion_profit_per_trade.append(hybrid_mean_reversion_profit_per_trade_val)
        hybrid_trend_profit_per_trade.append(hybrid_trend_profit_per_trade_val)
        ensemble_profit_per_trade.append(ensemble_profit_per_trade_val)

        mean_reversion_num_trades.append(trading_strategy.num_trades["mean_reversion"])
        trend_num_trades.append(trading_strategy.num_trades["trend"])
        forecasting_num_trades.append(trading_strategy.num_trades["pure_forcasting"])
        hybrid_mean_reversion_num_trades.append(trading_strategy.num_trades["hybrid_mean_reversion"])
        hybrid_trend_num_trades.append(trading_strategy.num_trades["hybrid_trend"])
        ensemble_num_trades.append(trading_strategy.num_trades["ensemble"])

        mean_reversion_sharpe_ratios.append(trading_strategy.sharpe_ratios["mean_reversion"])
        trend_sharpe_ratios.append(trading_strategy.sharpe_ratios["trend"])
        forecasting_sharpe_ratios.append(trading_strategy.sharpe_ratios["pure_forcasting"])
        hybrid_mean_reversion_sharpe_ratios.append(trading_strategy.sharpe_ratios["hybrid_mean_reversion"])
        hybrid_trend_sharpe_ratios.append(trading_strategy.sharpe_ratios["hybrid_trend"])
        ensemble_sharpe_ratios.append(trading_strategy.sharpe_ratios["ensemble"])
            
    cumulative_mean_reversion_profit = np.cumsum(mean_reversion_profit)
    cumulative_trend_profit = np.cumsum(trend_profit)
    cumulative_forecasting_profit = np.cumsum(forecasting_profit)
    cumulative_hybrid_mean_reversion_profit = np.cumsum(hybrid_mean_reversion_profit)
    cumulative_hybrid_trend_profit = np.cumsum(hybrid_trend_profit)
    cumulative_ensemble_profit = np.cumsum(ensemble_profit)

    cumulative_mean_reversion_profit_per_trade = [
        np.sum(mean_reversion_profit[:i+1]) / np.sum(mean_reversion_num_trades[:i+1]) for i in range(len(mean_reversion_profit))
    ]
    cumulative_trend_profit_per_trade = [
        np.sum(trend_profit[:i+1]) / np.sum(trend_num_trades[:i+1]) for i in range(len(trend_profit))
    ]
    cumulative_forecasting_profit_per_trade = [
        np.sum(forecasting_profit[:i+1]) / np.sum(forecasting_num_trades[:i+1]) for i in range(len(forecasting_profit))
    ]
    cumulative_hybrid_mean_reversion_profit_per_trade = [
        np.sum(hybrid_mean_reversion_profit[:i+1]) / np.sum(hybrid_mean_reversion_num_trades[:i+1]) for i in range(len(hybrid_mean_reversion_profit))
    ]
    cumulative_hybrid_trend_profit_per_trade = [
        np.sum(hybrid_trend_profit[:i+1]) / np.sum(hybrid_trend_num_trades[:i+1]) for i in range(len(hybrid_trend_profit))
    ]
    cumulative_ensemble_profit_per_trade = [
        np.sum(ensemble_profit[:i+1]) / np.sum(ensemble_num_trades[:i+1]) for i in range(len(ensemble_profit))
    ]

    plt.plot(cumulative_mean_reversion_profit_per_trade, color='purple', label='Cumulative Mean Reversion Profit Per Trade')
    plt.plot(cumulative_trend_profit_per_trade, color='blue', label='Cumulative Trend Profit Per Trade')
    plt.plot(cumulative_forecasting_profit_per_trade, color='red', label=f'Cumulative Forecasting Profit Per Trade {model_config.MODEL_NAME}')
    plt.plot(cumulative_hybrid_mean_reversion_profit_per_trade, color='green', label='Cumulative Hybrid Mean Reversion Profit Per Trade')
    plt.plot(cumulative_hybrid_trend_profit_per_trade, color='brown', label='Cumulative Hybrid Trend Profit Per Trade')
    plt.plot(cumulative_ensemble_profit_per_trade, color='orange', label='Cumulative Ensemble Profit Per Trade')
    plt.title('Cumulative Profits Per Trade Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Profit Per Trade')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/cumulative_profits_per_trade_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(cumulative_mean_reversion_profit, color='purple', label='Cumulative Mean Reversion Profit')
    plt.plot(cumulative_trend_profit, color='blue', label='Cumulative Trend Profit')
    plt.plot(cumulative_forecasting_profit, color='red', label=f'Cumulative Forecasting Profit {model_config.MODEL_NAME}')
    plt.plot(cumulative_hybrid_mean_reversion_profit, color='green', label='Cumulative Hybrid Mean Reversion Profit')
    plt.plot(cumulative_hybrid_trend_profit, color='brown', label='Cumulative Hybrid Trend Profit')
    plt.plot(cumulative_ensemble_profit, color='orange', label='Cumulative Ensemble Profit')
    plt.title('Cumulative Profits Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/cumulative_profits_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_profit, color='purple', label='Mean Reversion Profit')
    plt.plot(trend_profit, color='blue', label='Trend Profit')
    plt.plot(forecasting_profit, color = 'red', label = f'Forcasting Profit {model_config.MODEL_NAME}')
    plt.plot(hybrid_mean_reversion_profit, color='green', label='Hybrid Mean Reversion Profit')
    plt.plot(hybrid_trend_profit, color='brown', label='Hybrid Trend Profit')
    plt.plot(ensemble_profit, color='orange', label='Ensemble Profit')
    plt.title('Profits Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Profit')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/profits_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_profit_per_trade, color='purple', label='Mean Reversion Profit Per Trade')
    plt.plot(trend_profit_per_trade, color='blue', label='Trend Profit Per Trade')
    plt.plot(forecasting_profit_per_trade, color = 'red', label = f'Forcasting Profit Per Trade {model_config.MODEL_NAME}')
    plt.plot(hybrid_mean_reversion_profit_per_trade, color='green', label='Hybrid Mean Reversion Profit Per Trade')
    plt.plot(hybrid_trend_profit_per_trade, color='brown', label='Hybrid Trend Profit Per Trade')
    plt.plot(ensemble_profit_per_trade, color='orange', label='Ensemble Profit Per Trade')
    plt.title('Profit Per Trade Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Profit Per Trade')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/profit_per_trade_fixed.png', dpi=300, bbox_inches='tight')

    plt.clf()

    plt.plot(mean_reversion_sharpe_ratios, color='purple', label='Mean Reversion Sharpe Ratio')
    plt.plot(trend_sharpe_ratios, color='blue', label='Trend Sharpe Ratio')
    plt.plot(forecasting_sharpe_ratios, color='red', label=f'Forecasting Sharpe Ratio {model_config.MODEL_NAME}')
    plt.plot(hybrid_mean_reversion_sharpe_ratios, color='green', label='Hybrid Mean Reversion Sharpe Ratio')
    plt.plot(hybrid_trend_sharpe_ratios, color='brown', label='Hybrid Trend Sharpe Ratio')
    plt.plot(ensemble_sharpe_ratios, color='orange', label='Ensemble Sharpe Ratio')
    plt.title('Sharpe Ratios Using Fixed Postion Size')
    plt.xlabel('Observation')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.savefig(f'{model_config.OUTPUT_DIR}/sharpe_ratios_fixed.png', dpi=300, bbox_inches='tight')

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
    model_config.HOLD_POSITION = args.hold_position
    model_config.USE_FRAC_KELLY = args.use_frac_kelly
    model_config.ENABLE_TRANSACTION_COSTS = args.enable_transaction_costs

    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_config.OUTPUT_DIR = os.path.join(root_dir, args.output_dir)
    os.makedirs(model_config.OUTPUT_DIR, exist_ok=True)

    model_config.MODEL_NAME = args.model_name

    print_model_config(model_config)

    # Run the trading strategy based on the specified model and configuration.
    run_sl_based_trading_strategy(model_config)

def print_model_config(config):
    print("\n🔧 Model Configuration:")
    print(f"  Input Chunk Length        : {config.INPUT_CHUNK_LENGTH}")
    print(f"  Output Chunk Length       : {config.OUTPUT_CHUNK_LENGTH}")
    print(f"  Number of Epochs          : {config.N_EPOCHS}")
    print(f"  Batch Size                : {config.BATCH_SIZE}")
    print(f"  Train Ratio               : {config.TRAIN_RATIO}")
    print(f"  Data File Path            : {config.DATA_FILE_PATH}")
    print(f"  Wallet A Initial Amount   : {config.WALLET_A}")
    print(f"  Wallet B Initial Amount   : {config.WALLET_B}")
    print(f"  Hold Position Enabled     : {config.HOLD_POSITION}")
    print(f"  Fractional Kelly Enabled  : {config.USE_FRAC_KELLY}")
    print(f"  Transaction Costs Enabled : {config.ENABLE_TRANSACTION_COSTS}")
    print(f"  Output Directory          : {config.OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet_a", type=float, default=5487000.0, help="Amount of money in wallet A (currency A).")
    parser.add_argument("--wallet_b", type=float, default=1000000.0, help="Amount of money in wallet B (currency B).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="tcn",
        help="Specify the supervised learning model to use. Supported models include 'bilstm' for Bidirectional LSTM with attention, \
            'nbeats' for NBEATS, 'nhits' for NHiTS, 'transformer' for Transformer, and 'tcn' for Temporal Convolutional Network. \
            Default is 'tcn'."
    )
    parser.add_argument("--input_chunk_length", type=int, default=64, help="Length of the input sequences.")
    parser.add_argument("--output_chunk_length", type=int, default=1, help="Length of the output sequences.")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Ratio of training data used in the train/test split.")
    parser.add_argument("--data_path", type=str, default="", help="Path to the training data. Currency rates should be provided as 1 A / 1 B, where A and B are the respective currencies.", required=True)
    parser.add_argument("--use_frac_kelly", action="store_true", help="Use fractional Kelly to size bets. Default is False.")
    parser.add_argument("--enable_transaction_costs", action="store_true", help="Enable transaction costs. Default is False.")
    parser.add_argument("--hold_position", action="store_true", help="Enable holding position. Default is False.")
    parser.add_argument("--output_dir", type=str, default="results/usd-cny-2023", help="Directory to save all outputs.")

    args = parser.parse_args()
    run(args)