# Machine Learning Enhanced FX Trading

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Example Usage](#example-usage)

## Introduction
This project implements a financial trading strategy that combines machine learning models and financial theory to predict currency exchange rates and execute trades. By leveraging different supervised learning models, the project aims to optimize financial decision-making through precise prediction and risk management techniques.

## Key Features
- Data Processing & Management: The project utilizes a comprehensive data processing pipeline to clean, scale, and split financial time series data, making it ready for model training and evaluation.
- Model Training & Evaluation: Supports a variety of supervised learning models, including BiLSTM with Attention, Temporal Convolutional Networks (TCN), NBEATS, NHiTS, and Transformers. These models are trained on historical financial data and evaluated for their prediction accuracy.
- Trading Strategy Implementation: Implements multiple trading strategies, including mean reversion, pure forecasting, and hybrid approaches. Each strategy uses the Kelly criterion for optimal bet sizing, balancing risk and reward to maximize profit.
- Kelly Criterion for Bet Sizing: The project uses the Kelly criterion to determine the optimal fraction of the wallet to invest in each trade, considering both win/loss probabilities and the expected return. Users can choose between full Kelly and fractional Kelly strategies.
- Visualization of Predictions: The project includes tools to visualize actual vs. predicted currency exchange rates, allowing users to visually assess model performance and refine strategies.
- Customizable Trading Thresholds: Users can set specific thresholds to control trade sensitivity, determining when trades should be executed based on changes in predicted currency ratios.
- Simulation of Trading Strategies: The project simulates trading over a historical period, allowing users to test and refine their strategies before applying them to real-world trading scenarios.

## Dataset
Before training or evaluating the model and running the trading simulation with different pairs trading strategies, it is necessary to have a dataset. The dataset should be in the following format:
| date	            | bid_price | ask_price | mid_price | with_prompt | without_prompt |
| :---                  | :---:  | :---:  | :---:  | :---:  | :---:  |
| 2/3/2023 16:56	| 6.8004     | 6.8004     | 6.8004     | 0            | 1     |
| 2/3/2023 16:57	| 6.8038	 | 6.8038	  | 6.8038	   | 0            | 0     |
| 2/3/2023 16:58	| 6.8036	 | 6.8036	  | 6.8036	   | 1            | 0     |
| 2/3/2023 16:59	| 6.805	 | 6.805	  | 6.805	   | 0            | 0	  |

In this dataset:
- 'date' represents the timestamp of each data point.
- 'bid_price' represents the bid price from Currency A to Currency B.
- 'ask_price' represents the ask price from Currency A to Currency B.
- 'mid_price' represents the mid price (average of bid and ask prices) from Currency A to Currency B.
- 'with_prompt' represents the LLM's prediction for Currency A direction after being given an article with prompt engineering. Values: 1 (up), 0 (neutral), -1 (down).
- 'without_prompt' represents the LLM's prediction for Currency A direction after being given an article without prompt engineering. Values: 1 (up), 0 (neutral), -1 (down).

## Example Usage
```bash
python ml_fx_trading/run_trading_strategy.py \
    --model_name toto \
    --data_path_train /path/to/train/data.csv \
    --data_path_val /path/to/val/data.csv \
    --data_path_test /path/to/test/data.csv \
    --n_epochs 1 \
    --output_dir results \
    --use_frac_kelly
```

The full list of flags and options for the python script is as follows:
```
--wallet_a: Amount of money in wallet A (currency A).
--wallet_b: Amount of money in wallet B (currency B).
--model_name: Specify the supervised learning model to use. Supported models include 'bilstm' for Bidirectional LSTM with attention, 'nbeats' for NBEATS, 'nhits' for NHiTS, 'transformer' for Transformer, and 'tcn' for Temporal Convolutional Network.
--input_chunk_length: Length of the input sequences.
--output_chunk_length: Length of the output sequences.
--n_epochs: Number of training epochs.
--batch_size: Batch size for training.
--data_path_train: Path to the training data.
--data_path_val: Path to the validation data.
--data_path_test: Path to the test data.
--use_frac_kelly: Enable fractional kelly to size bets.
--enable_transaction_costs: Enable transaction costs.
--hold_position: Enable holding position.
--output_dir: Directory to save all outputs.
```
