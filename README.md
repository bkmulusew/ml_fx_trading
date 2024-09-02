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
| Time	                  | A_to_B |
| :---                    | :---:  |
| 2023 01 03 10 09 00.000	| 0.21	 |
| 2023 01 03 10 10 00.000	| 0.22	 |
| 2023 01 03 10 11 00.000	| 0.23	 |
| 2023 01 03 10 12 00.000	| 0.24	 |
| 2023 01 03 10 13 00.000	| 0.25	 |
| 2023 01 03 10 14 00.000	| 0.26	 |

In this dataset:
- Time represents the timestamp of each data point.
- A_to_B represents the exchange rate from Currency A to Currency B.

## Example Usage
```bash
python run_trading_strategy.py --model tcn --data_path /path/to/data --n_epochs 50
```

The full list of flags and options for the python script is as follows:
```
--wallet_a: Amount of money in wallet A (currency A).
--wallet_b: Amount of money in wallet B (currency B).
--model: Specify the supervised learning model to use. Supported models include 'bilstm' for Bidirectional LSTM with attention, 'nbeats' for NBEATS, 'nhits' for NHiTS, 'transformer' for Transformer, and 'tcn' for Temporal Convolutional Network.
--input_chunk_length: Length of the input sequences.
--output_chunk_length: Length of the output sequences.
--n_epochs: Number of training epochs.
--batch_size: Batch size for training.
--train_ratio: Ratio of training data used in the train/test split. 1% of the data is used for validation.
--data_path: Path to the dataset.
--thresholds: Specify a list of threshold values for trading. Provide the values as a comma-separated list of size 4.
            For example, use '--threshold 0,0.00025,0.0005,0.001' to set thresholds at 0, 0.00025, 0.0005, and 0.001.
--frac_kelly: Enable fractional kelly to size bets.
```