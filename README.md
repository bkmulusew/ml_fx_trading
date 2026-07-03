# Machine Learning Enhanced FX Trading

[![Paper](https://img.shields.io/badge/paper-Mathematics%20(MDPI)-blue)](https://doi.org/10.3390/math14132319)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fmath14132319-orange)](https://doi.org/10.3390/math14132319)
[![Python](https://img.shields.io/badge/python-3.11.8-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Machine learning enhanced foreign exchange trading, built as the reference implementation for the paper **"ForExAI: Time Series Inference and News Article Analysis Reveal Profitable Foreign Exchange Signals"** (Lemeneh et al., *Mathematics* 2026, 14, 2319).

## Why this repo

Most FX trading repos are either a single notebook with a moving average crossover, or a black box with no evaluation. This one is neither. It is a full research grade pipeline: clean and split time series data, train and benchmark six forecasting approaches (classical, neural, and zero shot foundation models), extract trading signals from news articles with LLMs, size every position with an adapted Kelly criterion, and validate every result with permutation tests and bootstrap confidence intervals. You can run it end to end on your own FX data, swap in a new currency pair, or just borrow the Kelly sizing logic for your own strategy.

## Key Results

Results below are from the paper's out of sample test period (September 2024 through July 2025 for USD/CNY and USD/BRL, extended through December 2025 for USD/EUR and USD/GBP), starting from $1,000,000 in capital.

| Currency Pair | Best Model (Active Kelly) | Avg. Cumulative Profit | Sharpe Ratio | Notes |
|---|---|---|---|---|
| USD/BRL | Ensemble | ~$1.07M | 7.66 | ARIMA alone reached ~$2.05M ACP |
| USD/CNY | Ensemble | ~$25.6K | 3.48 | Weak signal, consistent with a managed exchange rate |
| USD/GBP | Mean Reversion | ~$43.1K | 2.28 | Best Sharpe ratios across all four pairs were modest |
| USD/EUR | Ensemble | ~$109.9K | 2.82 | Best in class on both cumulative profit and Sharpe |

A few findings worth knowing before you dive into the code:

- **Zero shot foundation models punch above their weight.** Chronos-Bolt and Toto, used with no task specific fine tuning, achieved the lowest prediction error (RMSE, MASE, MAPE) of any model across all four currency pairs.
- **Short prompts beat expert prompts.** For LLM based news sentiment, simple prompts like "buy, sell, or hold" consistently outperformed long, econometrically informed expert prompts, sometimes by a factor of 100.
- **Transaction costs matter a lot.** At $2.50 per order, fixed position sizing strategies are wiped out across the board due to trade frequency. Kelly based strategies survive better, and on USD/BRL the ARIMA and ensemble strategies remain profitable even after costs.
- **The Kelly criterion helps.** An adapted Kelly criterion, weighted by both historical win rate and predicted trade size, consistently increased cumulative profit relative to fixed bet sizing, though it did not always improve the Sharpe ratio.

See Sections 12 through 18 of the paper for the full breakdown by currency pair, including validation tuning, statistical significance testing, and the effect of trading thresholds.

## Architecture at a glance

```
FX price data + news articles
        |
        v
Data pipeline (clean, scale, split into train / val / test)
        |
        v
Forecasting models: ARIMA, N-BEATS, N-HiTS, TCN, Chronos-Bolt, Toto, Ensemble
        |
        v
Trading strategies: mean reversion, trend, MA crossover, model driven, news sentiment
        |
        v
Position sizing: fixed, Active Kelly, Passive Kelly
        |
        v
Evaluation: cumulative profit, Sharpe ratio, drawdown, skewness, permutation tests
```

## Key Features

- **Data Processing and Management:** A pipeline to clean, scale, and split financial time series data for training and evaluation.
- **Model Training and Evaluation:** Trains Temporal Convolutional Networks (TCN), N-BEATS, and N-HiTS on historical data, and incorporates zero shot forecasters (Toto, Chronos-Bolt) that require no task specific training.
- **Trading Strategy Implementation:** Mean reversion, pure forecasting, hybrid, and news sentiment strategies, each sized with the Kelly criterion.
- **Kelly Criterion for Bet Sizing:** Determines the optimal fraction of the wallet to invest per trade, using both win/loss history and predicted gain.
- **Visualization of Predictions and Profits:** Generates actual vs. predicted price plots alongside profit curves.
- **Simulation of Trading Strategies:** Backtests strategies over a historical period before you'd ever risk real capital.

## Dataset

Before training or evaluating models and running trading simulations, prepare two datasets: one for FX market data and one for news sentiment data.

### FX Dataset

The FX dataset provides historical exchange rate information, formatted as follows:

| date              | bid_price | ask_price | mid_price |
| :---------------- | :-------: | :-------: | :-------: |
| 2/3/2023 16:56    | 6.8004    | 6.8004    | 6.8004    |
| 2/3/2023 16:57    | 6.8038    | 6.8038    | 6.8038    |
| 2/3/2023 16:58    | 6.8036    | 6.8036    | 6.8036    |
| 2/3/2023 16:59    | 6.8050    | 6.8050    | 6.8050    |

- **date**: Timestamp of each FX data point.
- **bid_price**: Price at which the market is willing to buy Currency B using Currency A.
- **ask_price**: Price at which the market is willing to sell Currency B using Currency A.
- **mid_price**: Average of bid and ask prices.

### News Dataset

The News dataset provides sentiment labels for financial news headlines and articles, formatted as follows:

| date              | label_1 | label_2 | label_3 |
| :---------------- | :-----: | :-----: | :-----: |
| 2/3/2023 16:56    | 0       | 1       | 1       |
| 2/3/2023 16:57    | 1       | 0       | 0       |
| 2/3/2023 16:58    | -1      | 1       | 0       |
| 2/3/2023 16:59    | 0       | 1       | 0       |

- **date**: Timestamp of each news data point.
- **label_1, label_2, label_3**: Sentiment signals automatically generated by a Large Language Model (LLM).
  - `0`: Neutral sentiment.
  - `1`: News favorable to Currency B (or less favorable to Currency A).
  - `-1`: News favorable to Currency A (or less favorable to Currency B).

Multiple sentiment columns are provided so you can experiment with different labeling schemes. Select which sentiment source to use with the `--sentiment_source` flag, for example `--sentiment_source label_2`.

## Installation

This project requires Python **3.11.8** and the dependencies listed in `requirements.txt`. We recommend [conda](https://docs.conda.io/en/latest/) for environment management.

### 1. Create a Conda Environment

```bash
conda create -n ml_fx_trading_env python=3.11.8
conda activate ml_fx_trading_env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Example Usage

Once the environment is set up and datasets are prepared, train models, run trading strategies, and generate performance visualizations with:

```bash
python ml_fx_trading/run_trading_strategy.py \
      --wallet_a 1000000 \
      --wallet_b 1000000 \
      --model_name toto \
      --fx_data_path_train /path/to/fx/data/train.csv \
      --fx_data_path_val /path/to/fx/data/val.csv \
      --fx_data_path_test /path/to/fx/data/test.csv \
      --news_data_path_train /path/to/news/data/train.csv \
      --news_data_path_test /path/to/news/data/test.csv \
      --bet_sizing fixed \
      --n_epochs 3 \
      --output_dir results/graphs \
      --allow_news_overlap \
      --news_hold_minutes 30 \
      --sentiment_source fatouros_p2 \
      --seed 0 \
      --input_chunk_length 32 \
      --min_trades_for_full_kelly 10 \
      --kelly_window_days 1 \
      --min_kelly_fraction 0.001 \
      --fast_ma_window 10 \
      --slow_ma_window 30 \
      --threshold 0.0
```

Full list of flags:

```
--wallet_a: Amount of money in wallet A (currency A).
--wallet_b: Amount of money in wallet B (currency B).
--model_name: Specify the model to use. Options: arima, nbeats, nhits, tcn, chronos, or ensemble.
--input_chunk_length: Length of the input sequences.
--output_chunk_length: Length of the output sequences.
--n_epochs: Number of training epochs.
--train_batch_size: Batch size for training.
--eval_batch_size: Batch size for evaluation.
--fx_data_path_train: Path to the fx training data. Currency rates should be provided as 1 A / 1 B, where A and B are the respective currencies.
--fx_data_path_val: Path to the fx validation data. Currency rates should be provided as 1 A / 1 B, where A and B are the respective currencies.
--fx_data_path_test: Path to the fx test data. Currency rates should be provided as 1 A / 1 B, where A and B are the respective currencies.
--news_data_path_train: Path to the news training data.
--news_data_path_test: Path to the news test data.
--bet_sizing: Bet sizing strategy. Options: active_kelly, passive_kelly, or fixed.
--enable_transaction_costs: Enable transaction costs.
--output_dir: Directory to save all outputs.
--news-hold-minutes: Number of minutes to hold a position before allowing exit for news sentiment strategy.
--sentiment_source: Choose which sentiment label column to use for trading.
--allow_news_overlap: Enable overlapping news sentiment trades. When set, multiple news-driven positions may be open at the same time.
--kelly_window_days: Rolling window size in days for Kelly criterion stats.
--min_trades_for_full_kelly: Minimum number of trades required for full Kelly criterion.
--min_kelly_fraction: Minimum Kelly fraction to use.
--threshold: Minimum predicted percentage return required to open a position.
--fast_ma_window: Window size for fast moving average.
--slow_ma_window: Window size for slow moving average.
--seed: Seed for reproducibility.
```

## Citation

If you use this code or build on this work, please cite the paper:

```bibtex
@article{lemeneh2026forexai,
  title   = {ForExAI: Time Series Inference and News Article Analysis Reveal Profitable Foreign Exchange Signals},
  author  = {Lemeneh, Beakal and Hadad, Eli and Ajith, Allen George and Hou, Yanbo and Zha, Charlie and Scarozza, Ganesh and Baannou, Zakaria and Liyeh, Ermiyas and Tomasic, Anthony and Shasha, Dennis},
  journal = {Mathematics},
  volume  = {14},
  number  = {13},
  pages   = {2319},
  year    = {2026},
  publisher = {MDPI},
  doi     = {10.3390/math14132319},
  url     = {https://doi.org/10.3390/math14132319}
}
```

## Disclaimer

Results in the paper are reported without transaction costs unless explicitly stated otherwise. As shown in Section 18, transaction costs materially change the profitability picture. This project is a research tool for studying signal detection in FX markets, not a ready made trading system.
