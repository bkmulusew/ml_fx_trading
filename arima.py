import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter, pandas2ri
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1) 

if not rpackages.isinstalled('forecast'):
    utils.install_packages('forecast')

forecast = rpackages.importr('forecast')


def backtest_arfima_strategy(prices, train_size=200, initial_capital=10000, position_size=1):
    forecast_pkg = rpackages.importr('forecast')
    forecast_fn = robjects.r['forecast']

    preds = []
    actuals = []
    signals = []  # 1 = long, -1 = short, 0 = hold
    capital = initial_capital
    positions = 0
    equity_curve = []

    for i in range(train_size, len(prices)):
        train = prices.iloc[i-train_size:i]
        test_value = prices.iloc[i]

        # Convert to R object
        with localconverter(default_converter + pandas2ri.converter):
            r_train = pandas2ri.py2rpy(train)

        # Fit ARFIMA
        arfima_model = forecast_pkg.arfima(r_train)

        # Forecast 1 step ahead
        fc = forecast_fn(arfima_model, h=1)
        pred = float(np.array(fc.rx2('mean'))[0])

        preds.append(pred)
        actuals.append(test_value)

        # Trading signal: long if prediction > current, short if prediction < current
        if pred > train.iloc[-1]:
            signal = 1
        elif pred < train.iloc[-1]:
            signal = -1
        else:
            signal = 0

        signals.append(signal)

        # Simulate trade outcome
        daily_return = (test_value - train.iloc[-1]) / train.iloc[-1] * signal
        capital *= (1 + daily_return * position_size)  # position_size = leverage fraction
        equity_curve.append(capital)

    preds = pd.Series(preds, index=prices.index[train_size:])
    actuals = pd.Series(actuals, index=prices.index[train_size:])
    signals = pd.Series(signals, index=prices.index[train_size:])
    equity_curve = pd.Series(equity_curve, index=prices.index[train_size:])

    # Accuracy metrics
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)

    return {
        "predictions": preds,
        "actuals": actuals,
        "signals": signals,
        "equity_curve": equity_curve,
        "rmse": rmse,
        "mae": mae
    }


def plot_acf_pacf_arima010(series, lags=40):
    series = series[~series.index.duplicated(keep='first')]
    diff_series = series.diff().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(diff_series, lags=lags, ax=axes[0])
    axes[0].set_title("ACF of Differenced Series (ARIMA(0,1,0))")
    plot_pacf(diff_series, lags=lags, ax=axes[1])
    axes[1].set_title("PACF of Differenced Series (ARIMA(0,1,0))")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('dataset/usd-cnh-2024-2025.csv', header=0, parse_dates=[0], index_col=0)
    # df = pd.read_csv('dataset/usd-cny-2023.csv', header=0, parse_dates=[0], index_col=0)
    df = df.squeeze('columns')

    mid = df[df.columns[2]]
    mid = mid[~mid.index.duplicated(keep='first')]
    diff_mid = mid.diff().dropna()
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(diff_mid)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    results = backtest_arfima_strategy(mid)

    print(f"RMSE: {results['rmse']:.6f}")
    print(f"MAE: {results['mae']:.6f}")

    # Plot predictions
    plt.figure(figsize=(12,5))
    plt.plot(results['actuals'], label='Actual')
    plt.plot(results['predictions'], label='Predicted', alpha=0.7)
    plt.title("ARFIMA Backtest: Predicted vs Actual")
    plt.legend()
    plt.show()

    # Plot equity curve
    plt.figure(figsize=(12,5))
    plt.plot(results['equity_curve'], label='Equity Curve', color='green')
    plt.title("ARFIMA Trading Strategy Performance")
    plt.legend()
    plt.show()
