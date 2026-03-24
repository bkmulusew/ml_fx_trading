import pytest
import numpy as np
from datetime import datetime, timedelta
from strategies import TradingStrategy

pytestmark = pytest.mark.unit

STRATEGIES = ["mean_reversion", "trend", "model_driven", "news_sentiment", "ensemble"]


def _populate_kelly_data(ts, strategy, n_wins, n_losses, total_gain, total_loss,
                         n_potential_wins=0, n_potential_losses=0,
                         total_estimated_gain=0.0, n_estimated_gains=0, day=0):
    """Populate windowed Kelly data for testing."""
    avg_win = total_gain / n_wins if n_wins else 0
    avg_loss = total_loss / n_losses if n_losses else 0
    for _ in range(n_wins):
        ts._kelly_actual_outcomes[strategy].append((day, avg_win))
    for _ in range(n_losses):
        ts._kelly_actual_outcomes[strategy].append((day, -avg_loss))
    for _ in range(n_potential_wins):
        ts._kelly_potential_outcomes[strategy].append((day, True))
    for _ in range(n_potential_losses):
        ts._kelly_potential_outcomes[strategy].append((day, False))
    avg_est = total_estimated_gain / n_estimated_gains if n_estimated_gains else 0
    for _ in range(n_estimated_gains):
        ts._kelly_estimated_gains[strategy].append((day, avg_est))

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestInitialization:
    def test_initial_wallets(self, trading_strategy_instance):
        ts = trading_strategy_instance
        for s in STRATEGIES:
            assert ts.wallet_a[s] == 10000
            assert ts.wallet_b[s] == 10000

    def test_initial_pnl_empty(self, trading_strategy_instance):
        for s in STRATEGIES:
            assert trading_strategy_instance.pnl[s] == []

    def test_initial_counters_zero(self, trading_strategy_instance):
        ts = trading_strategy_instance
        for s in STRATEGIES:
            assert ts.num_trades[s] == 0

    def test_initial_kelly_rolling_window_empty(self, trading_strategy_instance):
        ts = trading_strategy_instance
        assert ts._kelly_day_index == 0
        assert ts.kelly_window_days is None
        for s in STRATEGIES:
            assert ts._kelly_actual_outcomes[s] == []
            assert ts._kelly_potential_outcomes[s] == []
            assert ts._kelly_estimated_gains[s] == []

    def test_initial_positions_none(self, trading_strategy_instance):
        for s in STRATEGIES:
            assert trading_strategy_instance.single_slot_positions[s]["type"] is None

# ---------------------------------------------------------------------------
# _get_prices
# ---------------------------------------------------------------------------
class TestGetPrices:
    def test_get_prices_no_transaction_costs(self, trading_strategy_instance):
        buy, sell = trading_strategy_instance._get_prices(bid_price=1.25, ask_price=1.27)
        mid = (1.25 + 1.27) / 2
        assert buy == pytest.approx(mid)
        assert sell == pytest.approx(mid)

    def test_get_prices_with_transaction_costs(self, trading_strategy_with_costs):
        buy, sell = trading_strategy_with_costs._get_prices(bid_price=1.25, ask_price=1.27)
        assert buy == pytest.approx(1.27)   # buy at ask
        assert sell == pytest.approx(1.25)  # sell at bid

# ---------------------------------------------------------------------------
# determine_trade_direction
# ---------------------------------------------------------------------------
class TestDetermineTradeDirection:
    def test_mean_reversion_negative_base(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_trade_direction("mean_reversion", -0.01, 0.0)
        assert d == "buy_currency_a"

    def test_mean_reversion_positive_base(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_trade_direction("mean_reversion", 0.01, 0.0)
        assert d == "sell_currency_a"

    def test_mean_reversion_zero_base(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_trade_direction("mean_reversion", 0.0, 0.0)
        assert d == "no_trade"

    def test_trend_negative_base(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_trade_direction("trend", -0.01, 0.0)
        assert d == "sell_currency_a"

    def test_trend_positive_base(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_trade_direction("trend", 0.01, 0.0)
        assert d == "buy_currency_a"

    def test_model_driven_uses_pred(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_trade_direction("model_driven", 0.01, -0.02)
        assert d == "sell_currency_a"

    def test_model_driven_negative_pred(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_trade_direction("model_driven", 0.0, -0.01)
        assert d == "sell_currency_a"

    def test_model_driven_positive_pred(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_trade_direction("model_driven", 0.0, 0.01)
        assert d == "buy_currency_a"

# ---------------------------------------------------------------------------
# determine_news_sentiment_trade_direction
# ---------------------------------------------------------------------------
class TestNewsSentimentDirection:
    def test_news_sentiment_negative(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_news_sentiment_trade_direction(-1)
        assert d == "buy_currency_a"

    def test_news_sentiment_positive(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_news_sentiment_trade_direction(1)
        assert d == "sell_currency_a"

    def test_news_sentiment_neutral(self, trading_strategy_instance):
        d = trading_strategy_instance.determine_news_sentiment_trade_direction(0)
        assert d == "no_trade"

# ---------------------------------------------------------------------------
# kelly_criterion
# ---------------------------------------------------------------------------
class TestKellyCriterion:
    def test_kelly_below_min_trades_returns_min_fraction(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False)
        s = "trend"
        _populate_kelly_data(ts, s, n_wins=5, n_losses=3,
                             total_gain=5.0, total_loss=3.0,
                             total_estimated_gain=4.0, n_estimated_gains=8)
        f = ts.kelly_criterion(s, 0.01)
        assert f == ts.min_kelly_fraction

    def test_kelly_active_uses_actual_wins(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False)
        s = "trend"
        _populate_kelly_data(ts, s, n_wins=80, n_losses=50,
                             total_gain=100.0, total_loss=60.0,
                             total_estimated_gain=50.0, n_estimated_gains=130)
        f = ts.kelly_criterion(s, 0.5)
        assert isinstance(f, float)
        p = 80 / 130
        q = 1 - p
        avg_gain = 100.0 / 80
        avg_loss = 60.0 / 50
        wl_ratio = avg_gain / avg_loss
        h = abs(0.5) / (50.0 / 130)
        expected = max(p - (q / (h * wl_ratio)), 0.0)
        assert expected > 0.0
        assert f == pytest.approx(expected)

    def test_kelly_passive_uses_potential_wins(self):
        ts = TradingStrategy(10000, 10000, 3, "passive_kelly", False)
        s = "model_driven"
        _populate_kelly_data(ts, s, n_wins=80, n_losses=50,
                             total_gain=100.0, total_loss=60.0,
                             n_potential_wins=90, n_potential_losses=40,
                             total_estimated_gain=50.0, n_estimated_gains=130)
        f = ts.kelly_criterion(s, 0.5)
        p = 90 / 130
        q = 1 - p
        avg_gain = 100.0 / 80
        avg_loss = 60.0 / 50
        wl_ratio = avg_gain / avg_loss
        h = abs(0.5) / (50.0 / 130)
        expected = max(p - (q / (h * wl_ratio)), 0.0)
        assert expected > 0.0
        assert f == pytest.approx(expected)

    def test_kelly_negative_fraction_clamped_to_zero(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False)
        s = "trend"
        _populate_kelly_data(ts, s, n_wins=80, n_losses=50,
                             total_gain=100.0, total_loss=60.0,
                             total_estimated_gain=50.0, n_estimated_gains=130)
        f = ts.kelly_criterion(s, 0.01)
        assert f == 0.0

    def test_kelly_zero_estimated_gain(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False)
        s = "trend"
        _populate_kelly_data(ts, s, n_wins=80, n_losses=50,
                             total_gain=100.0, total_loss=60.0,
                             total_estimated_gain=50.0, n_estimated_gains=130)
        assert ts.kelly_criterion(s, 0.0) == 0.0

    def test_kelly_passive_zero_potential_trades(self):
        ts = TradingStrategy(10000, 10000, 3, "passive_kelly", False)
        s = "trend"
        _populate_kelly_data(ts, s, n_wins=80, n_losses=50,
                             total_gain=100.0, total_loss=60.0,
                             n_potential_wins=0, n_potential_losses=0,
                             total_estimated_gain=50.0, n_estimated_gains=130)
        f = ts.kelly_criterion(s, 0.5)
        assert f == ts.min_kelly_fraction

# ---------------------------------------------------------------------------
# execute_trade & Position Management
# ---------------------------------------------------------------------------
class TestExecuteTrade:
    def _make_ts(self):
        return TradingStrategy(10000, 10000, 3, "fixed", False)

    def test_execute_trade_buy_updates_wallets(self):
        ts = self._make_ts()
        t = datetime(2023, 6, 1, 10, 0)
        ts.execute_trade("mean_reversion", t, "buy_currency_a", 1.25, 1.27, estimated_gain=0.01)
        assert ts.wallet_a["mean_reversion"] > 10000
        assert ts.wallet_b["mean_reversion"] < 10000

    def test_execute_trade_sell_updates_wallets(self):
        ts = self._make_ts()
        t = datetime(2023, 6, 1, 10, 0)
        ts.execute_trade("mean_reversion", t, "sell_currency_a", 1.25, 1.27, estimated_gain=0.01)
        assert ts.wallet_a["mean_reversion"] < 10000
        assert ts.wallet_b["mean_reversion"] > 10000

    def test_execute_trade_no_trade_appends_zero_pnl(self):
        ts = self._make_ts()
        t = datetime(2023, 6, 1, 10, 0)
        ts.execute_trade("trend", t, "no_trade", 1.25, 1.27, estimated_gain=0.0)
        assert ts.pnl["trend"] == [0.0]

    def test_execute_trade_opens_position(self):
        ts = self._make_ts()
        t = datetime(2023, 6, 1, 10, 0)
        ts.execute_trade("trend", t, "buy_currency_a", 1.25, 1.27, estimated_gain=0.01)
        assert ts.single_slot_positions["trend"]["type"] == "long"

    def test_execute_trade_closes_before_opening(self):
        ts = self._make_ts()
        t1 = datetime(2023, 6, 1, 10, 0)
        t2 = datetime(2023, 6, 1, 10, 1)
        ts.execute_trade("trend", t1, "buy_currency_a", 1.25, 1.27, estimated_gain=0.01)
        ts.execute_trade("trend", t2, "sell_currency_a", 1.26, 1.28, estimated_gain=0.01)
        # Position from first trade closed; new one opened
        assert ts.num_trades["trend"] == 1  # one close from the first position
        assert ts.single_slot_positions["trend"]["type"] == "short"

    def test_execute_trade_respects_hold_minutes(self):
        ts = self._make_ts()
        t1 = datetime(2023, 6, 1, 10, 0)
        t2 = datetime(2023, 6, 1, 10, 1)  # only 1 min later (< 3 min hold)
        ts.execute_trade("news_sentiment", t1, "buy_currency_a", 1.25, 1.27)
        ts.execute_trade("news_sentiment", t2, "sell_currency_a", 1.26, 1.28)
        # Should not have closed the first position because hold_minutes=3
        assert ts.single_slot_positions["news_sentiment"]["type"] == "long"

    def test_execute_trade_insufficient_funds_skipped(self):
        ts = self._make_ts()
        ts.wallet_b["mean_reversion"] = 0.0
        t = datetime(2023, 6, 1, 10, 0)
        ts.execute_trade("mean_reversion", t, "buy_currency_a", 1.25, 1.27, estimated_gain=0.01)
        assert ts.single_slot_positions["mean_reversion"]["type"] is None

    def test_execute_trade_records_kelly_estimated_gain(self):
        ts = self._make_ts()
        t = datetime(2023, 6, 1, 10, 0)
        ts.execute_trade("trend", t, "buy_currency_a", 1.25, 1.27, estimated_gain=0.01)
        assert len(ts._kelly_estimated_gains["trend"]) == 1
        day_idx, est = ts._kelly_estimated_gains["trend"][0]
        assert day_idx == 0
        assert est == pytest.approx(0.01)

    def test_execute_trade_news_no_kelly_estimated_gain(self):
        ts = self._make_ts()
        t = datetime(2023, 6, 1, 10, 0)
        ts.execute_trade("news_sentiment", t, "buy_currency_a", 1.25, 1.27)
        assert len(ts._kelly_estimated_gains["news_sentiment"]) == 0

# ---------------------------------------------------------------------------
# _close_single_position
# ---------------------------------------------------------------------------
class TestCloseSinglePosition:
    def _open_long(self, ts, strategy, entry_price=1.26):
        ts.single_slot_positions[strategy] = {
            "type": "long",
            "size_a": 1000,
            "size_b": 1000 * entry_price,
            "entry_ratio": entry_price,
            "entry_timestamp": datetime(2023, 6, 1, 10, 0),
            "hold_minutes": -1,
        }

    def _open_short(self, ts, strategy, entry_price=1.26):
        ts.single_slot_positions[strategy] = {
            "type": "short",
            "size_a": 1000,
            "size_b": 1000 * entry_price,
            "entry_ratio": entry_price,
            "entry_timestamp": datetime(2023, 6, 1, 10, 0),
            "hold_minutes": -1,
        }

    def test_close_long_position_profit(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_long(ts, "trend", 1.26)
        pos = ts.single_slot_positions["trend"]
        ts._close_single_position("trend", pos, sell_price=1.28, buy_price=1.28)
        assert len(ts._kelly_actual_outcomes["trend"]) == 1
        assert ts._kelly_actual_outcomes["trend"][0][1] > 0
        assert ts.pnl["trend"][-1] > 0

    def test_close_long_position_loss(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_long(ts, "trend", 1.26)
        pos = ts.single_slot_positions["trend"]
        ts._close_single_position("trend", pos, sell_price=1.24, buy_price=1.24)
        assert len(ts._kelly_actual_outcomes["trend"]) == 1
        assert ts._kelly_actual_outcomes["trend"][0][1] < 0
        assert ts.pnl["trend"][-1] < 0

    def test_close_short_position_profit(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_short(ts, "model_driven", 1.26)
        pos = ts.single_slot_positions["model_driven"]
        ts._close_single_position("model_driven", pos, sell_price=1.24, buy_price=1.24)
        assert len(ts._kelly_actual_outcomes["model_driven"]) == 1
        assert ts._kelly_actual_outcomes["model_driven"][0][1] > 0
        assert ts.pnl["model_driven"][-1] > 0

    def test_close_short_position_loss(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_short(ts, "model_driven", 1.26)
        pos = ts.single_slot_positions["model_driven"]
        ts._close_single_position("model_driven", pos, sell_price=1.28, buy_price=1.28)
        assert len(ts._kelly_actual_outcomes["model_driven"]) == 1
        assert ts._kelly_actual_outcomes["model_driven"][0][1] < 0
        assert ts.pnl["model_driven"][-1] < 0

    def test_close_long_records_kelly_actual_outcome(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_long(ts, "trend", 1.26)
        pos = ts.single_slot_positions["trend"]
        ts._close_single_position("trend", pos, sell_price=1.28, buy_price=1.28)
        assert len(ts._kelly_actual_outcomes["trend"]) == 1
        day_idx, profit = ts._kelly_actual_outcomes["trend"][0]
        assert day_idx == 0
        assert profit > 0

    def test_close_short_records_kelly_actual_outcome(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_short(ts, "model_driven", 1.26)
        pos = ts.single_slot_positions["model_driven"]
        ts._close_single_position("model_driven", pos, sell_price=1.24, buy_price=1.24)
        assert len(ts._kelly_actual_outcomes["model_driven"]) == 1
        day_idx, profit = ts._kelly_actual_outcomes["model_driven"][0]
        assert day_idx == 0
        assert profit > 0

    def test_close_zero_profit_not_recorded_in_kelly(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_long(ts, "trend", 1.26)
        pos = ts.single_slot_positions["trend"]
        ts._close_single_position("trend", pos, sell_price=1.26, buy_price=1.26)
        assert len(ts._kelly_actual_outcomes["trend"]) == 0

# ---------------------------------------------------------------------------
# _track_potential_outcome
# ---------------------------------------------------------------------------
class TestTrackPotentialOutcome:
    def test_potential_win_buy_positive_change(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "buy_currency_a", 0.01)
        assert len(ts._kelly_potential_outcomes["trend"]) == 1
        assert ts._kelly_potential_outcomes["trend"][0][1] is True

    def test_potential_loss_buy_negative_change(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "buy_currency_a", -0.01)
        assert len(ts._kelly_potential_outcomes["trend"]) == 1
        assert ts._kelly_potential_outcomes["trend"][0][1] is False

    def test_potential_no_trade_no_change(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "no_trade", 0.05)
        assert len(ts._kelly_potential_outcomes["trend"]) == 0

    def test_potential_zero_change_no_count(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "buy_currency_a", 0.0)
        assert len(ts._kelly_potential_outcomes["trend"]) == 0

    def test_potential_win_records_kelly_outcome(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "buy_currency_a", 0.01)
        assert len(ts._kelly_potential_outcomes["trend"]) == 1
        day_idx, is_win = ts._kelly_potential_outcomes["trend"][0]
        assert day_idx == 0
        assert is_win is True

    def test_potential_loss_records_kelly_outcome(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "buy_currency_a", -0.01)
        assert len(ts._kelly_potential_outcomes["trend"]) == 1
        day_idx, is_win = ts._kelly_potential_outcomes["trend"][0]
        assert day_idx == 0
        assert is_win is False

# ---------------------------------------------------------------------------
# News Overlap
# ---------------------------------------------------------------------------
class TestNewsOverlap:
    def test_news_overlap_enabled_opens_multiple(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False, allow_news_overlap=True)
        t1 = datetime(2023, 6, 1, 10, 0)
        t2 = datetime(2023, 6, 1, 10, 0, 30)
        ts.execute_trade("news_sentiment", t1, "buy_currency_a", 1.25, 1.27)
        ts.execute_trade("news_sentiment", t2, "sell_currency_a", 1.26, 1.28)
        assert len(ts.multi_slot_positions["news_sentiment"]) == 2

    def test_news_overlap_disabled_single_slot(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False, allow_news_overlap=False)
        t1 = datetime(2023, 6, 1, 10, 0)
        t2 = datetime(2023, 6, 1, 10, 4)  # after hold
        ts.execute_trade("news_sentiment", t1, "buy_currency_a", 1.25, 1.27)
        ts.execute_trade("news_sentiment", t2, "sell_currency_a", 1.26, 1.28)
        assert ts.multi_slot_positions["news_sentiment"] == []

    def test_close_all_remaining_clears_multi_slot(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False, allow_news_overlap=True)
        t = datetime(2023, 6, 1, 10, 0)
        ts.execute_trade("news_sentiment", t, "buy_currency_a", 1.25, 1.27)
        ts.execute_trade("news_sentiment", t, "sell_currency_a", 1.26, 1.28)
        assert len(ts.multi_slot_positions["news_sentiment"]) > 0

        ts._close_all_remaining_positions(["news_sentiment"], 1.26, 1.28)
        assert len(ts.multi_slot_positions["news_sentiment"]) == 0

# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
class TestEnsemble:
    def test_train_ensemble_model_empty_data(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        ts.train_ensemble_model([], seed=42)
        assert ts.ensemble_model_trained is False

    def test_train_ensemble_model_with_data(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False, optimize_ensemble=False)
        data = [([0.01, 0.02, 0.03], 0.005) for _ in range(50)]
        ts.train_ensemble_model(data, seed=42)
        assert ts.ensemble_model_trained is True

    def test_generate_training_data_format(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        actual = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
        preds = {"m1": [1.0, 1.005, 1.015, 1.025, 1.035, 1.045],
                 "m2": [1.0, 1.002, 1.012, 1.022, 1.032, 1.042]}
        result = ts._generate_training_data(actual, preds)
        assert isinstance(result, list)
        for x, y in result:
            assert isinstance(x, list)
            assert isinstance(y, float)

    def test_generate_training_data_length(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        actual = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
        preds = {"m1": [1.0]*6, "m2": [1.0]*6}
        result = ts._generate_training_data(actual, preds)
        assert len(result) == len(actual) - 3

    def test_ensemble_no_trade_when_untrained(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        assert ts.ensemble_model_trained is False
        n = 10
        base_ts = datetime(2023, 6, 1, 10, 0)
        fx_timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
        actual_rates = [1.25 + 0.0001 * i for i in range(n)]
        pred_rates = {"m1": [1.25 + 0.00015 * i for i in range(n)],
                      "m2": [1.25 + 0.00012 * i for i in range(n)]}
        bid_prices = [r - 0.0001 for r in actual_rates]
        ask_prices = [r + 0.0001 for r in actual_rates]

        ts._execute_ensemble_strategy(fx_timestamps, actual_rates, pred_rates, bid_prices, ask_prices)
        assert ts.num_trades["ensemble"] == 0

# ---------------------------------------------------------------------------
# simulate_trading_with_strategies
# ---------------------------------------------------------------------------
class TestSimulateTradingWithStrategies:
    def _make_market_data(self, n=60, base_time=None):
        """Generate data within market hours 09:00-17:00."""
        if base_time is None:
            base_time = datetime(2023, 6, 1, 9, 30)
        fx_timestamps = [base_time + timedelta(minutes=i) for i in range(n)]
        mid = 1.25 + np.cumsum(np.random.default_rng(42).normal(0, 0.0001, n))
        actual_rates = mid.tolist()
        pred_rates = (mid + 0.0001).tolist()
        bid_prices = (mid - 0.0001).tolist()
        ask_prices = (mid + 0.0001).tolist()
        news_ts = [base_time + timedelta(minutes=5*i) for i in range(10)]
        news_sentiments = [1, -1, 0, 1, -1, 1, 0, -1, 1, -1]
        return fx_timestamps, actual_rates, pred_rates, bid_prices, ask_prices, news_ts, news_sentiments

    def test_simulate_filters_market_hours(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        # Generate timestamps outside market hours (before 9am)
        base = datetime(2023, 6, 1, 6, 0)
        n = 20
        fx_timestamps = [base + timedelta(minutes=i) for i in range(n)]
        actual_rates = [1.25] * n
        pred_rates = [1.25] * n
        bid_prices = [1.2499] * n
        ask_prices = [1.2501] * n
        news_ts = [base + timedelta(minutes=5)]
        news_sentiments = [1]

        ts.simulate_trading_with_strategies(
            fx_timestamps, actual_rates, pred_rates,
            bid_prices, ask_prices, news_ts, news_sentiments,
        )
        for s in ["mean_reversion", "trend", "model_driven"]:
            assert ts.num_trades[s] == 0

    def test_simulate_runs_all_base_strategies(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        (fx_ts, actual, pred, bid, ask, news_ts, news_sent) = self._make_market_data(n=60)
        ts.simulate_trading_with_strategies(fx_ts, actual, pred, bid, ask, news_ts, news_sent)
        for s in ["mean_reversion", "trend", "model_driven"]:
            assert ts.num_trades[s] > 0

    def test_simulate_closes_remaining_at_end(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        (fx_ts, actual, pred, bid, ask, news_ts, news_sent) = self._make_market_data(n=60)
        ts.simulate_trading_with_strategies(fx_ts, actual, pred, bid, ask, news_ts, news_sent)
        for s in ["mean_reversion", "trend", "model_driven"]:
            assert ts.single_slot_positions[s]["type"] is None

    def test_simulate_news_disabled_if_no_news(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        (fx_ts, actual, pred, bid, ask, _, _) = self._make_market_data(n=60)
        # Only one news timestamp in market hours
        ts.simulate_trading_with_strategies(
            fx_ts, actual, pred, bid, ask,
            [datetime(2023, 6, 1, 10, 0)], [1],
        )
        assert ts.num_trades["news_sentiment"] == 0


# ---------------------------------------------------------------------------
# Kelly Rolling Window
# ---------------------------------------------------------------------------
class TestKellyRollingWindow:
    def test_advance_kelly_day_increments_index(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False)
        assert ts._kelly_day_index == 0
        ts.advance_kelly_day()
        assert ts._kelly_day_index == 1
        ts.advance_kelly_day()
        assert ts._kelly_day_index == 2

    def test_prune_removes_old_records(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False, kelly_window_days=3)
        s = "trend"
        for day in range(3):
            ts._kelly_actual_outcomes[s].append((day, 1.0))
            ts._kelly_potential_outcomes[s].append((day, True))
            ts._kelly_estimated_gains[s].append((day, 0.5))

        assert len(ts._kelly_actual_outcomes[s]) == 3
        # Advance to day 3 → cutoff = 3 - 3 = 0, keep day > 0 → days 1, 2
        ts._kelly_day_index = 3
        ts._prune_kelly_history()
        assert len(ts._kelly_actual_outcomes[s]) == 2
        assert len(ts._kelly_potential_outcomes[s]) == 2
        assert len(ts._kelly_estimated_gains[s]) == 2
        assert all(d > 0 for d, _ in ts._kelly_actual_outcomes[s])

    def test_prune_no_op_when_window_none(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False, kelly_window_days=None)
        s = "trend"
        for day in range(10):
            ts._kelly_actual_outcomes[s].append((day, 1.0))
        ts._kelly_day_index = 100
        ts._prune_kelly_history()
        assert len(ts._kelly_actual_outcomes[s]) == 10

    def test_kelly_window_uses_only_recent_days(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False, kelly_window_days=2)
        s = "trend"
        # Day 0: 60 wins, 60 losses
        _populate_kelly_data(ts, s, n_wins=60, n_losses=60,
                             total_gain=60.0, total_loss=60.0,
                             total_estimated_gain=50.0, n_estimated_gains=120, day=0)
        # Day 1: 80 wins, 40 losses
        _populate_kelly_data(ts, s, n_wins=80, n_losses=40,
                             total_gain=100.0, total_loss=40.0,
                             total_estimated_gain=50.0, n_estimated_gains=120, day=1)
        # Advance to day 2 → cutoff = 2 - 2 = 0, prune day 0
        ts._kelly_day_index = 2
        ts._prune_kelly_history()

        wins, losses, tg, tl, _, _, _, _ = ts._windowed_kelly_stats(s)
        assert wins == 80
        assert losses == 40
        assert tg == pytest.approx(100.0)
        assert tl == pytest.approx(40.0)

    def test_kelly_window_none_uses_all_history(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False, kelly_window_days=None)
        s = "trend"
        for day in range(20):
            _populate_kelly_data(ts, s, n_wins=5, n_losses=3,
                                 total_gain=5.0, total_loss=3.0,
                                 total_estimated_gain=4.0, n_estimated_gains=8, day=day)
            ts.advance_kelly_day()

        wins, losses, _, _, _, _, _, _ = ts._windowed_kelly_stats(s)
        assert wins == 100   # 20 * 5
        assert losses == 60  # 20 * 3

    def test_kelly_adapts_after_day_advance(self):
        """Kelly fraction changes as old poor-performing days fall out of the window."""
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False, kelly_window_days=2)
        s = "trend"

        # Day 0: terrible performance (30 wins, 90 losses) → low p
        _populate_kelly_data(ts, s, n_wins=30, n_losses=90,
                             total_gain=30.0, total_loss=90.0,
                             total_estimated_gain=50.0, n_estimated_gains=120, day=0)

        f_day0 = ts.kelly_criterion(s, 0.5)

        # Day 1: great performance (100 wins, 20 losses) → high p
        _populate_kelly_data(ts, s, n_wins=100, n_losses=20,
                             total_gain=100.0, total_loss=20.0,
                             total_estimated_gain=50.0, n_estimated_gains=120, day=1)

        # Advance to day 2 → prune day 0
        ts._kelly_day_index = 2
        ts._prune_kelly_history()

        f_day1_only = ts.kelly_criterion(s, 0.5)

        # After pruning the bad day 0, the Kelly fraction should be higher
        assert f_day1_only > f_day0

    def test_kelly_window_days_default_is_none(self, trading_strategy_instance):
        assert trading_strategy_instance.kelly_window_days is None

    def test_kelly_window_days_configurable(self, trading_strategy_kelly_window):
        assert trading_strategy_kelly_window.kelly_window_days == 5

    def test_windowed_kelly_stats_empty(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False)
        s = "trend"
        wins, losses, tg, tl, pw, pl, egs, egc = ts._windowed_kelly_stats(s)
        assert wins == 0
        assert losses == 0
        assert tg == 0.0
        assert tl == 0.0
        assert pw == 0
        assert pl == 0
        assert egs == 0.0
        assert egc == 0

    def test_kelly_returns_min_fraction_when_no_estimated_gains(self):
        """After window prune removes all estimated gains, Kelly falls back to min fraction."""
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False, kelly_window_days=1)
        s = "trend"
        _populate_kelly_data(ts, s, n_wins=80, n_losses=50,
                             total_gain=100.0, total_loss=60.0,
                             total_estimated_gain=50.0, n_estimated_gains=130, day=0)
        # Advance past the window so all data is pruned
        ts._kelly_day_index = 2
        ts._prune_kelly_history()
        f = ts.kelly_criterion(s, 0.5)
        assert f == ts.min_kelly_fraction