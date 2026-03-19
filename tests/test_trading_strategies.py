import pytest
import numpy as np
from datetime import datetime, timedelta
from strategies import TradingStrategy

pytestmark = pytest.mark.unit

STRATEGIES = ["mean_reversion", "trend", "model_driven", "news_sentiment", "ensemble"]

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
            assert ts.num_actual_wins[s] == 0
            assert ts.num_actual_losses[s] == 0
            assert ts.num_potential_wins[s] == 0
            assert ts.num_potential_losses[s] == 0

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
    def test_kelly_below_min_trades_returns_min_fraction(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts.num_actual_wins["trend"] = 5
        ts.num_actual_losses["trend"] = 3
        f = ts.kelly_criterion("trend", 0.01)
        assert f == ts.min_kelly_fraction

    def test_kelly_active_uses_actual_wins(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False)
        s = "trend"
        ts.num_actual_wins[s] = 80
        ts.num_actual_losses[s] = 50
        ts.total_gain[s] = 100.0
        ts.total_loss[s] = 60.0
        ts.total_estimated_gain[s] = 50.0
        ts.num_estimated_gains[s] = 130
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
        ts.num_actual_wins[s] = 80
        ts.num_actual_losses[s] = 50
        ts.num_potential_wins[s] = 90
        ts.num_potential_losses[s] = 40
        ts.total_gain[s] = 100.0
        ts.total_loss[s] = 60.0
        ts.total_estimated_gain[s] = 50.0
        ts.num_estimated_gains[s] = 130
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
        ts.num_actual_wins[s] = 80
        ts.num_actual_losses[s] = 50
        ts.total_gain[s] = 100.0
        ts.total_loss[s] = 60.0
        ts.total_estimated_gain[s] = 50.0
        ts.num_estimated_gains[s] = 130
        # Tiny estimated_gain relative to average produces negative raw fraction
        f = ts.kelly_criterion(s, 0.01)
        assert f == 0.0

    def test_kelly_zero_estimated_gain(self):
        ts = TradingStrategy(10000, 10000, 3, "active_kelly", False)
        s = "trend"
        ts.num_actual_wins[s] = 80
        ts.num_actual_losses[s] = 50
        ts.total_gain[s] = 100.0
        ts.total_loss[s] = 60.0
        ts.total_estimated_gain[s] = 50.0
        ts.num_estimated_gains[s] = 130
        assert ts.kelly_criterion(s, 0.0) == 0.0

    def test_kelly_passive_zero_potential_trades(self):
        ts = TradingStrategy(10000, 10000, 3, "passive_kelly", False)
        s = "trend"
        ts.num_actual_wins[s] = 80
        ts.num_actual_losses[s] = 50
        ts.num_potential_wins[s] = 0
        ts.num_potential_losses[s] = 0
        ts.total_gain[s] = 100.0
        ts.total_loss[s] = 60.0
        ts.total_estimated_gain[s] = 50.0
        ts.num_estimated_gains[s] = 130
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
        assert ts.num_actual_wins["trend"] == 1
        assert ts.pnl["trend"][-1] > 0

    def test_close_long_position_loss(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_long(ts, "trend", 1.26)
        pos = ts.single_slot_positions["trend"]
        ts._close_single_position("trend", pos, sell_price=1.24, buy_price=1.24)
        assert ts.num_actual_losses["trend"] == 1
        assert ts.pnl["trend"][-1] < 0

    def test_close_short_position_profit(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_short(ts, "model_driven", 1.26)
        pos = ts.single_slot_positions["model_driven"]
        ts._close_single_position("model_driven", pos, sell_price=1.24, buy_price=1.24)
        assert ts.num_actual_wins["model_driven"] == 1
        assert ts.pnl["model_driven"][-1] > 0

    def test_close_short_position_loss(self):
        ts = TradingStrategy(10000, 10000, 3, "fixed", False)
        self._open_short(ts, "model_driven", 1.26)
        pos = ts.single_slot_positions["model_driven"]
        ts._close_single_position("model_driven", pos, sell_price=1.28, buy_price=1.28)
        assert ts.num_actual_losses["model_driven"] == 1
        assert ts.pnl["model_driven"][-1] < 0

# ---------------------------------------------------------------------------
# _track_potential_outcome
# ---------------------------------------------------------------------------
class TestTrackPotentialOutcome:
    def test_potential_win_buy_positive_change(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "buy_currency_a", 0.01)
        assert ts.num_potential_wins["trend"] == 1

    def test_potential_loss_buy_negative_change(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "buy_currency_a", -0.01)
        assert ts.num_potential_losses["trend"] == 1

    def test_potential_no_trade_no_change(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "no_trade", 0.05)
        assert ts.num_potential_wins["trend"] == 0
        assert ts.num_potential_losses["trend"] == 0

    def test_potential_zero_change_no_count(self, trading_strategy_instance):
        ts = trading_strategy_instance
        ts._track_potential_outcome("trend", "buy_currency_a", 0.0)
        assert ts.num_potential_wins["trend"] == 0
        assert ts.num_potential_losses["trend"] == 0

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