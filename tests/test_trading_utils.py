import pytest
import numpy as np
from utils import TradingUtils

pytestmark = pytest.mark.unit

class TestCalculatePctIncList:
    def test_pct_inc_list_basic(self):
        actual_rates = [1.0, 1.01, 1.02, 1.03, 1.04]
        pred_rates = [1.0, 1.005, 1.015, 1.025, 1.035]
        base_pct, pred_pct = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)

        for i in range(1, len(actual_rates) - 1):
            expected_base = (actual_rates[i] - actual_rates[i - 1]) / actual_rates[i - 1]
            expected_pred = (pred_rates[i + 1] - actual_rates[i]) / actual_rates[i]
            assert base_pct[i] == pytest.approx(expected_base)
            assert pred_pct[i] == pytest.approx(expected_pred)

    def test_pct_inc_list_starts_with_zero(self):
        actual_rates = [1.0, 1.01, 1.02, 1.03]
        pred_rates = [1.0, 1.005, 1.015, 1.025]
        base_pct, pred_pct = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        assert base_pct[0] == 0.0
        assert pred_pct[0] == 0.0

    def test_pct_inc_list_length(self):
        actual_rates = [1.0, 1.01, 1.02, 1.03, 1.04]
        pred_rates = [1.0, 1.005, 1.015, 1.025, 1.035]
        base_pct, pred_pct = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        expected_len = len(actual_rates) - 1  # 1 initial 0.0 + (N-2) computed
        assert len(base_pct) == expected_len
        assert len(pred_pct) == expected_len

    def test_pct_inc_list_zero_prev_price(self):
        actual_rates = [0.0, 1.0, 1.01, 1.02]
        pred_rates = [0.0, 1.0, 1.005, 1.015]
        base_pct, pred_pct = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        # prev_ratio=0 -> base zeroed; curr_ratio=1.0 is valid -> pred computed
        assert base_pct[1] == 0.0
        expected_pred = (pred_rates[2] - actual_rates[1]) / actual_rates[1]
        assert pred_pct[1] == pytest.approx(expected_pred)

    def test_pct_inc_list_zero_curr_price(self):
        actual_rates = [1.0, 0.0, 1.01, 1.02]
        pred_rates = [1.0, 0.0, 1.005, 1.015]
        base_pct, pred_pct = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        # prev_ratio=1.0 is valid -> base computed; curr_ratio=0 -> pred zeroed
        expected_base = (actual_rates[1] - actual_rates[0]) / actual_rates[0]
        assert base_pct[1] == pytest.approx(expected_base)
        assert pred_pct[1] == 0.0

class TestCalculatePctIncDict:
    def test_pct_inc_dict_basic(self):
        actual_rates = [1.0, 1.01, 1.02, 1.03, 1.04]
        pred_rates = {
            "model_a": [1.0, 1.005, 1.015, 1.025, 1.035],
            "model_b": [1.0, 1.002, 1.012, 1.022, 1.032],
        }
        base_pct, pred_dict = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        assert isinstance(pred_dict, dict)
        assert "model_a" in pred_dict
        assert "model_b" in pred_dict

    def test_pct_inc_dict_keys_preserved(self):
        actual_rates = [1.0, 1.01, 1.02, 1.03, 1.04]
        pred_rates = {"alpha": [1.0]*5, "beta": [1.0]*5, "gamma": [1.0]*5}
        _, pred_dict = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        assert set(pred_dict.keys()) == {"alpha", "beta", "gamma"}

    def test_pct_inc_dict_lengths(self):
        actual_rates = [1.0, 1.01, 1.02, 1.03, 1.04]
        pred_rates = {"m1": [1.0]*5, "m2": [1.0]*5}
        base_pct, pred_dict = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        for key in pred_dict:
            assert len(pred_dict[key]) == len(base_pct)

    def test_pct_inc_invalid_type(self):
        with pytest.raises(TypeError):
            TradingUtils.calculate_pct_inc([1.0, 2.0], "not_a_list_or_dict")

class TestCalculateSharpeRatio:
    def test_sharpe_ratio_all_same_returns(self):
        returns = [0.01, 0.01, 0.01, 0.01]
        assert TradingUtils.calculate_sharpe_ratio(returns) == 0.0

    def test_sharpe_ratio_known_values(self):
        returns = [0.05, -0.02, 0.03, 0.01]
        excess = [r - 0.0 for r in returns]
        expected = np.mean(excess) / np.std(excess)
        assert TradingUtils.calculate_sharpe_ratio(returns) == pytest.approx(expected)

    def test_sharpe_ratio_single_return(self):
        assert TradingUtils.calculate_sharpe_ratio([0.05]) == 0.0

    def test_sharpe_ratio_empty(self):
        assert TradingUtils.calculate_sharpe_ratio([]) == 0.0

    def test_sharpe_ratio_with_risk_free_rate(self):
        returns = [0.05, -0.02, 0.03, 0.01]
        risk_free = 0.01
        excess = [r - risk_free for r in returns]
        expected = np.mean(excess) / np.std(excess)
        assert TradingUtils.calculate_sharpe_ratio(returns, risk_free_rate=risk_free) == pytest.approx(expected)