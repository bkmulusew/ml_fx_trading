import pytest
import numpy as np
from metrics import ModelEvalMetrics

pytestmark = pytest.mark.unit

@pytest.fixture
def metrics():
    return ModelEvalMetrics()

class TestRMSE:
    def test_rmse_perfect_predictions(self, metrics):
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        assert metrics.calculate_rmse(actual, predicted) == pytest.approx(0.0)

    def test_rmse_known_values(self, metrics):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.1, 2.2, 2.8]
        # RMSE = sqrt(((0.1)^2 + (0.2)^2 + (0.2)^2) / 3)
        #      = sqrt((0.01 + 0.04 + 0.04) / 3) = sqrt(0.09/3) = sqrt(0.03)
        expected = np.sqrt(np.mean([(1.0-1.1)**2, (2.0-2.2)**2, (3.0-2.8)**2]))
        assert metrics.calculate_rmse(actual, predicted) == pytest.approx(expected)

class TestMAPE:
    def test_mape_perfect_predictions(self, metrics):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 3.0]
        assert metrics.calculate_mape(actual, predicted) == pytest.approx(0.0)

    def test_mape_known_values(self, metrics):
        actual = [100.0, 200.0, 300.0]
        predicted = [110.0, 190.0, 280.0]
        # |10/100| + |10/200| + |20/300| = 0.1 + 0.05 + 0.0666..
        expected = 100 * np.mean([10.0/100.0, 10.0/200.0, 20.0/300.0])
        assert metrics.calculate_mape(actual, predicted) == pytest.approx(expected)

    def test_mape_zero_actual(self, metrics):
        actual = [0.0, 1.0, 2.0]
        predicted = [0.5, 1.0, 2.0]
        result = metrics.calculate_mape(actual, predicted)
        assert np.isfinite(result)

class TestSMAPE:
    def test_smape_perfect_predictions(self, metrics):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 3.0]
        assert metrics.calculate_smape(actual, predicted) == pytest.approx(0.0)

    def test_smape_known_values(self, metrics):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        denom = (np.abs(actual) + np.abs(predicted)) / 2.0
        expected = 100 * np.mean(np.abs(actual - predicted) / denom)
        assert metrics.calculate_smape(actual, predicted) == pytest.approx(expected)

    def test_smape_both_zero(self, metrics):
        actual = [0.0, 0.0]
        predicted = [0.0, 0.0]
        result = metrics.calculate_smape(actual, predicted)
        assert np.isfinite(result)
        assert result == pytest.approx(0.0)

class TestMASE:
    def test_mase_perfect_predictions(self, metrics):
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        assert metrics.calculate_mase(actual, predicted) == pytest.approx(0.0)

    def test_mase_known_values(self, metrics):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.5, 2.5, 3.5, 4.5])
        d = np.abs(np.diff(actual)).sum() / (len(actual) - 1)
        expected = np.mean(np.abs(actual - predicted)) / d
        assert metrics.calculate_mase(actual, predicted) == pytest.approx(expected)

    def test_mase_constant_series(self, metrics):
        actual = [1.25, 1.25, 1.25, 1.25]
        predicted = [1.26, 1.24, 1.25, 1.27]
        result = metrics.calculate_mase(actual, predicted)
        assert np.isfinite(result)
        assert result == 0.0

class TestCalculatePredictionError:
    def test_calculate_prediction_error_returns_all_keys(self, metrics):
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.1, 2.1, 3.1, 4.1, 5.1]
        result = metrics.calculate_prediction_error(predicted, actual)
        assert set(result.keys()) == {"RMSE", "MASE", "MAPE", "sMAPE"}

    def test_calculate_prediction_error_consistency(self, metrics):
        actual = [10.0, 20.0, 30.0, 40.0, 50.0]
        predicted = [11.0, 19.0, 31.0, 38.0, 52.0]
        result = metrics.calculate_prediction_error(actual, predicted)

        assert result["RMSE"] == pytest.approx(metrics.calculate_rmse(actual, predicted))
        assert result["MASE"] == pytest.approx(metrics.calculate_mase(actual, predicted))
        assert result["MAPE"] == pytest.approx(metrics.calculate_mape(actual, predicted))
        assert result["sMAPE"] == pytest.approx(metrics.calculate_smape(actual, predicted))