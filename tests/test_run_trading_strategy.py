import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys

pytestmark = pytest.mark.unit

def _mock_heavy_imports():
    """Mock all heavy ML imports so run_trading_strategy can be loaded."""
    mock_modules = {}
    for mod_name in [
        "torch", "torch.cuda",
        "models", "models.base_model", "models.darts_model",
        "models.chronos_model", "models.toto_model",
        "matplotlib", "matplotlib.pyplot",
    ]:
        mock_modules[mod_name] = MagicMock()
    return mock_modules

def _import_group_data_by_date():
    mocks = _mock_heavy_imports()
    with patch.dict("sys.modules", mocks):
        # Force re-import by removing cached module
        sys.modules.pop("run_trading_strategy", None)
        from run_trading_strategy import group_data_by_date
    return group_data_by_date

def _import_set_seed():
    mocks = _mock_heavy_imports()
    with patch.dict("sys.modules", mocks):
        sys.modules.pop("run_trading_strategy", None)
        from run_trading_strategy import set_seed
    return set_seed

class TestGroupDataByDate:
    def test_group_by_date_list_predictions(self):
        group_data_by_date = _import_group_data_by_date()
        base = datetime(2023, 6, 1, 10, 0)
        n = 5
        fx_ts = [base + timedelta(minutes=i) for i in range(n)]
        news_ts = [base + timedelta(minutes=2)]
        true_vals = [1.25 + 0.001 * i for i in range(n)]
        pred_vals = [v + 0.0005 for v in true_vals]
        bids = [v - 0.0001 for v in true_vals]
        asks = [v + 0.0001 for v in true_vals]
        news_sents = [1]

        result = group_data_by_date(fx_ts, news_ts, true_vals, pred_vals, bids, asks, news_sents)
        date_key = base.date()
        assert date_key in result
        assert isinstance(result[date_key]["predicted_values"], list)
        assert len(result[date_key]["predicted_values"]) == n

    def test_group_by_date_dict_predictions(self):
        group_data_by_date = _import_group_data_by_date()
        base = datetime(2023, 6, 1, 10, 0)
        n = 5
        fx_ts = [base + timedelta(minutes=i) for i in range(n)]
        news_ts = [base + timedelta(minutes=2)]
        true_vals = [1.25 + 0.001 * i for i in range(n)]
        pred_vals = {"m1": [v + 0.0005 for v in true_vals], "m2": [v + 0.001 for v in true_vals]}
        bids = [v - 0.0001 for v in true_vals]
        asks = [v + 0.0001 for v in true_vals]
        news_sents = [1]

        result = group_data_by_date(fx_ts, news_ts, true_vals, pred_vals, bids, asks, news_sents)
        date_key = base.date()
        assert isinstance(result[date_key]["predicted_values"], dict)
        assert set(result[date_key]["predicted_values"].keys()) == {"m1", "m2"}

    def test_group_by_date_multiple_dates(self):
        group_data_by_date = _import_group_data_by_date()
        dates = [
            datetime(2023, 6, 1, 10, 0),
            datetime(2023, 6, 1, 11, 0),
            datetime(2023, 6, 2, 10, 0),
            datetime(2023, 6, 2, 11, 0),
            datetime(2023, 6, 3, 10, 0),
        ]
        n = len(dates)
        vals = [1.25] * n
        preds = [1.26] * n
        bids = [1.2499] * n
        asks = [1.2501] * n

        result = group_data_by_date(dates, [], vals, preds, bids, asks, [])
        assert len(result) == 3

    def test_group_by_date_news_attached(self):
        group_data_by_date = _import_group_data_by_date()
        base = datetime(2023, 6, 1, 10, 0)
        n = 5
        fx_ts = [base + timedelta(minutes=i) for i in range(n)]
        news_ts = [base + timedelta(minutes=1), base + timedelta(minutes=3)]
        vals = [1.25] * n
        preds = [1.26] * n
        bids = [1.2499] * n
        asks = [1.2501] * n
        sents = [1, -1]

        result = group_data_by_date(fx_ts, news_ts, vals, preds, bids, asks, sents)
        date_key = base.date()
        assert len(result[date_key]["news_timestamps"]) == 2
        assert result[date_key]["news_sentiments"] == [1, -1]

    def test_group_by_date_news_missing_date(self, capsys):
        group_data_by_date = _import_group_data_by_date()
        base = datetime(2023, 6, 1, 10, 0)
        fx_ts = [base]
        news_ts = [datetime(2023, 6, 5, 10, 0)]
        vals = [1.25]
        preds = [1.26]
        bids = [1.2499]
        asks = [1.2501]
        sents = [1]

        result = group_data_by_date(fx_ts, news_ts, vals, preds, bids, asks, sents)
        captured = capsys.readouterr()
        assert "News Date key not found" in captured.out
        # Orphan news must NOT create a phantom date bucket
        assert datetime(2023, 6, 5).date() not in result
        assert len(result) == 1

class TestSetSeed:
    def test_set_seed_determinism(self):
        set_seed = _import_set_seed()
        set_seed(42)
        val1 = np.random.rand()
        set_seed(42)
        val2 = np.random.rand()
        assert val1 == val2