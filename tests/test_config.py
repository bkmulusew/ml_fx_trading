import pytest
from utils import FXTradingConfig

pytestmark = pytest.mark.unit

class TestFXTradingConfigDefaults:
    def test_default_values(self):
        config = FXTradingConfig()
        assert config.MODEL_NAME == "tcn"
        assert config.INPUT_CHUNK_LENGTH == 64
        assert config.OUTPUT_CHUNK_LENGTH == 1
        assert config.N_EPOCHS == 3
        assert config.TRAIN_BATCH_SIZE == 1024
        assert config.EVAL_BATCH_SIZE == 128
        assert config.WALLET_A == 10000.0
        assert config.WALLET_B == 10000.0
        assert config.BET_SIZING == "fixed"
        assert config.ENABLE_TRANSACTION_COSTS is False
        assert config.NEWS_HOLD_MINUTES == 3
        assert config.ALLOW_NEWS_OVERLAP is False
        assert config.SENTIMENT_SOURCE == "competitor_label"
        assert config.SEED == 59

    def test_custom_values(self):
        config = FXTradingConfig(
            MODEL_NAME="chronos",
            INPUT_CHUNK_LENGTH=128,
            OUTPUT_CHUNK_LENGTH=5,
            WALLET_A=50000.0,
            BET_SIZING="active_kelly",
            ENABLE_TRANSACTION_COSTS=True,
            SEED=99,
        )
        assert config.MODEL_NAME == "chronos"
        assert config.INPUT_CHUNK_LENGTH == 128
        assert config.OUTPUT_CHUNK_LENGTH == 5
        assert config.WALLET_A == 50000.0
        assert config.BET_SIZING == "active_kelly"
        assert config.ENABLE_TRANSACTION_COSTS is True
        assert config.SEED == 99

    def test_mutable_fields(self):
        config = FXTradingConfig()
        config.MODEL_NAME = "ensemble"
        config.WALLET_A = 99999.0
        config.INPUT_CHUNK_LENGTH = 256
        config.ENABLE_TRANSACTION_COSTS = True

        assert config.MODEL_NAME == "ensemble"
        assert config.WALLET_A == 99999.0
        assert config.INPUT_CHUNK_LENGTH == 256
        assert config.ENABLE_TRANSACTION_COSTS is True
