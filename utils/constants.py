from dataclasses import dataclass

@dataclass
class ModelConfig:
    MODEL_NAME: str = "tcn"
    INPUT_CHUNK_LENGTH: int = 64
    OUTPUT_CHUNK_LENGTH: int = 1
    N_EPOCHS: int = 3
    BATCH_SIZE: int = 1024
    DATA_PATH_TRAIN: str = "ml_fx_trading/dataset/exchange_rate-train.csv"
    DATA_PATH_VAL: str = "ml_fx_trading/dataset/exchange_rate-val.csv"
    DATA_PATH_TEST: str = "ml_fx_trading/dataset/exchange_rate-test.csv"
    WALLET_A: float = 10000.0
    WALLET_B: float = 10000.0
    USE_FRAC_KELLY: bool = True
    ENABLE_TRANSACTION_COSTS: bool = False
    OUTPUT_DIR: str = "results/usd-cny-2023"
    NEWS_MIN_HOLD_BARS: int = 3
