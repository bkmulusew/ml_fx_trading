from dataclasses import dataclass

@dataclass
class ModelConfig:
    MODEL_NAME: str = "tcn"
    INPUT_CHUNK_LENGTH: int = 64
    OUTPUT_CHUNK_LENGTH: int = 1
    N_EPOCHS: int = 3
    BATCH_SIZE: int = 1024
    FX_DATA_PATH_TRAIN: str = "ml_fx_trading/dataset/fx/usdcnh-fx-train.csv"
    FX_DATA_PATH_VAL: str = "ml_fx_trading/dataset/fx/usdcnh-fx-val.csv"
    FX_DATA_PATH_TEST: str = "ml_fx_trading/dataset/fx/usdcnh-fx-test.csv"
    NEWS_DATA_PATH_TRAIN: str = "ml_fx_trading/dataset/news/usdcnh-news-train.csv"
    NEWS_DATA_PATH_VAL: str = "ml_fx_trading/dataset/news/usdcnh-news-val.csv"
    NEWS_DATA_PATH_TEST: str = "ml_fx_trading/dataset/news/usdcnh-news-test.csv"
    WALLET_A: float = 10000.0
    WALLET_B: float = 10000.0
    USE_FRAC_KELLY: bool = True
    ENABLE_TRANSACTION_COSTS: bool = False
    OUTPUT_DIR: str = "results/usd-cnh"
    NEWS_HOLD_MINUTES: int = 3
    ALLOW_NEWS_OVERLAP: bool = False
    SENTIMENT_SOURCE: str = "competitor_label"
