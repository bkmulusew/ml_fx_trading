from dataclasses import dataclass

@dataclass
class ModelConfig:
    INPUT_CHUNK_LENGTH: int = 128
    OUTPUT_CHUNK_LENGTH: int = 1
    N_EPOCHS: int = 3
    BATCH_SIZE: int = 512
    NUM_SAMPLES: int = 16
    DATA_FILE_PATH: str = "ml_fx_trading/dataset/exchange_rate.csv"
    TRAIN_RATIO: float = 0.5
    WALLET_A: float = 10000.0
    WALLET_B: float = 10000.0
    ENABLE_FRAC_KELLY: bool = True
    ENABLE_TRANSACTION_COSTS: bool = False
    HOLD_POSITION: bool = False
    OUTPUT_DIR: str = "results/usd-cny-2023"
