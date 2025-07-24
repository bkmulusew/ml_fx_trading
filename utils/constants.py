from dataclasses import dataclass


@dataclass
class ModelConfig:
    INPUT_CHUNK_LENGTH: int = 64
    OUTPUT_CHUNK_LENGTH: int = 1
    N_EPOCHS: int = 3
    BATCH_SIZE: int = 256
    NUM_SAMPLES: int = 4
    DATA_FILE_PATH: str = "../ml_fx_trading/dataset/usd-cyn-2023.xlsx"
    TRAIN_RATIO: float = 0.5
    WALLET_A: float = 10000.0
    WALLET_B: float = 10000.0
    FRAC_KELLY: bool = True
    ENABLE_TRANSACTION_COSTS: bool = False
