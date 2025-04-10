import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
TRIMMED_DATA_DIR = DATA_DIR / "trimmed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Constants
SAMPLING_FREQUENCY = 2000  # Hz
NPERSEG = 2048
NOVERLAP = 1024
PADDING_VALUE = -1.0  # Spectrogram values >= 0.0
BATCH_SIZE = 32

# Configure tqdm logging if tqdm is available
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
