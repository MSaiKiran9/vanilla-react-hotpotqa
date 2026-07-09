

from pathlib import Path
import torch


# =============================================================================
# Project Directories
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

for directory in (DATA_DIR, OUTPUT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset
# =============================================================================

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"

# Number of evaluation samples.
# Set to None to evaluate the full validation set.
SAMPLE_SIZE = 100

# Random seed for reproducibility.
RANDOM_SEED = 42


# =============================================================================
# Device
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)


# =============================================================================
# Models
# =============================================================================

# MODELS = {
#     "Qwen3-8B": "Qwen/Qwen3-8B",
#     "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
#     "Gemma-2-9B": "google/gemma-2-9b",
#     "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
# }
MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma-2-9B": "google/gemma-2-9b-it",
    "Phi-4-mini": "microsoft/Phi-4-mini-instruct",
}
# Load one model at a time to reduce GPU memory usage.
LOAD_ONE_MODEL_AT_A_TIME = True

# Optional 4-bit quantization.
#
# NOTE:
# This is disabled by default.
# We will add support inside models/loader.py.
USE_4BIT = True


# =============================================================================
# Generation Parameters
# =============================================================================

MAX_NEW_TOKENS = 192

TEMPERATURE = 0.0

TOP_P = 1.0

DO_SAMPLE = False

REPETITION_PENALTY = 1.0


# =============================================================================
# Vanilla ReAct
# =============================================================================

MAX_REACT_STEPS = 5

MAX_SEARCH_RESULTS = 5

MAX_OBSERVATION_CHARS = 1800

MAX_PAGE_CHARS = 4000


# =============================================================================
# Retrieval
# =============================================================================

USER_AGENT = (
    "VanillaReActResearchBot/1.0 "
    "(https://github.com/your-repository)"
)

REQUEST_TIMEOUT = 15


# =============================================================================
# Evaluation
# =============================================================================

OUTPUT_EXCEL = OUTPUT_DIR / "evaluation.xlsx"

OUTPUT_FIGURE = OUTPUT_DIR / "model_comparison.png"

TRACE_LOG_DIR = LOG_DIR

SAVE_TO_DRIVE = False

DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/VanillaReActResults"


# =============================================================================
# Plot Settings
# =============================================================================

FIGURE_DPI = 300

FIGURE_WIDTH = 14

FIGURE_HEIGHT = 8


# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = "INFO"

DEBUG = False


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility.
    """

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
