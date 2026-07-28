from pathlib import Path


ASSET_ROOT = Path("/opt/lexiangrui")

LLAVA_MODEL = ASSET_ROOT / "models/llava-1.5-7b-hf"
JUDGE_MODEL = ASSET_ROOT / "models/Qwen3-4B-Instruct-2507"
DATA_ROOT = ASSET_ROOT / "datasets"

METHOD_VERSION = "gasp-predictive-visual-noisy-or-v7"
PERTURBATION_SEEDS = (11, 23, 37, 53, 71)
SENSITIVE_TOKEN_RATIO = 0.10
REPLACEMENT_NOISE_SCALE = 1.0
NORM_ISOTROPIC_SIGMA = 0.01
SEMANTIC_VOLUME_JITTER = 1e-6
MAX_NEW_TOKENS = 64
SEED = 42
