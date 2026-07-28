import os
from pathlib import Path


# 路径与解释器默认值指向 MiliLab 集群；换机器时用同名环境变量覆盖，
# 例如 MALP_ASSET_ROOT=/data/assets MALP_PROJECT_ROOT=/work/me/proj ...
PYTHON_BIN = os.environ.get(
    "MALP_PYTHON_BIN",
    "/home/lexiangrui/.venvs/vlm-transformers/bin/python",
)
ASSET_ROOT = Path(os.environ.get("MALP_ASSET_ROOT", "/opt/lexiangrui"))
PROJECT_ROOT = Path(
    os.environ.get("MALP_PROJECT_ROOT", "/home/lexiangrui/Uncertainty-Quantification-of-MLLM")
)

LLAVA_MODEL = ASSET_ROOT / "models/llava-1.5-7b-hf"
DATA_ROOT = ASSET_ROOT / "datasets"

RESULT_ROOT = PROJECT_ROOT / "results" / "malp"
PERTURB_ROOT = RESULT_ROOT / "perturb"

SEED = 42
MAX_NEW_TOKENS = 64

NUM_PERTURBATIONS = 5
PERTURBATION_SEEDS = (42, 43, 44, 45, 46)

# Answer-consistency experiments need more Monte Carlo draws than the
# teacher-forcing PIS/KL experiment.  Seeds are derived as SEED + draw_index.
NUM_CONSISTENCY_GENERATIONS = 10

# 扰动强度。LLaVA 以 fp16 运行，Stage-2 embedding 量级 O(1~10)，
# 0.01 作为当前默认实验强度；如需扫描更强扰动，
# 按设计文档网格 {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} 通过 --sigma 覆盖。
PERTURB_SIGMA = 0.01
TEXT_GAMMA = 1.0

# LLaMA decoder layers whose self-attention outputs are perturbed together to
# measure middle-block reasoning uncertainty. Indices are zero-based.
REASONING_LAYERS = (15, 16, 17, 18, 19, 20)
