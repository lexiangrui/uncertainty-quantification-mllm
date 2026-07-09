#!/bin/bash
#SBATCH --job-name=vauq-llava-cvbench
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
fi

export PATH="/opt/slurm/bin:$PATH"

# 模型走本地路径（$LLAVA_HF_MODEL），数据集走 HF_DATASETS_CACHE 本地缓存。
# 计算节点无公网，HF_HUB_OFFLINE=1 让 huggingface_hub 跳过 HEAD 探测、直接用缓存，
# 避免 DNS 失败后的 5 次重试噪音（不强制 datasets/transformers 离线）。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

if [ -f configs/vauq.env ]; then
    source configs/vauq.env
else
    echo "ERROR: configs/vauq.env missing (copy from vauq.env.example)" >&2
    exit 1
fi

"$PYTHON_BIN" scripts/run_vauq.py \
    --backend llava \
    --benchmark cvbench \
    --judge letter \
    --model-path "$LLAVA_HF_MODEL" \
    --mask-strategy core \
    --inference-temp 0.1 \
    --max-new-tokens 64 \
    ${LIMIT:+--limit "$LIMIT"} \
    --output "${OUTPUT:-results/cvbench_llava_vauq.jsonl}"
