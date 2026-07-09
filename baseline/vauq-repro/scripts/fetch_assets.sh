#!/bin/bash
# 在 mg01 登录节点预下载模型与数据集（plain shell，不是 Slurm 作业）。
# 用法：
#   source configs/vauq.env   # 取 PYTHON_BIN / HF_* / HF_ENDPOINT
#   bash scripts/fetch_assets.sh
# 资产下载完后，推理作业即可在 HF_*_OFFLINE=1 下纯离线运行。
# 用 Python huggingface_hub.snapshot_download / datasets.load_dataset，
# 不依赖已废弃的 huggingface-cli。

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VAUQ_ASSETS_DIR="${VAUQ_ASSETS_DIR:-/opt/lexiangrui/vauq_assets}"
VAUQ_MODELS_DIR="${VAUQ_MODELS_DIR:-${VAUQ_ASSETS_DIR}/models}"
VAUQ_DATASETS_DIR="${VAUQ_DATASETS_DIR:-${VAUQ_ASSETS_DIR}/datasets}"

LLAVA_7B_REPO="${LLAVA_7B_REPO:-llava-hf/llava-1.5-7b-hf}"
LLAVA_7B_DIR="${LLAVA_7B_DIR:-${VAUQ_MODELS_DIR}/llava-1.5-7b-hf}"
LLAVA_13B_REPO="${LLAVA_13B_REPO:-llava-hf/llava-1.5-13b-hf}"
LLAVA_13B_DIR="${LLAVA_13B_DIR:-${VAUQ_MODELS_DIR}/llava-1.5-13b-hf}"
QWEN25_REPO="${QWEN25_REPO:-Qwen/Qwen2.5-VL-7B-Instruct}"
QWEN25_DIR="${QWEN25_DIR:-${VAUQ_MODELS_DIR}/Qwen2.5-VL-7B-Instruct}"
INTERNVL_REPO="${INTERNVL_REPO:-OpenGVLab/InternVL3_5-8B-HF}"
INTERNVL_DIR="${INTERNVL_DIR:-${VAUQ_MODELS_DIR}/InternVL3_5-8B-HF}"

# Comma-separated selectors. Defaults fetch only models used for inference.
# qwen25/internvl selectors are retained for preserving model-weight assets.
FETCH_MODELS="${FETCH_MODELS:-llava7,llava13}"
FETCH_DATASETS="${FETCH_DATASETS:-cvbench,mmvet,vilp}"
FETCH_VISUALCOT_IMAGES="${FETCH_VISUALCOT_IMAGES:-0}"

mkdir -p "$VAUQ_MODELS_DIR" "$VAUQ_DATASETS_DIR"

export \
    LLAVA_7B_REPO LLAVA_7B_DIR LLAVA_13B_REPO LLAVA_13B_DIR \
    QWEN25_REPO QWEN25_DIR INTERNVL_REPO INTERNVL_DIR \
    VAUQ_DATASETS_DIR FETCH_MODELS FETCH_DATASETS FETCH_VISUALCOT_IMAGES

echo "[1/3] downloading selected models -> ${VAUQ_MODELS_DIR}"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

selected = {x.strip() for x in os.environ["FETCH_MODELS"].split(",") if x.strip()}
models = {
    "llava7": (os.environ["LLAVA_7B_REPO"], os.environ["LLAVA_7B_DIR"]),
    "llava13": (os.environ["LLAVA_13B_REPO"], os.environ["LLAVA_13B_DIR"]),
    "qwen25": (os.environ["QWEN25_REPO"], os.environ["QWEN25_DIR"]),
    "internvl": (os.environ["INTERNVL_REPO"], os.environ["INTERNVL_DIR"]),
}
for name, (repo_id, local_dir) in models.items():
    if name not in selected:
        continue
    print(f"downloading {name}: {repo_id} -> {local_dir}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"{name} downloaded: {local_dir}", flush=True)
PY

echo "[2/3] caching selected datasets -> ${VAUQ_DATASETS_DIR}"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

selected = {x.strip() for x in os.environ["FETCH_DATASETS"].split(",") if x.strip()}

if "cvbench" in selected:
    load_dataset("nyu-visionx/CV-Bench", "2D")
    load_dataset("nyu-visionx/CV-Bench", "3D")
    print("cached CV-Bench 2D/3D", flush=True)
if "mmvet" in selected:
    load_dataset("whyu/mm-vet")
    print("cached MM-Vet", flush=True)
if "vilp" in selected:
    hf_hub_download(
        repo_id="ViLP/ViLP",
        filename="ViLP.parquet",
        repo_type="dataset",
        local_dir=os.path.join(os.environ["VAUQ_DATASETS_DIR"], "vilp"),
        token=os.environ.get("HF_TOKEN"),
    )
    print("cached ViLP parquet", flush=True)
if "visualcot" in selected:
    visualcot_root = os.path.join(os.environ["VAUQ_DATASETS_DIR"], "visualcot")
    snapshot_download(
        repo_id="deepcs233/Visual-CoT",
        repo_type="dataset",
        local_dir=visualcot_root,
        allow_patterns=["cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"],
        token=os.environ.get("HF_TOKEN"),
    )
    print("cached VisualCoT metadata", flush=True)
    if os.environ.get("FETCH_VISUALCOT_IMAGES") == "1":
        snapshot_download(
            repo_id="deepcs233/Visual-CoT",
            repo_type="dataset",
            local_dir=visualcot_root,
            allow_patterns=["cot_images_tar_split/*"],
            token=os.environ.get("HF_TOKEN"),
        )
        print("cached VisualCoT image tar shards", flush=True)

print("datasets cached under", os.environ.get("HF_DATASETS_CACHE", "(default)"))
PY

echo "[3/3] extracting VisualCoT image tar shards if requested"
if [ "$FETCH_VISUALCOT_IMAGES" = "1" ]; then
    VISUALCOT_ROOT="${VAUQ_DATASETS_DIR}/visualcot"
    VISUALCOT_IMAGE_DIR="${VISUALCOT_IMAGE_DIR:-${VAUQ_DATASETS_DIR}/visualcot_images}"
    mkdir -p "$VISUALCOT_IMAGE_DIR"
    if [ -d "${VISUALCOT_ROOT}/cot_images_tar_split" ]; then
        if [ -z "$(find "$VISUALCOT_IMAGE_DIR" -type f -name '*.jpg' -print -quit 2>/dev/null)" ]; then
            echo "extracting VisualCoT images -> ${VISUALCOT_IMAGE_DIR}"
            cat "${VISUALCOT_ROOT}"/cot_images_tar_split/cot_images_* | tar -xf - -C "$VISUALCOT_IMAGE_DIR"
        else
            echo "VisualCoT images already present under ${VISUALCOT_IMAGE_DIR}"
        fi
    else
        echo "WARNING: VisualCoT tar shards were not found under ${VISUALCOT_ROOT}/cot_images_tar_split" >&2
    fi
else
    echo "skip VisualCoT images (set FETCH_VISUALCOT_IMAGES=1 to fetch/extract ~130GB tar shards)"
fi

echo "--- verify ---"
echo "models:"; ls -1 "${VAUQ_MODELS_DIR}" 2>/dev/null || true
echo "llava7 shards:"; ls -lh "${LLAVA_7B_DIR}"/*.safetensors 2>/dev/null || true
echo "llava13 shards:"; ls -lh "${LLAVA_13B_DIR}"/*.safetensors 2>/dev/null || true
echo "qwen25 shards:"; ls -lh "${QWEN25_DIR}"/*.safetensors 2>/dev/null || true
echo "internvl shards:"; ls -lh "${INTERNVL_DIR}"/*.safetensors 2>/dev/null || true
echo "datasets:"; ls -1 "${VAUQ_DATASETS_DIR}" 2>/dev/null || true
echo "done."
