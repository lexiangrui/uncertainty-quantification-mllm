#!/bin/bash
# Login-node parallel range downloader for the large Xet-backed safetensors.
# hf-mirror currently closes long responses after a short range; independent
# range requests make resumable progress without touching compute nodes.
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/opt/${USER}/models}
MODEL_DIR="$MODEL_ROOT/InternVL3_5-8B"
MODEL_ID="OpenGVLab/InternVL3_5-8B"
ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PART_DIR="$MODEL_DIR/.parallel-parts"
CHUNK_BYTES=${CHUNK_BYTES:-67108864}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}

mkdir -p "$MODEL_DIR" "$PART_DIR"
export HF_ENDPOINT="$ENDPOINT"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60

download_metadata() {
  HF_PYTHON=${HF_PYTHON:-/home/${USER}/.venvs/MLLM-UQ/bin/python}
  HF_ENDPOINT="$ENDPOINT" HF_HUB_DISABLE_XET=1 "$HF_PYTHON" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    endpoint=__import__("os").environ["HF_ENDPOINT"],
    allow_patterns=["*"],
    ignore_patterns=["*.safetensors"],
    max_workers=8,
)
PY
}

file_size() {
  local file=$1
  curl --http1.1 -fsSIL --retry 5 --retry-all-errors "$ENDPOINT/$MODEL_ID/resolve/main/$file" \
    | awk 'tolower($1)=="x-linked-size:" {print $2}' | tr -d '\r' | tail -1
}

download_part() {
  local file=$1 index=$2 start=$3 end=$4
  local part="$PART_DIR/$file.$(printf '%06d' "$index")"
  local expected=$((end - start + 1))
  if [[ -s "$part" ]] && [[ "$(stat -c '%s' "$part")" -eq "$expected" ]]; then
    return 0
  fi
  local temporary="$part.tmp.$$"
  rm -f "$temporary"
  curl --http1.1 -fL --retry 20 --retry-all-errors --connect-timeout 30 \
    --range "$start-$end" "$ENDPOINT/$MODEL_ID/resolve/main/$file" \
    -o "$temporary"
  [[ "$(stat -c '%s' "$temporary")" -eq "$expected" ]] || {
    echo "short range for $file [$start,$end]" >&2
    rm -f "$temporary"
    return 1
  }
  mv "$temporary" "$part"
}

download_shard() {
  local file=$1
  local size
  size=$(file_size "$file")
  [[ "$size" =~ ^[0-9]+$ ]] || { echo "cannot determine size for $file" >&2; return 1; }
  local count=$(( (size + CHUNK_BYTES - 1) / CHUNK_BYTES ))
  echo "parallel download: $file size=$size chunks=$count" >&2
  local active=0 index start end
  for ((index=0; index<count; index++)); do
    start=$((index * CHUNK_BYTES))
    end=$((start + CHUNK_BYTES - 1))
    if ((end >= size)); then end=$((size - 1)); fi
    download_part "$file" "$index" "$start" "$end" &
    active=$((active + 1))
    if ((active >= MAX_CONCURRENT)); then
      wait -n
      active=$((active - 1))
    fi
  done
  while ((active > 0)); do wait -n; active=$((active - 1)); done
  local output="$MODEL_DIR/$file" temporary="$MODEL_DIR/$file.tmp.$$"
  for ((index=0; index<count; index++)); do
    local part="$PART_DIR/$file.$(printf '%06d' "$index")"
    [[ -s "$part" ]] || { echo "missing part $part" >&2; return 1; }
    cat "$part" >> "$temporary"
  done
  [[ "$(stat -c '%s' "$temporary")" -eq "$size" ]] || {
    echo "assembled size mismatch for $file" >&2
    rm -f "$temporary"
    return 1
  }
  mv "$temporary" "$output"
  echo "completed: $output ($size bytes)" >&2
}

download_small_file() {
  local file=$1 output="$MODEL_DIR/$1" temporary="$MODEL_DIR/$1.tmp.$$"
  if [[ -s "$output" ]]; then return 0; fi
  curl --http1.1 -fL --retry 20 --retry-all-errors --connect-timeout 30 \
    "$ENDPOINT/$MODEL_ID/resolve/main/$file" -o "$temporary"
  mv "$temporary" "$output"
}

if ! download_metadata; then
  echo "metadata snapshot had a transient failure; continuing with direct mirror downloads" >&2
fi
download_small_file modeling_intern_vit.py
download_small_file model.safetensors.index.json
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  download_shard "$shard"
done
test -s "$MODEL_DIR/model.safetensors.index.json"
echo "download_complete model=$MODEL_ID endpoint=$ENDPOINT target=$MODEL_DIR" >&2
