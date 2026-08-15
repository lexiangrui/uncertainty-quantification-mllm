#!/bin/bash
# Submit SAI extraction with the FROZEN score conditions for the given models:
# anchors {unembed, mention_state} × locate top-16 × layer n/2 × σ=1,
# lens layers scaled to each model's depth. Full LUH subset per model.
# Usage: bash slurm/improvement/submit_sai.sh [model ...]   (default: all three)
set -euo pipefail
cd "$(dirname "$0")/../.."
MODELS=${@:-llava qwen internvl}
for MODEL in $MODELS; do
  case "$MODEL" in
    llava)    N=32 ;;
    qwen)     N=28 ;;
    internvl) N=36 ;;
    *) echo "unknown model $MODEL"; exit 1 ;;
  esac
  HALF=$((N / 2))
  LAST=$((N - 1))
  L4=$((N / 4))
  L3Q=$((3 * N / 4))
  LENS="$L4 $HALF $((HALF + N / 8 + 1)) $L3Q $((L3Q + N / 8 + 1)) $LAST"
  for DS in vilp hallusionbench mmvet; do
    sed -e "s/@MODEL@/${MODEL}/" -e "s/@DS@/${DS}/" \
      slurm/improvement/compute_sai.sbatch > /tmp/sai_${MODEL}_${DS}.sbatch
    sbatch --export=ALL,ANCHOR_MODES="unembed mention_state",LOCATE_MODES="topk", \
      INTERVENE_LAYERS="$HALF",SIGMAS="1.0",LENS_LAYERS="$LENS" \
      /tmp/sai_${MODEL}_${DS}.sbatch
  done
done
