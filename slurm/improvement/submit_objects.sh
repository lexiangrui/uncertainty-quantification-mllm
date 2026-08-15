#!/bin/bash
# Submit SAI object extraction (independent text model) for the given models.
# Usage: bash slurm/improvement/submit_objects.sh [model ...]   (default: all three)
set -euo pipefail
cd "$(dirname "$0")/../.."
MODELS=${@:-llava qwen internvl}
for MODEL in $MODELS; do
  for DS in vilp hallusionbench mmvet; do
    sed -e "s/@MODEL@/${MODEL}/" -e "s/@DS@/${DS}/" \
      slurm/improvement/extract_objects_sai.sbatch > /tmp/sai_obj_${MODEL}_${DS}.sbatch
    sbatch /tmp/sai_obj_${MODEL}_${DS}.sbatch
  done
done
