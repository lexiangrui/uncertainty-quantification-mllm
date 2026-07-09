# Uncertainty Quantification of MLLM

This workspace contains three reproduced baseline projects and a new
gradient-only Grad-VAUQ implementation.

## Layout

```text
baseline/
├── semantic_uncertainty_repro/
├── vauq-repro/
└── vl_uncertainty_repro/

Grad/
└── grad_vauq/
```

`baseline/` keeps the prior reproduction code plus curated final summaries and
successful logs. `Grad/` is the new method directory. It removes attention-based
core-token selection and uses gradient attribution over visual embeddings, so it
can run with Flash Attention 2.

## Grad-VAUQ Quick Start

```bash
python Grad/scripts/run_grad_vauq.py \
  --backend llava \
  --benchmark cvbench \
  --judge letter \
  --model-path "$LLAVA_HF_MODEL" \
  --attn-implementation flash_attention_2 \
  --limit 4 \
  --output results/grad_cvbench_debug.jsonl
```

See `Grad/README.md` for the method and adapter design.
