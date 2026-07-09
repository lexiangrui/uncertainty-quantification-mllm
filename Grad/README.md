# Grad-VAUQ

Gradient-only VAUQ variant for white-box LVLM self-evaluation.

This directory is intentionally separate from `src/vauq/` so the original
reproduction stays untouched. The method removes the attention-based core-region
selector and uses gradient attribution over visual embeddings instead.

## Idea

```text
answer likelihood -> gradient over visual tokens -> top-k visual tokens
                  -> ablate selected tokens -> entropy change
```

The score keeps the VAUQ outer formula:

```text
IS_grad = H(y | v_grad_masked, t) - H(y | v, t)
s       = H(y | v, t) - alpha * IS_grad
```

Unlike the original attention-based implementation, this path does not call
`output_attentions=True`, so the backend can use Flash Attention 2.

## Mathematical Model

See `Grad/docs/grad_vauq_math.md` for the full derivation from multimodal input
to teacher-forced entropy, gradient visual-token selection, masked forward, IS,
and the final Grad-VAUQ score.

## Structure

```text
Grad/
├── grad_vauq/
│   ├── adapters/          # model-specific visual-token capture/ablation
│   ├── backends/          # generation and teacher-forced logits
│   ├── selectors.py       # generic gradient selectors
│   ├── scoring.py         # entropy / IS / VAUQ
│   └── types.py
└── scripts/
    └── run_grad_vauq.py
```

Only the adapter layer is model-specific. To add another LVLM, implement a new
`VisionTokenAdapter` that captures the tensor of visual embeddings entering the
language model and ablates selected token indices in that tensor.

## LLaVA Example

```bash
cd vauq-repro
python Grad/scripts/run_grad_vauq.py \
  --backend llava \
  --benchmark cvbench \
  --judge letter \
  --model-path "$LLAVA_HF_MODEL" \
  --limit 4 \
  --output results/grad_cvbench_debug.jsonl
```

Useful options:

```bash
--attn-implementation flash_attention_2
--selector grad_x_act
--ablation-baseline attention_mask
--topk-ratio 0.3
--alpha 1.2
```

For Flash Attention 2, install a compatible `flash-attn` build in the runtime
environment used by Transformers.
