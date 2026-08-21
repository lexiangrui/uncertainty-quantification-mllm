# 三模型 XML 格式 LoRA

本目录保存 XML 格式数据构造、LoRA 训练和验证代码。LoRA 的目标是稳定回答组织方式，不额外学习 ViLP、HallusionBench 或 MM-Vet 的正式评测内容。

## 训练目标

三个模型只输出一行、标签唯一且顺序固定的 XML：

```xml
<vision>视觉证据</vision><reasoning>简短推理</reasoning><answer>最终答案</answer>
```

正式模型为 LLaVA-1.5-7B、Qwen2.5-VL-7B-Instruct 和 InternVL3.5-8B。InternVL 使用 `InternVL3_5-8B` checkpoint 与 `results/lora/internvl/adapter-original`。

## 数据与正式配置

训练数据来自 VQAv2 train2014。每条样本由多模态教师生成 `vision`、`reasoning`、`answer`，再由本地代码验证并封装成 XML。当前正式数据共 5,000 条：4,000 train、1,000 validation。

| 模型 | 正式配置 | 正式 adapter |
| --- | --- | --- |
| LLaVA-1.5-7B | `LoRA/configs/llava_inline_lora_5000.json` | `results/lora/llava/adapter/` |
| Qwen2.5-VL-7B-Instruct | `LoRA/configs/qwen2_5_vl_inline_lora_5000.json` | `results/lora/qwen/adapter/` |
| InternVL3.5-8B | `LoRA/configs/internvl3_5_original_lora.json` | `results/lora/internvl/adapter-original/` |

正式映射以 `slurm/generation/generate.sbatch` 和 `slurm/improvement/run_era.sbatch` 为准。

共同超参数：

| 参数 | 值 |
| --- | ---: |
| epochs | 1 |
| learning rate | `2e-4` |
| LoRA rank / alpha / dropout | `8 / 16 / 0.05` |
| micro batch / gradient accumulation | `1 / 16` |
| warmup ratio | `0.03` |
| target modules | 语言模型的 `q_proj`、`v_proj` |

视觉塔、projector 和基础模型参数保持冻结。LLaVA 与 Qwen2.5-VL 最大序列长度为 1024；InternVL3.5-8B 为 4096，图像尺寸 448，当前正式配置每张图最多使用一个 patch。LLaVA 的监督结尾 token 为 `</s>`，Qwen 与 InternVL 为 `<|im_end|>`。

训练输出包含 adapter 权重、adapter 配置、tokenizer/processor 配置、训练与验证指标、`training_config.json` 和可恢复 checkpoint。`training_config.json` 记录训练 Prompt 的 SHA256，adapter 不是合并后的完整模型权重。

## Prompt 与教师数据

- XML 指令：`prompts/LoRA/xml_lora_instruction.md`
- 教师 system prompt：`prompts/LoRA/teacher_prompt.md`
- few-shot 示例：`prompts/LoRA/few_shot_examples.json`

教师数据记录保存 Prompt 哈希；训练配置保存 XML 指令哈希。教师 API 凭据只通过环境变量提供：

```bash
export QWEN_TEACHER_BASE_URL=
export QWEN_TEACHER_API_KEY=
```

## 数据构造与训练

数据、基础模型和教师凭据不进入仓库。集群默认路径：

```text
/opt/lexiangrui/datasets/vqav2_xml_sft/
/opt/lexiangrui/models/
```

数据构造入口：

```text
LoRA/scripts/prepare_candidates.py
LoRA/scripts/generate_teacher_data.py
LoRA/scripts/finalize_dataset.py
```

训练入口：

```text
LoRA/scripts/train_multimodal_lora.py
slurm/lora/train_internvl35_original.sbatch
```

示例：

```bash
sbatch --export=ALL,CONFIG=LoRA/configs/internvl3_5_original_lora.json \
  slurm/lora/train_internvl35_original.sbatch
```

InternVL checkpoint 与 adapter 必须使用上表中的配套版本。

## 正式生成中的挂载

不建议手工逐数据集调用底层生成脚本。正式 vLLM + HF 流水线会自动选择 checkpoint 与 adapter：

```bash
sbatch --export=ALL,MODEL=llava slurm/generation/generate.sbatch
sbatch --export=ALL,MODEL=qwen slurm/generation/generate.sbatch
sbatch --export=ALL,MODEL=internvl slurm/generation/generate.sbatch
```

InternVL 使用 `family=internvl3_5_original`、`$MODEL_ROOT/InternVL3_5-8B` 和 `results/lora/internvl/adapter-original`。vLLM 负责生成，HF 对同一 token 序列回放；两阶段必须挂载同一个 adapter。

## 测试

LoRA 单元测试不访问真实 API，也不下载模型或数据：

```bash
pytest -q LoRA/tests
```
