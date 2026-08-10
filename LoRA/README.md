# 三模型 XML 格式 LoRA

本目录保存第一阶段使用的 XML 格式 LoRA 数据构造、训练和测试代码。正式训练已完成，三个 adapter 与完整训练记录位于仓库根目录的 `results/lora/vqav2_5000_4to1/`。

## 训练目标

三个模型学习只输出一行、标签顺序固定的 XML：

```xml
<vision>视觉证据</vision><reasoning>简短推理</reasoning><answer>最终答案</answer>
```

模型分别为 LLaVA-1.5-7B、Qwen2.5-VL-7B-Instruct 和 InternVL3.5-8B-HF。LoRA 只用于约束回答格式；ViLP、HallusionBench 和 MM-Vet 未参与 LoRA 数据构造。

## 正式训练状态

正式数据来自 VQAv2 train2014。每条样本由教师模型生成 `vision`、`reasoning`、`answer` 字段，再由本地代码验证并封装为 XML。最终训练集共有 5,000 条：4,000 条训练、1,000 条 validation。

| 模型 | adapter 目录 | train / validation | updates | validation loss |
| --- | --- | ---: | ---: | ---: |
| LLaVA-1.5-7B | `results/lora/vqav2_5000_4to1/llava-1.5-7b/` | 4,000 / 1,000 | 250 | 0.764681 |
| Qwen2.5-VL-7B-Instruct | `results/lora/vqav2_5000_4to1/qwen2.5-vl-7b/` | 4,000 / 1,000 | 250 | 0.693499 |
| InternVL3.5-8B-HF | `results/lora/vqav2_5000_4to1/internvl3.5-8b-hf/` | 4,000 / 1,000 | 250 | 0.633412 |

每个 adapter 目录均包含：

```text
adapter_model.safetensors
adapter_config.json
tokenizer.json
tokenizer_config.json
processor_config.json
chat_template.jinja
train_metrics.jsonl
validation_metrics.jsonl
training_config.json
checkpoint-latest.pt
```

这些是可挂载到基础模型的 LoRA adapter，不是合并后的完整模型权重。

训练时的 XML 指令由根目录的 `prompts/LoRA/xml_lora_instruction_v1.md` 显式加载。
每次训练写出的 `training_config.json` 会记录 `prompt_version` 和 `prompt_sha256`，以便
对应到具体 Prompt 内容。教师数据生成使用的 system prompt 和 few-shot 示例也在
`prompts/LoRA/teacher_prompt.md` 与 `prompts/LoRA/few_shot_examples.json`；生成记录保存其哈希。

## 训练配置

三个正式配置位于：

```text
LoRA/configs/llava_inline_lora_5000.json
LoRA/configs/qwen2_5_vl_inline_lora_5000.json
LoRA/configs/internvl3_5_inline_lora_5000.json
```

共同超参数为：1 epoch、学习率 `2e-4`、LoRA rank `8`、alpha `16`、dropout `0.05`、micro batch `1`、梯度累积 `16`、warmup ratio `0.03`。仅语言模型的 `q_proj` 和 `v_proj` 注入 LoRA；视觉塔、projector 和原始模型参数保持冻结。

LLaVA 与 Qwen2.5-VL 的最大序列长度为 `1024`，InternVL3.5 为 `4096`。监督目标覆盖 assistant XML 和模型的 end-of-turn token：LLaVA 使用 `</s>`，其余两个模型使用 `<|im_end|>`。

## 数据与训练入口

数据、基础权重和教师 API 凭据不进入仓库：

- VQAv2 XML 数据：`/opt/lexiangrui/datasets/vqav2_xml_sft/`
- 基础模型：`/opt/lexiangrui/models/`
- 正式 adapter：`results/lora/vqav2_5000_4to1/`

数据构造入口：

```text
LoRA/scripts/prepare_candidates.py
LoRA/scripts/generate_teacher_data.py
LoRA/scripts/finalize_dataset.py
```

训练入口：

```text
LoRA/scripts/train_multimodal_lora.py
slurm/lora/train_multimodal_lora.sbatch
```

教师服务配置仅通过环境变量提供：

```bash
QWEN_TEACHER_BASE_URL=
QWEN_TEACHER_API_KEY=
```

## 在第一阶段生成中挂载 adapter

以 LLaVA 为例：

```bash
python3 scripts/generation/generate_responses.py \
  --dataset vilp \
  --dataset-source /opt/lexiangrui/datasets/vilp \
  --model-family llava_1_5 \
  --model-path /opt/lexiangrui/models/llava-1.5-7b-hf \
  --adapter-path results/lora/vqav2_5000_4to1/llava-1.5-7b \
  --output results/generation/example/llava-vilp.jsonl
```

Qwen2.5-VL 和 InternVL3.5 分别使用对应 family、基础模型目录和表中的 adapter 目录。

## 测试

LoRA 单元测试不访问真实 API，也不下载模型或数据：

```bash
pytest -q LoRA/tests --ignore=LoRA/tests/test_reject_resample.py
```

`test_reject_resample.py` 依赖已移除的实验代码，当前不纳入验收范围。
