# Uncertainty Quantification of MLLM

本仓库保存多模态大模型（MLLM）不确定性量化第一阶段的代码、正式实验产物和结果分析。第一阶段已完成：三个模型在三个数据集上生成结构化回答，计算三种 UQ 分数，使用多模态 LLM Judge 标注正确性与幻觉，并计算统一指标。

## 实验范围

### 模型

- LLaVA-1.5-7B
- Qwen2.5-VL-7B-Instruct
- InternVL3.5-8B-HF

三个模型均挂载独立 LoRA adapter，并按固定单行 XML 协议回答：

```xml
<vision>...</vision><reasoning>...</reasoning><answer>...</answer>
```

### 数据集

- ViLP：900 个 QIA 样本
- HallusionBench：1,129 条样本
- MM-Vet：218 条样本

### UQ 方法

- Perplexity
- Semantic Entropy
- UMPIRE

## 当前状态

| 工作流 | 状态 | 正式产物 |
| --- | --- | --- |
| VQAv2 XML 数据与三模型 LoRA | 完成 | 5,000 条数据（4,000 train / 1,000 validation）和 3 个 adapter |
| 三模型 × 三数据集回答生成 | 完成 | 6,741 条主回答 |
| 三种 UQ 计算 | 完成 | 6,662 条有效 UQ 记录 |
| 多模态 LLM Judge | 完成 | 6,741 条判定，6,662 条有效 joined 记录 |
| 指标与数据分析 | 完成 | 9 个模型 × 数据集单元格的指标和汇总分析 |

正式分析表明，三种 UQ 对回答错误有中等预测能力（九个单元格宏平均 AUROC：Perplexity `0.645`、Semantic Entropy `0.681`、UMPIRE `0.671`），但对幻觉的预测明显较弱（`0.548`、`0.628`、`0.588`）。在错误回答条件下，幻觉区分能力进一步下降，说明主要信号来自回答正确性，而非对视觉幻觉的直接识别。完整结论见 [第一阶段分析报告](results/analysis/phase1_uq/report.md)。

## 保留的仓库内容

```text
.
├── LoRA/                         # XML 格式 LoRA 数据构造、训练与测试
├── baseline/
│   ├── perplexity_repro/          # 正式第一阶段 UQ 方法
│   ├── semantic_uncertainty_repro/
│   ├── umpire_repro/
│   ├── vauq-repro/                # 保留的独立论文复现
│   └── vl_uncertainty_repro/      # 保留的独立论文复现
├── docs/                          # 工程说明、实验设计与历史研究笔记
├── prompts/                       # 生成、LoRA 与闭源 Judge 的版本化 Prompt
├── results/
│   ├── generation/full_transformers_k5/
│   ├── hidden/
│   ├── uq/full_transformers_k5/
│   ├── judging/full_transformers_k5/
│   ├── metrics/full_transformers_k5/
│   ├── analysis/phase1_uq/
│   └── lora/vqav2_5000_4to1/
├── scripts/
│   ├── analysis/analyze_phase1_results.py
│   ├── analysis/compute_common_luh.py
│   ├── analysis/extract_luh_subset.py
│   ├── analysis/extract_per_model_subset.py
│   ├── evaluation/compute_metrics.py
│   ├── generation/generate_responses.py
│   ├── generation/extract_hidden_states.py
│   ├── judging/judge_responses.py
│   └── uq/compute_uq.py
├── src/                           # 数据集、模型、生成、Judge、UQ 和指标公共代码
├── slurm/                         # LoRA、生成和 UQ 作业入口
└── tests/
```

数据集与基础模型权重不提交到 Git。正式结果目录同样被 `.gitignore` 忽略，但当前工作区保留了第一阶段的最终 JSONL、指标和分析报告。

## 正式结果位置

所有结果的运行标签为 `full_transformers_k5`，其中 `k5` 表示每题包含一条 greedy 主回答和五条随机采样回答。

| 产物 | 路径 |
| --- | --- |
| 生成结果 | `results/generation/full_transformers_k5/{llava,qwen,internvl}/{vilp,hallusionbench,mmvet}.jsonl` |
| UQ 结果 | `results/uq/full_transformers_k5/{llava,qwen,internvl}/{vilp,hallusionbench,mmvet}.jsonl` |
| UMPIRE hidden 输入 | `results/hidden/` |
| Judge 结果 | `results/judging/full_transformers_k5/{llava,qwen,internvl}/{vilp,hallusionbench,mmvet}.jsonl` |
| 指标报告 | `results/metrics/full_transformers_k5/{llava,qwen,internvl}/{vilp,hallusionbench,mmvet}.metrics.json` |
| 汇总分析 | `results/analysis/phase1_uq/` |
| LUH 子集提取 | `results/analysis/luh/` |
| LoRA adapter | `results/lora/vqav2_5000_4to1/` |

## 工作流

### 1. 生成结构化回答

入口为 `scripts/generation/generate_responses.py`。它要求在 Slurm 计算节点离线运行，显式传入基础模型、对应 adapter 和数据集路径：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python3 scripts/generation/generate_responses.py \
  --dataset vilp \
  --dataset-source /opt/lexiangrui/datasets/vilp \
  --model-family llava_1_5 \
  --model-path /opt/lexiangrui/models/llava-1.5-7b-hf \
  --adapter-path results/lora/vqav2_5000_4to1/llava-1.5-7b \
  --output results/generation/example/llava-vilp.jsonl
```

默认生成一条 greedy 回答和五条随机采样回答；输出严格记录 XML 解析状态和必要的生成特征。
生成指令从 `prompts/generation/xml_lora_zero_shot_v1.md` 显式加载；运行 JSONL 的
`run` 记录中保存 `prompt_version` 和 `prompt_sha256`，用于锁定实际使用的内容。

### 2. 计算 UQ

入口为 `scripts/uq/compute_uq.py`，在 Slurm 作业内读取生成结果、对应 hidden 输入以及本地 DeBERTa entailment 模型，计算三个 UQ 方法。

正式 UQ JSONL 及 `results/hidden/` 中的 UMPIRE 输入均已保留。hidden 目录保存每条采样回答的最终答案末 token 最后一层向量，用于 UMPIRE；它不包含完整层级 hidden states、logits 或 KV cache。若需重建这些中间产物，可使用 `scripts/generation/extract_hidden_states.py` 和 `slurm/generation/extract_hidden_states.sbatch`。

### 3. Judge 与指标

`scripts/judging/judge_responses.py` 通过 OpenAI-compatible Responses API 调用多模态 Judge。服务配置仅从 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 环境变量读取。
闭源 Judge 的评分指令从
`prompts/judge/closed_source_correctness_hallucination_v1.md` 加载；Judge JSONL 的
`run` 记录保存 `judge_prompt_version` 和 `judge_prompt_sha256`。

`scripts/evaluation/compute_metrics.py` 将同一模型和数据集的 UQ/Judge JSONL 按 `sample_id` 合并，报告错误检测和幻觉检测的 AUROC、AUPRC、PRR、ECE 及 group-level bootstrap 置信区间。

`scripts/analysis/analyze_phase1_results.py` 对完整 3 × 3 矩阵生成当前保留的汇总报告、CSV 和 SVG 图表。

### 4. 低不确定性幻觉（LUH）子集提取

在既有 UQ 与 Judge 结果之上，为每个模型提取 400 样本的低不确定性幻觉（LUH）子集：
200 条幻觉且不确定性低的样本为正例，200 条在三维 PPL/SE/UMPIRE 百分位空间最近邻匹配的
非幻觉样本为负例。匹配保证三种 baseline 在子集上的 AUROC 接近 0.5，使子集成为改进方法
的诚实测试集。

- `scripts/analysis/compute_common_luh.py`：在每个 dataset × model × method 单元格内做
  average-rank percentile 归一化（默认低 UQ 阈值 0.25），输出跨至少两个模型的 common LUH
  与三个模型一致的 core LUH。
- `scripts/analysis/extract_per_model_subset.py`：按模型独立提取 400 样本子集，负例用
  三维百分位空间贪心最近邻匹配，并打印子集上的 baseline AUROC 自检。
- `scripts/analysis/extract_luh_subset.py`：跨模型公共交集策略的早期提取版本，保留作参考。

三个脚本分别通过 `--uq-root/--uq-dir`、`--judge-root/--judge-dir`、`--gen-dir` 指定输入，
产物写入 `results/analysis/luh/`。提取定义、流程与验证结果见
[低不确定性子集提取说明](docs/低不确定性子集提取说明.md)。

## LoRA adapter

三个正式 adapter 位于 `results/lora/vqav2_5000_4to1/`。训练配置、训练 loss、validation loss、adapter 权重和 tokenizer/processor 配置均与 adapter 一同保存。详情见 [LoRA README](LoRA/README.md)。

| 模型 | adapter 目录 | validation loss |
| --- | --- | ---: |
| LLaVA-1.5-7B | `llava-1.5-7b/` | 0.764681 |
| Qwen2.5-VL-7B-Instruct | `qwen2.5-vl-7b/` | 0.693499 |
| InternVL3.5-8B-HF | `internvl3.5-8b-hf/` | 0.633412 |

## 测试

测试依赖未随仓库固定安装。安装项目所需依赖后，可运行第一阶段相关测试：

```bash
pytest -q \
  tests \
  LoRA/tests \
  baseline/perplexity_repro/tests \
  baseline/semantic_uncertainty_repro/tests \
  baseline/umpire_repro/tests \
  --ignore=LoRA/tests/test_reject_resample.py
```

`LoRA/tests/test_reject_resample.py` 依赖已移除的实现，暂不作为当前验收范围。
