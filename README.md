# Uncertainty Quantification of MLLM

本仓库保存多模态大模型（MLLM）不确定性量化第一阶段的代码和实验结果：三个模型在三个数据集上生成结构化回答，计算三种 UQ 分数，使用 LLM Judge 标注正确性与幻觉，并计算统一指标。当前 `results/` 已按固定 K=10 结构完成整理。

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

| 工作流 | 状态 | 当前产物 |
| --- | --- | --- |
| VQAv2 XML 数据与三模型 LoRA | 完成 | 3 个 adapter 位于 `results/lora/<mllm>/adapter/` |
| 三模型 × 三数据集回答生成 | 完成 | 6,741 条 greedy + 6,741 组 K=10 samples |
| samples hidden | 完成 | 6,678 个 sidecar 位于 `results/hidden/` |
| 三种 UQ 计算 | 完成 | 9 个文件、6,662 条有效 UQ 记录 |
| 多模态 LLM Judge | 完成 | 9 个文件、6,741 条 frozen greedy 标签 |
| 指标计算 | 完成 | 9 个指标报告位于 `results/metrics/` |
| 结果数据分析 | 完成 | 描述性统计、检测性能对比与 LUH 画像位于 `results/analysis/{descriptive,detection,luh_profile}/` |
| 实验二（改进 UQ 方法：ERA） | 完成 | 早期推理归因（Early Rationale Attribution）单前向 UQ 方法，产物见 `docs/ERA早期推理归因不确定性量化方法.md` 与 `src/improvement/` |
| 下一步研究方向 | 推进中 | 全基准泛化评估、ERA 引导的自适应解码/幻觉抑制与细粒度归因定位 |

采样数固定为 K=10；每个 sample 最多进行 50 次 XML 格式拒绝重采样。greedy 回答和 Judge 标签复用原第一阶段 K=5 运行中的 greedy 部分，不重新生成或重新 Judge；samples、hidden 和 UQ 已按 K=10 重算。目录不再使用 K 作为层级。

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
├── docs/                          # 工程说明、实验设计与 ERA 方法说明
├── prompts/                       # 生成、LoRA 与闭源 Judge 的版本化 Prompt
├── results/
│   ├── generation/<mllm>/{greedy,samples}/
│   ├── hidden/<mllm>/<dataset>/
│   ├── uq/<mllm>/
│   ├── judging/<mllm>/
│   ├── metrics/<mllm>/
│   ├── analysis/{descriptive,detection,luh,era}/
│   └── lora/<mllm>/adapter/
├── scripts/
│   ├── extract_per_model_subset.py     # 每模型 LUH 困难子集提取
│   ├── improvement/run_era.py          # ERA 特征提取入口
│   ├── analysis/                       # 实验一结果分析与 ERA 评估
│   ├── evaluation/compute_metrics.py
│   ├── generation/generate_responses.py
│   ├── judging/judge_responses.py
│   └── uq/compute_uq.py
├── src/                           # 数据集、模型、生成、Judge、UQ 和指标公共代码
│   └── improvement/               # ERA 改进方法核心实现
├── slurm/                         # LoRA、生成、UQ 与 ERA 作业入口
└── tests/
```

数据集与基础模型权重不提交到 Git。`results/` 同样被 `.gitignore` 忽略。

## 正式结果位置

正式流水线固定为一条 greedy 主回答、10 条 samples、每条 sample 最多 50 次 XML 拒绝重采样。每次实验只使用一种 Judge，Judge 具体实现与模型名称记录在 JSONL 的 `run` metadata 中，不作为目录层级。

| 产物 | 路径 |
| --- | --- |
| greedy 回答 | `results/generation/<mllm>/greedy/<dataset>.jsonl` |
| sampled 回答 | `results/generation/<mllm>/samples/<dataset>.jsonl` |
| UMPIRE hidden | `results/hidden/<mllm>/<dataset>/<sample-hash>.pt` |
| UQ 结果 | `results/uq/<mllm>/<dataset>.jsonl` |
| Judge 结果 | `results/judging/<mllm>/<dataset>.jsonl` |
| 指标报告 | `results/metrics/<mllm>/<dataset>.json` |
| LUH 子集 | `results/analysis/luh/` |
| 结果分析 | `results/analysis/{descriptive,detection,luh_profile}/` |
| LoRA adapter | `results/lora/<mllm>/adapter/` |

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
  --adapter-path results/lora/llava/adapter \
  --output results/generation/llava/greedy/vilp.jsonl \
  --phase greedy \
  --num-samples 0
```

生成严格分为两个互不读取对方结果的阶段：`--phase greedy --num-samples 0` 只写主回答，
`--phase samples --num-samples 10` 只写 10 条随机采样回答，每条 sample 的 XML 拒绝重采样
上限为 50。samples 阶段同时写入
每条采样答案末 token 的最后一层向量到 `results/hidden/<model>/<dataset>/`；不会写入
greedy 回答或读取 greedy 文件。
生成指令从 `prompts/generation/xml_lora_zero_shot.md` 显式加载；运行 JSONL 的
`run` 记录中保存 `prompt_sha256` 用于锁定实际使用的内容。

三模型 × 三数据集的批量提交也按阶段分开执行，不存在同时生成 greedy 与 samples 的入口：

```bash
GENERATION_PHASE=greedy bash slurm/generation/submit_generation_grid.sh
GENERATION_PHASE=samples bash slurm/generation/submit_generation_grid.sh
```

### 2. 计算 UQ

入口为 `scripts/uq/compute_uq.py`，在 Slurm 作业内显式读取 `--greedy-input` 和
`--sample-input` 以及本地 DeBERTa entailment 模型，计算三个 UQ 方法。Perplexity 只用
greedy 的最终答案 token 概率；Semantic Entropy 和 UMPIRE 只用 samples 的最终答案，UMPIRE 的向量从
`results/hidden/` 的 sidecar 读取。当前代码不支持旧的 combined JSONL 或
单独的 hidden manifest 输入，也不支持完整 XML 作为 UQ 输入。

全部 generation 结果就绪后，独立提交 3 × 3 UQ：

```bash
bash slurm/generation/submit_uq_grid.sh
```

### 3. Judge 与指标

`scripts/judging/judge_responses.py` 通过 OpenAI-compatible Responses API 调用多模态 Judge，
并且只接受 `--greedy-input`。服务配置仅从 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 环境变量读取。
闭源 Judge 的评分指令从
`prompts/judge/closed_source_judge.md` 加载；Judge JSONL 的
`run` 记录保存 `judge_prompt_sha256`。

`scripts/evaluation/compute_metrics.py` 将同一模型和数据集的 UQ/Judge JSONL 按 `sample_id` 合并，报告错误检测和幻觉检测的 AUROC、AUPRC、PRR 及 group-level bootstrap 置信区间。

`scripts/extract_per_model_subset.py` 为每个模型独立提取 400 样本低不确定性子集
（200 条 LUH 正例 + 200 条三维 baseline 百分位最近邻匹配的非幻觉负例），产物位于
`results/analysis/luh/`，提取定义与验证结果见
[低不确定性子集提取说明](docs/低不确定性子集提取说明.md)。

实验一结果分析入口为 `scripts/analysis/` 下的三个模块：`a_descriptive.py`（描述性统计）、
`b_detection.py`（检测性能对比，点估计与 `results/metrics/` 逐值核对）、`c1_luh_profile.py`
（LUH 画像与漏检归因）。产物位于 `results/analysis/{descriptive,detection,luh_profile}/`，
方案与字段审计见 [实验一结果分析](docs/实验一结果分析.md)。

### 4. 实验二：改进方法 ERA（Early Rationale Attribution）

针对三大传统 Baseline 在低不确定性幻觉（LUH）难例子集上失效（AUROC 仅接近 0.5）的问题，本项目提出了 **ERA（Early Rationale Attribution，早期推理归因）**：
- **核心机理**：单次前向解耦浅层解码器（Layer 0-1）注意力流向，量化答案决策对自身生成思考（$V+R$）相较于真实外部输入（$I+Q$）的相对依赖比率 $U_{\mathrm{ERA}}$。
- **评测表现**：在 400 条 LUH 难例子集上，LLaVA-1.5、Qwen2.5-VL 和 InternVL3.5 的 AUROC 分别达到 **0.7147**、**0.6003** 和 **0.5970**，显著超越所有传统 Baseline。
- **运行命令**：
  ```bash
  # 提取 ERA 5 桶注意力分量
  python scripts/improvement/run_era.py --greedy-input ... --output ...
  # 运行难例子集对比评估
  python scripts/analysis/evaluate_era.py
  ```
详情见 [ERA 早期推理归因不确定性量化方法](docs/ERA早期推理归因不确定性量化方法.md)。

## 下一步研究方向

在完成第一阶段主实验与第二阶段 ERA 改进方法的研究后，后续拟重点推进以下三个研究方向：

1. **全量基准与跨任务泛化评估（Full-Benchmark Generalization）**：
   - 将 ERA 从 400 条 LUH 难例子集向全量数据集（ViLP 900 样本、HallusionBench 1,129 样本、MM-Vet 218 样本）以及更多多模态基准（如 POPE、MME）拓展，系统评估其在全局幻觉预测上的鲁棒性。
2. **ERA 引导的自适应解码与幻觉主动抑制（ERA-Guided Decoding & Mitigation）**：
   - 将 ERA 从事后检测（Post-hoc Detection）推向生成期主动干预（In-generation Mitigation）。利用浅层注意力异常信号，设计动态注意力调节（Attention Steering）或候选响应自适应重排（Adaptive Re-ranking），直接在推理阶段阻断过度自依赖幻觉。
3. **实体与属性级细粒度归因（Fine-Grained Entity Attribution & Localization）**：
   - 进一步细化注意力在自生成上下文（$V$ 与 $R$）各实体、属性与关系 token 上的流动分布，将不确定性精准溯源到具体的幻觉发生点（视觉误识 vs 逻辑谬误）。


## LoRA adapter

三个正式 adapter 分别位于 `results/lora/<mllm>/adapter/`。训练配置、训练 loss、validation loss、adapter 权重和 tokenizer/processor 配置均与 adapter 一同保存。详情见 [LoRA README](LoRA/README.md)。

| 模型 | adapter 目录 | validation loss |
| --- | --- | ---: |
| LLaVA-1.5-7B | `results/lora/llava/adapter/` | 0.764681 |
| Qwen2.5-VL-7B-Instruct | `results/lora/qwen/adapter/` | 0.693499 |
| InternVL3.5-8B-HF | `results/lora/internvl/adapter/` | 0.633412 |

## 测试

测试依赖未随仓库固定安装。安装项目所需依赖后，可通过根目录 `pytest.ini` 直接运行第一阶段相关测试：

```bash
pytest
```

