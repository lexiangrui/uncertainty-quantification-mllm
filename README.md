# Uncertainty Quantification of MLLM

本项目研究多模态大模型（MLLM）的不确定性量化与高置信度幻觉检测，整体分为两个阶段：

1. 在 ViLP、HallusionBench 和 MM-Vet 上生成回答、计算 baseline UQ 分数并使用多模态 LLM Judge 标注幻觉，提取“模型置信度高但实际存在幻觉”的样本子集；
2. 在固定子集上评测 MALP、GASP 等改进的不确定性量化方法，并与 baseline 进行统一比较。

工程方案见 [docs/工程实现.md](docs/工程实现.md)，实验定义见 [docs/高置信度幻觉子集提取实验设计.md](docs/高置信度幻觉子集提取实验设计.md)。

## 被测对象

### 模型

- LLaVA-1.5-7B
- Qwen2.5-VL-7B-Instruct
- InternVL3.5-8B-HF

三个模型均通过独立 LoRA adapter 学习固定的单行 XML 回答协议：

```xml
<vision>...</vision><reasoning>...</reasoning><answer>...</answer>
```

### 数据集

- ViLP：900 个 QIA 样本
- HallusionBench：1129 条样本
- MM-Vet：218 条样本

数据集和模型权重不进入 Git 仓库，统一保存在服务器 `/opt/lexiangrui/` 下；训练结果和正式实验产物写入 `/home/lexiangrui/results/`。

### 当前接入的 baseline UQ

- Perplexity
- Semantic Entropy
- UMPIRE

UQ 在模型完成当前问题的回答后、内部状态释放前计算。程序只保存回答、最终分数和必要审计信息，不持久化 logits、hidden states、attention 或 KV cache。

## 目录结构

```text
.
├── docs/
├── configs/
├── prompts/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── generation/
│   ├── llm_judge/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   ├── generation/
│   ├── judging/
│   ├── subset/
│   └── evaluation/
├── LoRA/
├── baseline/
├── MALP/
├── GASP/
├── slurm/
└── tests/
```

- `src/`：跨数据集、模型和实验阶段共享的公共能力，不放具体 UQ 方法。
- `baseline/`：复现或接入的 baseline UQ 方法，每种方法保持独立目录。
- `LoRA/`：VQAv2 XML 数据构造与三模型格式 LoRA 训练。
- `MALP/`、`GASP/`：待在固定幻觉子集上评测的改进方法。
- `scripts/`：生成、Judge、子集提取和评估的命令入口。
- `slurm/`：服务器计算节点作业入口。

## 已实现的主工作流

### 1. XML 格式 LoRA

当前流程为：

```text
VQAv2 候选筛选
→ Qwen3.7-Plus 教师生成 vision/reasoning/answer
→ 本地严格校验并生成单行 XML
→ 1600/200/200 划分
→ 分别训练三个模型的 LoRA adapter
```

训练器保存每个 optimizer update 的平均训练 loss、每个 epoch 的 validation loss、adapter 和实际训练配置。详细使用方法见 [LoRA/README.md](LoRA/README.md)。

当前正式训练状态：

- LLaVA 正式训练完成，最终 validation loss 为 `0.6637526`；
- Qwen2.5 与 InternVL 已通过真实权重 smoke，正式 Slurm 作业尚在队列中。

### 2. 回答生成与在线 UQ

公共入口：

```text
scripts/generation/generate_responses.py
slurm/generation/generate_responses.sbatch
```

生成阶段为每个样本生成：

- 1 条 greedy 主回答；
- 10 条 `temperature=1.0` 随机采样回答；
- 回答 token 概率统计；
- sampling hidden states，保存到 JSONL 同名的 `.hidden/` 目录中的 `.pt` sidecar。

生成入口不加载 DeBERTa，也不在线计算 UQ。它强制加载 LoRA adapter，严格解析 XML，支持 JSONL + `.pt` 断点续跑，并要求计算节点处于 Hugging Face 离线模式。服务器示例：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/generation/generate_responses.py \
  --dataset vilp \
  --dataset-source /opt/lexiangrui/datasets/vilp \
  --model-family llava_1_5 \
  --model-path /opt/lexiangrui/models/llava-1.5-7b-hf \
  --adapter-path /home/lexiangrui/results/lora/official_inline/llava-1.5-7b \
  --output /home/lexiangrui/results/subset_extraction/llava_1_5_7b/vilp.jsonl
```

生成完成后独立计算 UQ：

```bash
python scripts/uq/compute_uq.py \
  --generation-input /home/lexiangrui/results/subset_extraction/llava_1_5_7b/vilp.jsonl \
  --output /home/lexiangrui/results/uq/llava_1_5_7b/vilp.uq.jsonl \
  --entailment-model-path /opt/lexiangrui/sem_unc_assets/models/deberta-v2-xlarge-mnli
```

路径应以服务器上的实际目录为准。模型、数据和依赖必须提前在允许联网的登录或调试节点准备；正式生成在 Slurm 计算节点离线执行。

### 3. OpenAI-compatible 多模态 Judge

入口：

```text
scripts/judging/judge_responses.py
```

Judge 独立读取生成结果和原图，一次请求同时返回答案正确性及 MMHal 风格幻觉评分。服务配置只从环境变量读取：

```bash
export OPENAI_BASE_URL=
export OPENAI_API_KEY=

python scripts/judging/judge_responses.py \
  --dataset vilp \
  --dataset-source /opt/lexiangrui/datasets/vilp \
  --generation-input /home/lexiangrui/results/subset_extraction/llava_1_5_7b/vilp.jsonl \
  --output /home/lexiangrui/results/judging/llava_1_5_7b/vilp.jsonl \
  --model YOUR_JUDGE_MODEL
```

仓库不保存 API Key、私有服务地址或其他凭据。

### 4. 统一指标计算

入口：

```text
scripts/evaluation/compute_metrics.py
```

对同一“模型 × 数据集”的 UQ 输出与 Judge 输出按 `sample_id` 合并；两份输入的 `generation_run` 必须一致，否则拒绝计算。质量控制会排除 Judge 无效、缺少对应 UQ 记录或任一方法分数无效/非有限的样本，并逐类报告排除数量，不做插补。

在合并后的分析集上按两个二元目标分别评估每种 UQ 方法：

- 错误检测：正类 `E = 1 - correct`（主要评估）；
- 幻觉检测：正类 `H = hallucination`（次要评估）。

每个“目标 × 方法”报告 AUROC（并列分数按 1/2 计）、AUPRC、PRR 和 ECE（无标签 min–max 归一化 + 15 等宽分箱），并报告 Accuracy、Hallucination Rate 与 \(C\times H\) 四格计数。所有点估计附带以原始问题 `group_id` 为聚类单位重采样的 95% bootstrap 置信区间；只含单一类别的目标将 AUROC/AUPRC/PRR 记为 N/A。结果写入单个 JSON 报告并在终端打印汇总表：

```bash
python scripts/evaluation/compute_metrics.py \
  --uq-input /home/lexiangrui/results/uq/llava_1_5_7b/vilp.uq.jsonl \
  --judge-input /home/lexiangrui/results/judging/llava_1_5_7b/vilp.jsonl \
  --output /home/lexiangrui/results/metrics/llava_1_5_7b/vilp.metrics.json
```

指标计算只依赖 numpy，900 样本、1000 次 bootstrap 数秒完成，可直接在登录节点运行，不需要 GPU 或 Slurm。

## 当前完成度

| 工作流 | 代码 | 真实验证 | 正式结果 |
| --- | --- | --- | --- |
| VQAv2 → XML 数据 | 完成 | 完成 | 2000 条完成 |
| 三模型 LoRA | 完成 | 三模型 smoke 完成 | LLaVA 完成；另外两模型排队 |
| 三数据集适配 | 完成 | 离线适配验证完成 | 尚未全量生成 |
| 回答生成 + 三种 UQ | 完成 | 既有链路部分验证 | 正式 `3×3` 矩阵待跑 |
| OpenAI Chat Judge | 完成 | 单元/API 测试完成 | 待处理正式回答 |
| 高置信幻觉子集提取 | 待实现 | — | — |
| 统一指标（AUROC/AUPRC/PRR/ECE + bootstrap CI） | 完成 | 单元测试与暴力实现交叉验证完成 | 待处理正式结果 |
| MALP/GASP 子集评测 | 待接入 | — | — |

Smoke 或单元测试通过不代表正式全量实验已经完成。

## 测试

当前公共主链、LoRA、三个已接入 UQ、统一指标和 GASP 核心测试可执行：

```bash
pytest -q \
  tests \
  LoRA/tests \
  baseline/perplexity_repro/tests \
  baseline/semantic_uncertainty_repro/tests \
  baseline/umpire_repro/tests \
  GASP/tests \
  --ignore=LoRA/tests/test_reject_resample.py
```

当前结果为 `107 passed`。`LoRA/tests/test_reject_resample.py` 引用的 `lora_format.reject_resample` 模块尚未加入仓库，需暂时排除。根目录直接运行无范围的 `pytest` 尚不能作为验收命令：MALP 和部分旧 baseline 仍存在旧 `judge` 引用、独立 `PYTHONPATH` 要求以及同名顶层模块导入冲突。

## 后续工作

1. 完成并验收 Qwen2.5、InternVL 正式 LoRA；
2. 实现三模型 held-out XML 格式统一评测入口；
3. 对三个模型和三个数据集做少量端到端生成/UQ 验证；
4. 运行正式 `3 模型 × 3 数据集` 生成与 UQ；
5. 对 greedy 主回答运行 Judge；
6. 实现结果合并、高置信幻觉子集提取和敏感性分析；
7. 对正式 `3 模型 × 3 数据集` 的 UQ 与 Judge 输出运行统一指标并汇总报告；
8. 将 MALP、GASP 接入固定子集并与 baseline 比较。
