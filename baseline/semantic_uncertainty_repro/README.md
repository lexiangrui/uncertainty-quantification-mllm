# Semantic Uncertainty Reproduction

复现 Farquhar et al. (Nature 2024) "Detecting Hallucinations in Large Language
Models Using Semantic Entropy"，适配多模态模型（LLaVA），与 `vauq-repro/` 并列
作为 baseline。

## 方法概述

1. 对每个问题采样 1 个低温度 + N 个高温度回答
2. 用蕴含模型 (DeBERTa) 对回答做双向语义蕴含判断，聚类
3. 计算语义熵 = 聚类分布的信息熵
4. 高语义熵 → 模型对含义不确定 → 可能幻觉

## 目录结构

```
semantic_uncertainty_repro/
├── scripts/run_semantic_uncertainty.py   ← 主入口
├── configs/sem_unc.env.example           ← 环境变量模板
├── slurm/                                ← Slurm 脚本
├── src/sem_unc/
│   ├── models/        ← Model ABC + LLaVA 实现
│   ├── datasets/      ← Dataset ABC + CV-Bench/MMVet/ViLP
│   ├── entailment.py  ← DeBERTa 蕴含判断
│   ├── semantic_entropy.py  ← 语义聚类 + 熵计算
│   ├── metrics.py     ← AUROC / AURAC 评估
│   ├── types.py       ← 数据类型
│   └── utils.py       ← 工具函数
└── requirements.txt
```

## 快速开始

### 1. 环境配置

```bash
cp configs/sem_unc.env.example configs/sem_unc.env
# 编辑 sem_unc.env 设置模型/数据集路径
source configs/sem_unc.env
pip install -r requirements.txt
```

### 2. 运行

```bash
# 最小示例（CV-Bench, 100 样本, 5 次采样）
python scripts/run_semantic_uncertainty.py \
    --model llava \
    --model-path /path/to/llava-1.5-7b-hf \
    --dataset cvbench \
    --num-samples 100 \
    --num-generations 5 \
    --output-dir results/sem_unc

# 跳过生成阶段（仅重新计算不确定性/分析）
python scripts/run_semantic_uncertainty.py \
    --skip-generate \
    --output-dir results/sem_unc \
    --tag llava_cvbench
```

### 3. 输出

- `results/sem_unc/<tag>.jsonl` — 每样本一行，含预测、正确性、分数
- `results/sem_unc/<tag>.summary.json` — AUROC/AURAC/accuracy 汇总

## 模型与数据集接口

### Model ABC

```python
class Model(ABC):
    def generate(self, image, question, temperature, max_new_tokens) -> tuple[str, list[float], Tensor | None]:
        """返回 (answer_text, token_log_likelihoods, last_token_embedding)"""
```

### Dataset ABC

```python
class Dataset(ABC):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict:
        """返回 {id, question, img, gt_ans, ...}"""
```

两套接口与 `vauq-repro` 的 Backend / Benchmark 对齐，后续可直接复用对方的实现。

## 与原论文的差异

| 方面 | 原论文 / 官方代码 | 本实现 |
|------|------------------|--------|
| 模型 | LLaMA-2, Falcon, Mistral (纯文本) | Mistral-7B-Instruct-v0.1 已验证；LLaVA 接口预留 |
| 数据集 | TriviaQA, SQuAD, BioASQ, NQ, SVAMP | TriviaQA 已验证；CV-Bench/MMVet/ViLP 接口预留 |
| 实验追踪 | wandb + pickle | JSONL + `.summary.json` |
| 蕴含模型 | DeBERTa / GPT-4 / LLaMA | DeBERTa |
| 架构 | 脚本式三阶段 | ABC + 注册表 + 单 CLI 三阶段 |
| 当前口径差异 | wandb run + pickle，可记录 train/validation 两阶段细节 | JSONL + summary；`official` 跑法已保存 high-temp answers、token log-likelihoods、semantic ids 和 prompt |

## 复现结果

### 集群运行记录

在 MiliLab 集群校外入口 `mg01-out` 上检查到两次结果。当前用于论文验证的是
`official-style` 作业 `34767`：

- 作业：`sem_unc_mistral_official`, job id `34767`
- 节点：`gpu04`, NVIDIA GeForce RTX 5090 32607 MiB
- 时间：2026-07-07 01:43:25 至 02:02:17 CST
- 模型：`/opt/lexiangrui/sem_unc_assets/models/Mistral-7B-Instruct-v0.1`
- 蕴含模型：`/opt/lexiangrui/sem_unc_assets/models/deberta-v2-xlarge-mnli`
- 配置：TriviaQA, official short-phrase prompt, 5 few-shot examples, official RNG order, 400 validation samples, 10 high-temperature generations, temperature 1.0, max_new_tokens 50, strict DeBERTa entailment, SQuAD-F1 correctness
- 远端日志：`~/semantic_uncertainty_repro/logs/sem_unc_mistral_official_34767.{out,err}`
- 远端结果：`~/semantic_uncertainty_repro/results/sem_unc/Mistral-7B-Instruct_trivia_qa_official.{jsonl,summary.json}`

本地保留的 summary 见 `results/sem_unc/Mistral-7B-Instruct_trivia_qa_official.summary.json`。
旧作业 `34601` 未使用 official few-shot prompt/索引口径，仅保留为对照。

### 指标

AUROC 越高越好；bootstrap 为 90% CI。

| 不确定性度量 | AUROC | 90% CI | AURAC |
|---|---:|---:|---:|
| Semantic Entropy | **0.7847** | [0.7486, 0.8227] | **0.6603** |
| Regular Entropy | 0.7387 | [0.6976, 0.7797] | 0.6330 |
| Cluster Assignment Entropy | 0.7689 | [0.7312, 0.8062] | 0.6505 |

基础准确率：`205 / 400 = 0.5125`。

按 semantic entropy 保留低不确定样本：

| 保留比例 | 准确率 | 提升 |
|---|---|---|
| 100% | 0.5125 | baseline |
| 95% | 0.5316 | +1.9 pp |
| 90% | 0.5556 | +4.3 pp |
| 80% | **0.6094** | **+9.7 pp** |

与旧作业 `34601` 对比：

| 作业 | Prompt/索引口径 | Accuracy | Semantic Entropy AUROC |
|---|---|---:|---:|
| `34601` | 非 official prompt/索引 | 0.7200 | 0.7772 |
| `34767` | official-style prompt/索引 | 0.5125 | **0.7847** |

### 结论

- 当前结果复现了论文的核心趋势：semantic entropy 的 AUROC 高于 regular entropy，且拒绝高不确定样本后准确率明显上升。
- `Mistral + TriviaQA` 的 semantic entropy AUROC 为 `0.7847`，接近论文报告的跨模型/跨数据集总体水平约 `0.79`；但原文/官方仓库没有公开这个组合的单点表，因此不能过度宣称严格逐点一致。
- 当前 JSONL 已保存 high-temperature generations、token log-likelihoods、semantic ids 和 prompt，可复核 semantic entropy 计算。

仍会影响结果的差异：

- 当前只跑单 seed、400 validation samples；论文主结果是多模型/多数据集汇总，不提供该组合单点。
- correctness 使用本仓库的 SQuAD-F1 阈值实现，细节若与官方 `evaluate` 版本有差异，会改变 accuracy、AURAC 和 AUROC 标签。
- 模型权重路径为集群本地 `Mistral-7B-Instruct-v0.1`，需要继续记录权重 revision/hash 才能排除模型版本漂移。

## 下一步优化方案

1. **固定资产版本**：把 Mistral、DeBERTa、TriviaQA cache 的 revision/hash 写入 summary metadata。
2. **补官方代码同组合运行**：用 `/tmp/semantic_uncertainty_official` 的原始三阶段脚本跑同一模型/数据，验证本仓库 official-style 模式与官方实现的差异。
3. **扩大 seed 检查**：补 3 个 seed，报告均值和方差，判断 `0.7847` 的稳定性。
4. **增加审计脚本**：检查 JSONL 行数、字段完整性、summary 是否可由 JSONL 复算、AUROC/AURAC 是否一致。
5. **再比较 VL 验证结果**：远端 `~/vl_uncertainty_repro/results/verify_mistral_triviaqa.summary.json` 显示另一个验证流程的 semantic entropy AUROC 为 0.8097，可作为后续跨实现一致性检查对象。

## 论文引用

```
@article{farquhar2024detecting,
  title={Detecting hallucinations in large language models using semantic entropy},
  author={Farquhar, Sebastian and Kossen, Jannik and Kuhn, Lorenz and Gal, Yarin},
  journal={Nature},
  volume={630},
  number={8017},
  pages={625--630},
  year={2024},
  publisher={Nature Publishing Group}
}
```
