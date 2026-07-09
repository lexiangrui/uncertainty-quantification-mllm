# 设计文档：Semantic Uncertainty 复现

**日期**: 2026-07-06
**来源论文**: "Detecting Hallucinations in Large Language Models Using Semantic Entropy" (Farquhar et al., Nature 2024)
**上游仓库**: https://github.com/jlko/semantic_uncertainty
**目标**: 在 `semantic_uncertainty_repro/` 中复现语义不确定性方法，与 `vauq-repro/` 并列作为 baseline

---

## 1. 架构概览

采用与 vauq-repro 一致的 ABC + 注册表 + 工厂函数模式。纯文本 LLM 路线，JSONL 本地输出，环境变量配置。

```
scripts/run_semantic_uncertainty.py  ← 主入口 CLI（三级流水线）
src/sem_unc/
├── models/     ← Model ABC, HuggingFace LLM 实现
├── datasets/   ← Dataset ABC, 5 个 QA 数据集实现
├── entailment.py        ← 蕴含判断模型（DeBERTa 为主）
├── semantic_entropy.py  ← 语义聚类 + 语义熵计算
├── metrics.py           ← AUROC/AURAC/accuracy@quantile
├── types.py             ← 数据类
└── utils.py             ← prompt 构建、序列化
```

## 2. 核心 ABC 接口

### 2.1 Model ABC

```python
class Model(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float) -> tuple[str, list[float], Tensor | None]:
        """返回 (answer_text, token_log_likelihoods, last_token_embedding)"""
```

实现：`HuggingFaceLLM` — 统一加载 LLaMA-2 / Falcon / Mistral，返回生成文本 + token 对数似然 + 最后 token 的隐藏状态。

### 2.2 Dataset ABC

```python
class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """返回 {id, question, context, answers: {text: [...]}}"""
```

实现：QA 数据集的统一包装（TriviaQA, SQuAD, BioASQ, NQ, SVAMP），通过 HuggingFace datasets 加载。

## 3. 三级流水线

`run_semantic_uncertainty.py` 按阶段执行：

1. **generate**：对每个问题采样 1 个低温度 (0.1) + N 个高温度回答，收集 token likelihood 和 embedding
2. **compute**：用蕴含模型 (DeBERTa) 做双向蕴含判断 → 语义聚类 → 计算 semantic_entropy / regular_entropy / cluster_assignment_entropy
3. **analyze**：用 token-level 的 correctness 标签计算 AUROC / AURAC / accuracy@quantile，写入 `.summary.json`

## 4. 关键设计决策

| 方面 | 决策 |
|------|------|
| 模型范围 | 纯文本 LLM (LLaMA-2, Falcon, Mistral)，不扩展多模态 |
| 实验追踪 | JSONL + 本地 `.summary.json`，无 wandb 依赖 |
| 配置方式 | 环境变量 (`.env` 文件) + argparse 覆盖 |
| 蕴含模型 | DeBERTa 为主，GPT/LLaMA 保留接口 |
| p_ik / p_true | 保留，可选开关 |
| 打分指标 | SQuAD F1 ≥ 50 为主，LLM judge (GPT) 为可选 |

## 5. 输出格式（与 vauq-repro 对齐）

每样本一行 JSONL + 末尾 `.summary.json`：

```json
{
  "id": "abc123",
  "question": "What is...",
  "gt_answers": ["Paris"],
  "prediction": "Paris",
  "correct": true,
  "scores": {
    "semantic_entropy": 0.42,
    "regular_entropy": 0.38,
    "cluster_assignment_entropy": 0.45
  }
}
```

## 6. 与原论文的差异

| 方面 | 原论文 | 本实现 |
|------|--------|--------|
| 实验追踪 | wandb | JSONL + .summary.json |
| 中间状态 | pickle (wandb 托管) | 本地 JSON |
| 模型实现 | 分散在 HuggingfaceModel 中 | ABC + 注册表模式 |
| 数据集实现 | 函数式加载 | ABC + 注册表模式 |
| 流水线控制 | 脚本间 pickle 传递 | 单 CLI 三阶段 |

## 7. 自检

- 无 TBD / TODO
- 接口定义明确，无歧义
- 范围聚焦：纯文本、DeBERTa 为主、JSONL 输出
- 与 vauq-repro 结构对齐，便于后续对比
