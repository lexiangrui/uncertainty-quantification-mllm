# 三篇论文复现结果汇总

## 1. Semantic Entropy

> **Farquhar et al. (2024)** — Detecting Hallucinations in Large Language Models Using Semantic Entropy
>
> Paper: https://www.nature.com/articles/s41586-024-07421-0 | Code: https://github.com/jlko/semantic_uncertainty

## 方法简介

Semantic Entropy 用多次高温采样估计模型对“语义”的不确定性。它不直接把不同文本字符串当成不同答案，而是先用双向 entailment 把表达不同但含义相同的回答聚成语义簇，再在语义簇分布上计算熵。语义簇越多、概率质量越分散，说明模型对答案含义越不确定，更可能出现 hallucination。

1. 对同一个问题构造 short-phrase prompt，并用 5 个 few-shot QA 示例固定回答风格。
2. 低温/近似贪心生成 1 个主回答，作为被评估的模型预测；再用标准答案判断这个主回答是否正确，得到 correctness 标签。
3. 高温 (t=1.0) 采样 10 个候选回答，并记录每个回答的 token log-likelihood。
4. 对每个候选回答计算长度归一化 log-likelihood，作为该文本答案的概率估计。
5. 用 DeBERTa 做双向 entailment 判断，把含义相同的候选回答聚成 semantic cluster。
6. 对同一语义簇内的回答概率做 log-sum-exp 聚合，得到每个语义簇的概率质量。
7. 计算 semantic entropy：`H = -Σ p_c log(p_c)`。
8. 语义熵越高 → 答案含义越分散 → hallucination 风险越高；同时用 AUROC/AURAC 评估该分数区分正确与错误回答的能力。

![image-20260707101637202](/Users/lexiangrui/Library/Application Support/typora-user-images/image-20260707101637202.png)

![image-20260707102728692](/Users/lexiangrui/Library/Application Support/typora-user-images/image-20260707102728692.png)

## 与原论文的差异

| 方面     | 原论文 / 官方代码                  | 本实现                   |
| -------- | ---------------------------------- | ------------------------ |
| 模型     | LLaMA-2, Falcon, Mistral (纯文本)  | Mistral-7B-Instruct-v0.1 |
| 数据集   | TriviaQA, SQuAD, BioASQ, NQ, SVAMP | TriviaQA                 |
| 蕴含模型 | DeBERTa / GPT-4 / LLaMA            | DeBERTa（NLI）           |

### 复现结果

在 MiliLab 集群校外入口 `mg01-out` 运行。

- 作业：`sem_unc_mistral_official`, job id `34767`
- 节点：`gpu04`, NVIDIA GeForce RTX 5090 32607 MiB
- 时间：2026-07-07 01:43:25 至 02:02:17 CST
- 模型：`/opt/lexiangrui/sem_unc_assets/models/Mistral-7B-Instruct-v0.1`
- 蕴含模型：`/opt/lexiangrui/sem_unc_assets/models/deberta-v2-xlarge-mnli`
- 配置：TriviaQA, 400 samples, 10 high-temperature generations, temperature 1.0, max_new_tokens 50, strict DeBERTa entailment, SQuAD-F1 correctness

AUROC 越高越好；bootstrap 为 90% CI。

| 不确定性度量               |      AUROC |           90% CI |      AURAC |
| -------------------------- | ---------: | ---------------: | ---------: |
| Semantic Entropy           | **0.7847** | [0.7486, 0.8227] | **0.6603** |
| Regular Entropy            |     0.7387 | [0.6976, 0.7797] |     0.6330 |
| Cluster Assignment Entropy |     0.7689 | [0.7312, 0.8062] |     0.6505 |

### 与原文结果对比

**数据来源**：原文未公布具体数据，只有数据柱状图。在Supplementary Information 中有 short-phrase setting 下 SE 平均 AUROC 为 **0.792**，但是多个模型多个数据集平均的结果。

| 不确定性度量     | 复现 AUROC | 原文平均 AUROC |   差值 |
| ---------------- | ---------: | -------------: | -----: |
| Semantic Entropy |     0.7847 |          ~0.79 | ~0.005 |
| Regular Entropy  |     0.7387 |              — |      — |

<img src="/Users/lexiangrui/Library/Application Support/typora-user-images/image-20260707133209874.png" alt="image-20260707133209874" style="zoom: 50%;" />

## 2. VL-Uncertainty

> **Zhang et al. (2024)** — VL-Uncertainty: Detecting Hallucination in Large Vision-Language Model via Uncertainty Estimation
>
> Paper: https://arxiv.org/abs/2411.11919 | Code: https://github.com/Ruiyang-061X/VL-Uncertainty

## 方法简介

VL-Uncertainty 面向 LVLM 幻觉检测。它对同一个图文输入构造语义等价扰动：图像端使用不同强度的 Gaussian blur，文本端用 LLM 改写问题但保持语义不变。模型在这些扰动输入上的回答如果语义稳定，说明不确定性低；如果回答语义分散，则聚类熵高，更可能是 hallucination。

1. 原始图文输入低温 (t=0.1) 生成 1 次，得到 `prediction`。
2. 图像用 **Gaussian blur** 半径 `[0.6, 0.8, 1.0, 1.2, 1.4]` 生成 5 个扰动。
3. 问题用 **Qwen2.5-3B-Instruct** 在温度 `[0.1, 0.2, 0.3, 0.4, 0.5]` 下生成 5 个语义等价改写。
4. 将第 i 个图像扰动和第 i 个文本扰动 progressively 配对 → 5 个扰动 prompt。
5. 每个扰动 prompt 高温 (t=1.0) 采样 1 次 → 5 个回答。
6. **Free-form**: LLM 双向 entailment 聚类（"Does A entail B? Yes/No"）。
   **Multi-choice**: 按选项编号聚类。
7. 计算计数 Shannon 熵: `H = -Σ (count_c/N) log₂(count_c/N)`。
8. 阈值 1.0：熵 >= 1.0 → hallucination。

![image-20260707102605208](/Users/lexiangrui/Library/Application Support/typora-user-images/image-20260707102605208.png)  

### 与原论文的差异

| 方面 | 原论文 / 官方代码 | 本次复现 | 是否影响结果 |
| ---- | ----------------- | -------- | ------------ |
| 模型范围 | 10 个 LVLM：Qwen2VL-2B/7B/72B、LLaVA1.5-7B/13B、InternVL2-1B/8B/26B、LLaVANeXT-7B/13B | 只复现 LLaVA1.5-7B；文本扰动/聚类/judge 使用 Qwen2.5-3B-Instruct | 影响覆盖范围；当前只能和原文 LLaVA1.5-7B 的 MM-Vet 单项对比 |
| 数据集范围 | 4 个 benchmark：MM-Vet、LLaVABench、MMMU、ScienceQA；包含 free-form 和 multi-choice | 只复现 MM-Vet free-form，全量 218 样本 | 影响覆盖范围；LLaVABench/MMMU/ScienceQA 未验证 |
| 报告指标 | Table 1 只报告 hallucination detection accuracy | 额外报告 AUROC/AUPR | AUROC/AUPR 无原文逐项对照 |

### 复现结果

在 MiliLab 集群校外入口 `mg01-out` 运行。

- 作业：`vlu-mmvet-llava`, job id `34784`
- 节点：`gpu03`, 2× NVIDIA GeForce RTX 5090 (LLaVA → GPU0, Qwen → GPU1)
- 时间：2026-07-07 02:45:56 至 03:06:34 CST
- 模型：`/opt/lexiangrui/vauq_assets/models/llava-1.5-7b-hf`
- 文本扰动/聚类模型：`/opt/lexiangrui/vauq_assets/models/Qwen2.5-3B-Instruct`
- 配置：MM-Vet 全量 218 样本，视觉 blur `[0.6, 0.8, 1.0, 1.2, 1.4]`，文本 LLM rephrasing temps `[0.1, 0.2, 0.3, 0.4, 0.5]`，sampling temp 1.0，max_new_tokens 128，LLM judge

| 指标                             |       数值 |
| -------------------------------- | ---------: |
| Hallucination Detection Accuracy | **79.82%** |
| LVLM Accuracy (原始回答正确率)   |     26.15% |
| VL-Uncertainty AUROC             |     0.7881 |
| VL-Uncertainty AUPR              |     0.5785 |

### 与原文结果对比

**数据来源**：原文 Table 1 (Main Results, MM-Vet free-form, LLaVA1.5-7B)。原文仅报告 Hallucination Detection Accuracy，未报告 AUROC/AUPR。

| 指标                             |       复现 | 原文 Table 1 |     差值 |
| -------------------------------- | ---------: | -----------: | -------: |
| Hallucination Detection Accuracy | **79.82%** |   **82.11%** | -2.29 pp |

## 3. VAUQ

> **Park et al. (2026)** — VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation
>
> Paper: https://arxiv.org/abs/2602.21054 | Code: https://github.com/deeplearning-wisc/vauq

## 方法简介

VAUQ 直接利用白盒 LVLM 的 token entropy 和视觉 attention 信号评估回答可靠性。方法先计算原图条件下答案的预测熵，再遮挡 attention 选出的核心视觉 token，观察遮挡后熵的变化，得到 Image-Information Score。若回答真正依赖图像核心证据，遮挡会显著增加不确定性；若模型主要靠语言先验，即使遮挡图像仍很自信，则 hallucination 风险更高。

1. 对图文问题使用 greedy decoding 生成 1 个回答，并保留生成 token ids。
2. 在原始图像和问题条件下重新前向计算生成 token 的 logits。
3. 基于 token logits 计算原始预测熵：`H(y | v, t)`。
4. 读取模型视觉 attention，选出 top-K 最重要的视觉 token 作为 core visual evidence。
5. 遮挡这些核心视觉 token，再次计算同一回答 token 的 logits。
6. 计算遮挡后的预测熵：`H(y | v_masked, t)`。
7. 得到 Image-Information Score：`IS_core = H(y | v_masked, t) - H(y | v, t)`。
8. 组合成 VAUQ 分数：`s_VAUQ = H(y | v, t) - alpha * IS_core`；分数越高表示回答越可疑，更可能错误。

核心公式：

```text
IS_core = H(y | v_masked, t) - H(y | v, t)
s_VAUQ  = H(y | v, t) - alpha * IS_core
```

### 论文核心说明图

![image-20260707102917271](/Users/lexiangrui/Library/Application Support/typora-user-images/image-20260707102917271.png)

### 与原论文的差异

| 方面 | 原论文 | 本次复现 | 是否影响结果 |
| ---- | ------ | -------- | ------------ |
| 模型范围 | Table 1 报告 LLaVA-1.5-7B/13B、Qwen、InternVL 等模型 | 只复现 LLaVA-1.5-7B/13B | 影响覆盖范围，不影响已跑组合的单项对比 |
| 数据集范围 | ViLP、MMVet、CVBench、VisualCoT | ViLP、MMVet、CVBench；未跑 VisualCoT | VisualCoT 无法对比 |
| 生成设置 | Appendix A 写 greedy decoding，`max_new_tokens=128` | 已按论文设置：greedy decoding，`max_new_tokens=128` | 已对齐 |
| 随机种子 | Appendix A 写三次随机种子平均 | 当前为单次运行 | 会影响统计稳定性 |
| ViLP/MMVet correctness | 论文使用 GPT-5 三次多数投票标注 free-form 回答正确性 | 使用集群本地 `Qwen2.5-3B-Instruct` 作为开源替代 judge | 会直接影响 free-form 数据集 AUROC/AUPR |
| 超参 | 论文 Appendix F 给出各模型/数据集的 `alpha` 和 `K` | 按 Appendix F 设置运行 | 已按论文对齐 |
| 官方代码口径 | 官方 GitHub 默认生成设置、CVBench 加载和部分超参与论文不一致 | 本次优先按论文而不是官方默认代码 | —— |

### 当前复现结果

在 MiliLab 集群校外入口 `mg01-out` 运行。

- 主作业：`vauq-paper-grid`, job id `34769`，array `0-5`，已完成
- Qwen 判分作业：`vauq-qwen-judge`, job id `34770`，array `0-3`，已完成
- 模型：`/opt/lexiangrui/vauq_assets/models/llava-1.5-7b-hf` 与 `/opt/lexiangrui/vauq_assets/models/llava-1.5-13b-hf`
- 数据集：ViLP 600 条、MMVet 218 条、CVBench 2D+3D 全集 2638 条
- 配置：greedy decoding，`max_new_tokens=128`，core visual token masking，Appendix F 超参
- 判分：CVBench 使用字母正则匹配；ViLP/MMVet 使用本地 `Qwen2.5-3B-Instruct` 替代论文 GPT-5 多数投票 judge
- 远端结果：`/home/lexiangrui/vauq-repro/results/*paper_v2*.summary.json`

AUROC/AUPR 越高越好；下表 AUROC 为 VAUQ 主分数。

| 模型 | 数据集 | 样本数 | Accuracy | VAUQ AUROC | VAUQ AUPR | Entropy AUROC | IS AUROC |
| ---- | ------ | -----: | -------: | ---------: | --------: | ------------: | -------: |
| LLaVA-1.5-7B | ViLP | 600 | 45.17% | 63.59 | 56.26 | 67.06 | 53.44 |
| LLaVA-1.5-7B | MMVet | 218 | 22.94% | 80.75 | 56.44 | 81.25 | 62.05 |
| LLaVA-1.5-7B | CVBench | 2638 | 61.71% | 68.77 | 78.85 | 68.74 | 64.09 |
| LLaVA-1.5-13B | ViLP | 600 | 44.50% | 65.45 | 54.89 | 68.63 | 51.38 |
| LLaVA-1.5-13B | MMVet | 218 | 28.44% | 84.96 | 68.14 | 86.22 | 60.96 |
| LLaVA-1.5-13B | CVBench | 2638 | 61.14% | 65.47 | 74.86 | 65.87 | 52.66 |

### 与原文结果对比

**数据来源**：原文 Table 1 的 VAUQ AUROC。当前复现只跑 LLaVA-1.5-7B/13B × ViLP/MMVet/CVBench；VisualCoT 未纳入本次复现。

| 模型 | 数据集 | 复现 VAUQ AUROC | 原文 Table 1 | 差值 |
| ---- | ------ | --------------: | -----------: | ---: |
| LLaVA-1.5-7B | ViLP | 63.59 | 77.00 | -13.41 |
| LLaVA-1.5-7B | MMVet | 80.75 | 81.50 | -0.75 |
| LLaVA-1.5-7B | CVBench | 68.77 | 73.20 | -4.43 |
| LLaVA-1.5-13B | ViLP | 65.45 | 69.50 | -4.05 |
| LLaVA-1.5-13B | MMVet | 84.96 | 88.60 | -3.64 |
| LLaVA-1.5-13B | CVBench | 65.47 | 68.30 | -2.83 |
| LLaVA-1.5-7B/13B | VisualCoT | 未复现 | 77.80 / 80.20 | — |

当前最可能影响对比的因素是 free-form correctness 标签：论文使用 GPT-5 三次多数投票，本复现使用集群本地 Qwen2.5-3B-Instruct；这会直接改变 AUROC 的正负样本标签。另一个统计口径差异是论文 Appendix A 写结果 averaged over three random seeds，而当前为单次运行。

### 原文与官方代码问题

- **生成设置不一致**：论文 Appendix A 写 greedy decoding、最大生成长度 128；官方代码当前实现为 `do_sample=True, temperature=0.1, max_new_tokens=64`，且 LLaVA wrapper 里曾硬编码 64。本次复现按论文设置跑 greedy + 128。
- **CVBench 数据范围不一致**：论文数据集说明写 CVBench 共 2638 条，对应 2D+3D 全集；官方代码 `benchmark/cvbench.py` 只加载 2D 子集。本次复现跑 2D+3D 全集。
- **超参不一致**：论文 Appendix F 的 LLaVA-1.5-7B 超参与官方 GitHub 默认值不同，例如 ViLP、CVBench 的 `alpha` 和 `K/topk_ratio` 不一致。本次复现按论文 Appendix F。
- **官方代码覆盖不完整**：官方仓库主要内置 LLaVA-1.5-7B 默认超参，benchmark 目录没有 VisualCoT loader；但论文主表报告 LLaVA-1.5-13B、Qwen、InternVL 和 VisualCoT。
- **标签不可完全复现**：官方仓库不发布论文使用的 answer cache / GPT-5 三次多数投票标签，只说明 AUROC/AUPR 需要外部提供 `label`。因此 ViLP/MMVet 这类 free-form 数据集无法仅凭官方仓库完全复现论文标签。



问题总结：

1、目前自己的复现达不到原论文的数据，后续baseline是用自己跑的数据还是原文数据。若原文没有在我想跑的数据集上的结果该怎么办？

2、VAUQ方法与自己的idea相似，我是否要另想其他方法

3、VAUQ的数据真实性存在疑问，并且较新，我是否要将其作为baseline

4、我构想的方法只适用于开源模型，考核任务中要求需要覆盖到闭源模型
