# VL-Uncertainty 复现

> **Zhang et al. (2024)** — VL-Uncertainty: Detecting Hallucination in Large Vision-Language Model via Uncertainty Estimation
>
> Paper: https://arxiv.org/abs/2411.11919 | Code: https://github.com/Ruiyang-061X/VL-Uncertainty

## 方法

VL-Uncertainty 通过语义等价扰动 + 答案聚类熵检测 LVLM 幻觉：

1. 原始图文输入低温 (t=0.1) 生成 1 次，得到 `prediction`。
2. 图像用 **Gaussian blur** 半径 `[0.6, 0.8, 1.0, 1.2, 1.4]` 生成 5 个扰动。
3. 问题用 **Qwen2.5-3B-Instruct** 在温度 `[0.1, 0.2, 0.3, 0.4, 0.5]` 下生成 5 个语义等价改写。
4. 将第 i 个图像扰动和第 i 个文本扰动 progressively 配对 → 5 个扰动 prompt。
5. 每个扰动 prompt 高温 (t=1.0) 采样 1 次 → 5 个回答。
6. **Free-form**: LLM 双向 entailment 聚类（"Does A entail B? Yes/No"）。
   **Multi-choice**: 按选项编号聚类。
7. 计算计数 Shannon 熵: `H = -Σ (count_c/N) log₂(count_c/N)`。
8. 阈值 1.0：熵 >= 1.0 → hallucination。

## 运行

```bash
cd vl_uncertainty_repro
cp configs/vl_uncertainty.env.example configs/vl_uncertainty.env
# 编辑 env 文件设置模型路径

source configs/vl_uncertainty.env
"${PYTHON_BIN:-python3}" scripts/run_vl_uncertainty.py \
  --backend llava \
  --benchmark mmvet \
  --text-model qwen \
  --model-path "$VLU_MODEL_PATH" \
  --text-model-path "$VLU_TEXT_MODEL_PATH" \
  --limit 218 \
  --output results/mmvet_llava_vlu.jsonl
```

### 集群提交

```bash
sbatch slurm/run_vl_uncertainty_mmvet.sbatch
```

### 冒烟测试

```bash
"${PYTHON_BIN:-python3}" scripts/run_vl_uncertainty.py \
  --backend mock --benchmark toy --text-model echo --judge choice --limit 2
```

## 预期结果

**MM-Vet + LLaVA1.5-7B**（原文 Table 1）:

| 指标 | 原文 |
|---|---|
| Hallucination Detection Accuracy | **82.11%** |

## 复现结果（2026-07-07）

MiliLab 集群校外入口 `mg01-out` 检查结果：

| 作业 | 状态 | 说明 |
|---|---|---|
| `34768` `vlu-mmvet-llava` | FAILED | `HF_DATASETS_OFFLINE=1` 下无法从 Hub 访问 `whyu/mm-vet`，说明 Slurm 作业未命中本地缓存/数据源配置。 |
| `34784` `vlu-mmvet-llava` | COMPLETED | 改用集群已缓存数据后完成 MM-Vet + LLaVA1.5-7B 全量 218 条。 |

成功重跑配置：

- 模型：`/opt/lexiangrui/vauq_assets/models/llava-1.5-7b-hf`
- 文本扰动模型：`/opt/lexiangrui/vauq_assets/models/Qwen2.5-3B-Instruct`
- 远端日志：`~/vl_uncertainty_repro/logs/vlu-mmvet-llava_34784.out`
- 远端结果：`~/vl_uncertainty_repro/results/mmvet_llava_vlu.{jsonl,summary.json}`
- 本地 summary：`results/mmvet_llava_vlu.summary.json`

结果对比：

| 指标 | 本次复现 | 原文 | 差值 |
|---|---:|---:|---:|
| Hallucination Detection Accuracy | 79.82% | 82.11% | -2.29 pp |
| 原始回答 accuracy | 26.15% | 未在 README 表中给出 | - |
| Uncertainty AUROC | 78.81% | 未在 README 表中给出 | - |
| Uncertainty AUPR | 57.85% | 未在 README 表中给出 | - |

当前复现距离原文 Table 1 的 hallucination detection accuracy 差 `2.29 pp`。主要可疑影响项是
MM-Vet 数据版本/缓存来源、Qwen 文本扰动模型版本、LLaVA 权重 revision 和 entailment/judge 细节；
目前没有发现运行失败后的残留结果被误用。

## 输出

每行 JSONL：
- `prediction`: 低温原始答案
- `correct`: 是否正确
- `scores.vl_uncertainty`: VL-Uncertainty entropy
- `scores.cluster_distribution`: 聚类分布
- `samples.sampled_answers`: 扰动提示下的采样回答
- `samples.perturbed_questions`: 文本扰动结果

同名 `.summary.json` 包含 accuracy、hallucination_detection_accuracy、AUROC、AUPR。
