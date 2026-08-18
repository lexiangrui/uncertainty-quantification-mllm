# ERA（Early Rationale Attribution，早期推理归因）不确定性量化方法

## 一、核心思想与方法动机

多模态大模型在生成推理链（`<vision>` 视觉描述与 `<reasoning>` 推理过程）时，若产生幻觉，往往会表现出**对自身生成的思考过程产生强烈的内生自确信，而脱离了对客观事实输入的依赖**。

**ERA（Early Rationale Attribution，早期推理归因）** 旨在通过单次前向传播，在浅层解码器（Layer 0-1）中解耦并量化模型在给出最终答案时，**流向自身生成前文（$V+R$）相较于真实外部输入（$I+Q$）的相对注意力依赖比率**。

比值越高，表明最终答案越严重依赖模型自生成的未经验证上下文，内生不确定性与幻觉风险越高。

---

## 二、形式化定义

给定 Prompt 与模型自回归生成的响应序列，采用连续 Token 切片将注意力目标划分为 5 个语义区域（保留 XML 标签，零分词器依赖，零短答案丢失）：

| 区域 | 符号 | 定义 | 语义属性 |
|---|---|---|---|
| **图像证据** | **I** | 图像 Visual Tokens | 客观真实输入（Grounded Input） |
| **问题提示** | **Q** | System Prompt 与问题文本 Tokens | 客观真实输入（Grounded Input） |
| **视觉描述** | **V** | `<vision>...</vision>` 连续切片 | 模型自生成上下文（Generated Rationale） |
| **推理过程** | **R** | `<reasoning>...</reasoning>` 连续切片 | 模型自生成上下文（Generated Rationale） |
| **最终答案** | **A** | `<answer>...</answer>` 连续切片 | 决策目标（Decision Target） |

在早期解码层（第 0 和 1 层）上，记 $\alpha(A\to T)$ 为答案预测行流向目标区域 $T$ 的平均注意力质量（对全部注意力头及答案行求平均）。

### ERA 核心不确定性指标 $U_{\mathrm{ERA}}$
$$
U_{\mathrm{ERA}} =
\frac{\alpha(A\to V) + \alpha(A\to R)}{\alpha(A\to I) + \alpha(A\to Q) + \alpha(A\to V) + \alpha(A\to R) + \epsilon}
$$

---

## 三、工程实现与文件清单

- **核心算法实现**：[`src/improvement/era.py`](file:///Users/lexiangrui/Desktop/Uncertainty%20Quantification%20of%20MLLM/src/improvement/era.py)
- **多模态输入构造与对齐**：[`src/improvement/backend.py`](file:///Users/lexiangrui/Desktop/Uncertainty%20Quantification%20of%20MLLM/src/improvement/backend.py)
- **批量特征提取入口**：[`scripts/improvement/run_era.py`](file:///Users/lexiangrui/Desktop/Uncertainty%20Quantification%20of%20MLLM/scripts/improvement/run_era.py)
- **LUH 难例全量评估**：[`scripts/analysis/evaluate_era.py`](file:///Users/lexiangrui/Desktop/Uncertainty%20Quantification%20of%20MLLM/scripts/analysis/evaluate_era.py)
- **分母形式消融**：[`scripts/analysis/ablate_denominator.py`](file:///Users/lexiangrui/Desktop/Uncertainty%20Quantification%20of%20MLLM/scripts/analysis/ablate_denominator.py)
- **集群调度脚本**：[`slurm/improvement/run_era.sbatch`](file:///Users/lexiangrui/Desktop/Uncertainty%20Quantification%20of%20MLLM/slurm/improvement/run_era.sbatch)
- **针对性单元测试**：[`tests/improvement/test_era.py`](file:///Users/lexiangrui/Desktop/Uncertainty%20Quantification%20of%20MLLM/tests/improvement/test_era.py)
