# 第一阶段结果分析报告

本报告仅分析第一阶段的三个数据集、三个模型和三种UQ方法，不包含第二阶段改进方法。

## 分析口径

- 有效 joined 样本数：6662。
- 正确性与幻觉使用独立Judge字段，错误标签为 `error = 1 - correct`。
- 主比较在每个模型×数据集单元格内部完成，宏平均对9个单元格等权。
- AUROC置信区间使用group_id级cluster bootstrap，共500次重采样。
- 低不确定性区域在每个模型×数据集×方法内部取最低10%、20%、30%，不使用标签选择阈值。

## 1. 正确性与幻觉的标签关系

| 模型 | 数据集 | N | Accuracy | Hallucination | H given wrong | H given correct | phi |
|---|---|---:|---:|---:|---:|---:|---:|
| llava | vilp | 888 | 51.0% | 52.4% | 77.5% | 28.3% | 0.493 |
| llava | hallusionbench | 1109 | 49.4% | 72.2% | 90.2% | 53.8% | 0.406 |
| llava | mmvet | 217 | 22.6% | 66.8% | 77.4% | 30.6% | 0.415 |
| qwen | vilp | 896 | 67.3% | 20.0% | 45.7% | 7.5% | 0.449 |
| qwen | hallusionbench | 1113 | 58.2% | 46.6% | 78.9% | 23.5% | 0.548 |
| qwen | mmvet | 213 | 47.9% | 31.0% | 44.1% | 16.7% | 0.297 |
| internvl | vilp | 898 | 63.5% | 27.6% | 58.8% | 9.6% | 0.530 |
| internvl | hallusionbench | 1122 | 61.0% | 44.4% | 80.8% | 21.1% | 0.587 |
| internvl | mmvet | 206 | 60.2% | 28.2% | 57.3% | 8.9% | 0.527 |

## 2. UQ预测目标拆分

| 目标 | 方法 | 宏平均AUROC | 中位数 | 范围 | CI下界>0.5单元格 |
|---|---|---:|---:|---:|---:|
| error | perplexity | 0.645 | 0.626 | 0.520–0.786 | 8/9 |
| error | semantic_entropy | 0.681 | 0.668 | 0.507–0.824 | 8/9 |
| error | umpire | 0.671 | 0.637 | 0.532–0.836 | 8/9 |
| hallucination | perplexity | 0.548 | 0.549 | 0.493–0.604 | 5/9 |
| hallucination | semantic_entropy | 0.628 | 0.594 | 0.569–0.786 | 8/9 |
| hallucination | umpire | 0.588 | 0.569 | 0.499–0.675 | 8/9 |
| hallucination_given_error | perplexity | 0.409 | 0.418 | 0.350–0.469 | 0/9 |
| hallucination_given_error | semantic_entropy | 0.558 | 0.535 | 0.459–0.713 | 3/9 |
| hallucination_given_error | umpire | 0.445 | 0.430 | 0.318–0.528 | 0/9 |

按三种方法宏平均，错误检测AUROC为0.665，幻觉检测AUROC为0.588，错误样本内幻觉检测AUROC为0.471。这三个数应结合逐单元格热力图和置信区间解释，不能将错误检测结果直接等同于幻觉检测结果。

## 3. 低不确定性幻觉盲区

| 方法 | 低UQ比例 | 低UQ幻觉率 | 高UQ幻觉率 | 幻觉落入低UQ比例 | 高UQ幻觉召回率 | 严重LUH率 |
|---|---:|---:|---:|---:|---:|---:|
| perplexity | 10% | 0.3 | 0.5 | 0.1 | 0.1 | 0.2 |
| semantic_entropy | 10% | 0.3 | 0.6 | 0.1 | 0.1 | 0.2 |
| umpire | 10% | 0.2 | 0.5 | 0.1 | 0.1 | 0.1 |
| perplexity | 20% | 0.3 | 0.4 | 0.2 | 0.2 | 0.2 |
| semantic_entropy | 20% | 0.3 | 0.6 | 0.1 | 0.3 | 0.2 |
| umpire | 20% | 0.3 | 0.5 | 0.1 | 0.2 | 0.2 |
| perplexity | 30% | 0.4 | 0.5 | 0.3 | 0.3 | 0.3 |
| semantic_entropy | 30% | 0.3 | 0.6 | 0.2 | 0.4 | 0.2 |
| umpire | 30% | 0.3 | 0.5 | 0.2 | 0.3 | 0.2 |

低UQ幻觉率不为零，或较高比例的幻觉样本落入低UQ区域，即构成baseline低不确定性幻觉盲区的证据。该指标是描述性漏检分析，不应被解释为经过标签调参后的分类阈值性能。

## 4. 输出文件

- `metrics_by_cell.csv`：模型×数据集×方法×目标的AUROC、AUPRC、PRR和尾部风险指标。
- `macro_summary.csv`：9个单元格上的宏平均汇总。
- `label_relationship.csv`：正确性与幻觉的四象限关系。
- `risk_by_decile.csv`：UQ十分位数上的错误率、幻觉率和条件幻觉率。
- `low_uq_summary.csv`：最低10%、20%、30%区域的低UQ盲区统计。
- `low_uq_samples.csv`：最低20%区域的样本级审计清单。
- `method_overlap.csv`：三种方法低UQ区域的交集模式。
- `exclusions.json`：generation、Judge、UQ的纳入排除记录。
- `auroc_heatmap.svg`、`risk_by_decile.svg`、`low_uq_hallucination.svg`：主要图形。

## 5. 解释边界

- UQ分数衡量模型输出的不确定性，不等于幻觉概率。
- Judge标签属于自动标注，LUH样本应进一步人工抽样核验。
- 低UQ阈值只按分数分位数确定，未使用幻觉标签调节。
- 不同方法的原始分数不进行跨方法数值比较，只比较其在各自单元格中的排序能力。
