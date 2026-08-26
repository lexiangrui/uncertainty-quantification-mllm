# 3.2 检测性能汇总

## AUROC 主表（t1_detection_metrics.csv，目标=error / hallucination）
| 模型 × 数据集 | 目标 | PPL | SE | UMPIRE |
|---|---|---|---|---|
| llava / vilp | error | 0.575 | 0.655 | 0.609 |
| llava / vilp | hallucination | 0.541 | 0.605 | 0.570 |
| llava / hallusionbench | error | 0.567 | 0.553 | 0.579 |
| llava / hallusionbench | hallucination | 0.607 | 0.568 | 0.588 |
| llava / mmvet | error | 0.758 | 0.808 | 0.786 |
| llava / mmvet | hallucination | 0.567 | 0.697 | 0.679 |
| qwen / vilp | error | 0.612 | 0.717 | 0.653 |
| qwen / vilp | hallucination | 0.501 | 0.609 | 0.546 |
| qwen / hallusionbench | error | 0.643 | 0.653 | 0.629 |
| qwen / hallusionbench | hallucination | 0.558 | 0.639 | 0.613 |
| qwen / mmvet | error | 0.717 | 0.816 | 0.769 |
| qwen / mmvet | hallucination | 0.536 | 0.659 | 0.576 |
| internvl / vilp | error | 0.622 | 0.719 | 0.640 |
| internvl / vilp | hallucination | 0.533 | 0.657 | 0.563 |
| internvl / hallusionbench | error | 0.708 | 0.746 | 0.674 |
| internvl / hallusionbench | hallucination | 0.640 | 0.686 | 0.661 |
| internvl / mmvet | error | 0.724 | 0.847 | 0.828 |
| internvl / mmvet | hallucination | 0.632 | 0.779 | 0.720 |

## E vs H 目标差距：27 个单元格×方法的 AUROC(H)−AUROC(E) 均值 -0.077，范围 [-0.193, 0.040]。

与 results/metrics 交叉核对：162 项点估计，最大绝对差 0.00e+00。
