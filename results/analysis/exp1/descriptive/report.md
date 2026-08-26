# 3.1 描述性统计汇总

## 准确率与幻觉率（d2_label_stats.csv）
| 单元格 | n | Accuracy | Hallu. Rate |
|---|---:|---|---|
| llava/vilp | 886 | 0.529 | 0.527 |
| llava/hallusionbench | 1108 | 0.514 | 0.795 |
| llava/mmvet | 216 | 0.241 | 0.704 |
| qwen/vilp | 898 | 0.647 | 0.239 |
| qwen/hallusionbench | 1099 | 0.661 | 0.468 |
| qwen/mmvet | 218 | 0.560 | 0.298 |
| internvl/vilp | 900 | 0.618 | 0.254 |
| internvl/hallusionbench | 1126 | 0.710 | 0.386 |
| internvl/mmvet | 216 | 0.551 | 0.264 |
| llava/all | 2210 | 0.493 | 0.679 |
| qwen/all | 2215 | 0.645 | 0.358 |
| internvl/all | 2242 | 0.658 | 0.322 |
| all/all | 6667 | 0.599 | 0.452 |

## C×H 联合分布（d3_c_h_joint.csv，计数）
| 单元格 | 正确无幻 | 正确含幻 | 错误无幻 | 错误含幻 | P(H|C=1) | P(H|C=0) |
|---|---:|---:|---:|---:|---:|---:|
| llava/vilp | 365 | 104 | 54 | 363 | 0.222 | 0.871 |
| llava/hallusionbench | 220 | 349 | 7 | 532 | 0.613 | 0.987 |
| llava/mmvet | 39 | 13 | 25 | 139 | 0.250 | 0.848 |
| qwen/vilp | 546 | 35 | 137 | 180 | 0.060 | 0.568 |
| qwen/hallusionbench | 534 | 192 | 51 | 322 | 0.264 | 0.863 |
| qwen/mmvet | 109 | 13 | 44 | 52 | 0.107 | 0.542 |
| internvl/vilp | 515 | 41 | 156 | 188 | 0.074 | 0.547 |
| internvl/hallusionbench | 622 | 178 | 69 | 257 | 0.223 | 0.788 |
| internvl/mmvet | 112 | 7 | 47 | 50 | 0.059 | 0.515 |
| llava/all | 624 | 466 | 86 | 1034 | 0.428 | 0.923 |
| qwen/all | 1189 | 240 | 232 | 554 | 0.168 | 0.705 |
| internvl/all | 1249 | 226 | 272 | 495 | 0.153 | 0.645 |
| all/all | 3062 | 932 | 590 | 2083 | 0.233 | 0.779 |

## 幻觉类型构成（d4_hallu_types.csv）
| 单元格 | H=1 数 | vision | reasoning | both | 未列类型 |
|---|---:|---:|---:|---:|---:|
| llava/vilp | 467 | 85 | 72 | 310 | 0 |
| llava/hallusionbench | 881 | 140 | 91 | 650 | 0 |
| llava/mmvet | 152 | 42 | 20 | 90 | 0 |
| qwen/vilp | 215 | 35 | 95 | 85 | 0 |
| qwen/hallusionbench | 514 | 97 | 109 | 308 | 0 |
| qwen/mmvet | 65 | 15 | 21 | 29 | 0 |
| internvl/vilp | 229 | 40 | 80 | 109 | 0 |
| internvl/hallusionbench | 435 | 105 | 62 | 268 | 0 |
| internvl/mmvet | 57 | 13 | 27 | 17 | 0 |
| all/all | 3015 | 572 | 577 | 1866 | 0 |

## 与 results/metrics 交叉核对：18 项，最大绝对差 0.00e+00。
