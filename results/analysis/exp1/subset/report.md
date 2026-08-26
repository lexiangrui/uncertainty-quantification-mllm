# 3.5 困难子集汇总

## 组成（s2_subset_composition.csv）
| 模型 | 组 | n | ViLP | HB | MM-Vet | 正确率 | rating≤2 | vision 型 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llava | luh_positive | 200 | 49 | 145 | 6 | 0.410 | 1.000 | 41 |
| llava | matched_negative | 200 | 109 | 73 | 18 | 0.980 | 0.000 | 0 |
| qwen | luh_positive | 200 | 37 | 152 | 11 | 0.440 | 1.000 | 49 |
| qwen | matched_negative | 200 | 82 | 104 | 14 | 0.960 | 0.000 | 0 |
| internvl | luh_positive | 200 | 49 | 148 | 3 | 0.475 | 1.000 | 61 |
| internvl | matched_negative | 200 | 64 | 113 | 23 | 0.895 | 0.000 | 0 |

## 基线在子集上的 AUROC（s4_subset_baseline.csv）
| 模型 | PPL | SE | UMPIRE |
|---|---|---|---|
| llava | 0.402 | 0.476 | 0.413 |
| qwen | 0.485 | 0.499 | 0.477 |
| internvl | 0.489 | 0.498 | 0.502 |
