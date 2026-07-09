# VAUQ 复现工程

论文 **VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation** 的方法复现，工程化的目标是**模型/数据集/判分器三轴可插拔**，方便后续切换。

VAUQ 需要读取模型的 token 分布熵与注意力信号，因此**不使用 API**，而是在 GPU 节点上用 Hugging Face 直接加载本地白盒模型。

## 方法公式

```
IS_core = H(y | v_masked, t) - H(y | v, t)
s_VAUQ  = H(y | v, t) - alpha * IS_core
```

- `H(y | v, t)`：原图条件下生成答案的预测熵（响应 token 上 Shannon 熵的均值，bit）。
- `H(y | v_masked, t)`：移除核心视觉 token（按注意力 top-k_ratio 选出，attention-mask 置零）后，同一答案的预测熵。
- `IS_core`：图像核心信息对预测不确定性的影响。
- `s_VAUQ`：越高表示答案越可疑（越可能错误）。

核心视觉 token 的选取：在 `[prompt; generated_ids]` 上做一次带 `output_attentions` 的前向，取 `layer_range` 层、所有注意力头、所有响应 query 位置对视觉 token 区间的平均注意力，按 `topk_ratio` 取 top-k，在第二次前向里把这些位置的 `attention_mask` 置 0。

## 目录结构

```
vauq-repro/
├── README.md
├── requirements.txt
├── configs/
│   └── vauq.env.example      # 统一配置（所有模型路径 + venv + HF 缓存）
├── scripts/
│   ├── run_vauq.py            # 通用 CLI
│   └── fetch_assets.sh        # 登录节点预下载模型/数据集
├── slurm/
│   ├── run_llava_cvbench.sbatch
│   └── run_paper_grid.sbatch
└── src/vauq/
    ├── types.py               # VAUQResult
    ├── metrics.py             # 逐 token 熵（base=2）
    ├── scoring.py             # compute_vauq_scores（core / blank 两种掩码）
    ├── eval.py                # AUROC / AUPR
    ├── images.py              # blank / patch masking（blank 策略用）
    ├── semantic.py            # 文本归一化/语义辅助工具
    ├── backends/              # 模型：llava
    ├── benchmarks/            # 数据集：vilp, mmvet, cvbench
    └── judges/                # 判分器：letter, qwen_local, llm, none
```

`results/` 与 `logs/` 在运行时自动生成。

## 三轴可插拔

| 轴 | 选项 | 说明 |
|---|---|---|
| backend | `llava` | 白盒 LVLM 封装，暴露 `generate_with_ids / get_logits / get_logits_masked` |
| benchmark | `vilp`, `mmvet`, `cvbench` | 数据集加载器，返回 `{img, question, gt_ans, choices?}` |
| judge | `letter`, `qwen_local`, `llm`, `none` | `letter`=选择题字母正则；`qwen_local`=集群本地 Qwen2.5-3B free-form 判分；`llm`=API 判分；`none`=VAUQ 阶段暂不打标签 |

默认超参来自论文 Appendix F（在 `scripts/run_vauq.py` 的 `DEFAULT_HYPERPARAMETERS`）：LLaVA-1.5-7B/13B × ViLP/MMVet/CVBench。`K` 已换算为 `topk_ratio=K/100`。

新增任意一轴只需在对应子包注册表里加一项。

## MiliLab 集群连接

不在 `mg01` 上跑推理或下大模型；`mg01` 只用于登录、环境准备、提交 Slurm 作业与（按本工程约定）一次性预下载模型/数据集。GPU 任务通过 `sbatch` 交给计算节点。

在本机 `~/.ssh/config` 配置（替换 `<username>` 与 `<path/to/your/private/key>`）：

```sshconfig
Host mg01
    HostName 10.254.30.24
    User <username>
    Port 22
    IdentityFile <path/to/your/private/key>
    IdentitiesOnly yes

Host mg01-out
    HostName 10.254.30.24
    User <username>
    Port 22
    IdentityFile <path/to/your/private/key>
    ProxyJump repository-out
    IdentitiesOnly yes

Host repository-out
    HostName 49.232.163.49
    User <username>
    Port 19479
    IdentityFile <path/to/your/private/key>
    IdentitiesOnly yes
```

校内：`ssh mg01`；校外：`ssh mg01-out`。`*-out` 带宽有限，只用于登录/提交，不传大文件。

## 环境准备

集群上已存在 venv `/home/lexiangrui/.venvs/vlm-transformers`（Python 3.11，torch 2.11+cu128，transformers 5.12）。如需补依赖：

```bash
/home/lexiangrui/.venvs/vlm-transformers/bin/pip install scipy scikit-learn
```

本机从零装：

```bash
conda create -n vauq python=3.11 -y
conda activate vauq
pip install -r requirements.txt
```

## 配置

工程使用**一份统一配置** `configs/vauq.env`，集中放所有模型路径、venv 与 HF 缓存设置：

```bash
cd vauq-repro
cp configs/vauq.env.example configs/vauq.env
```

编辑 `configs/vauq.env`，把 `LLAVA_7B_HF_MODEL` / `LLAVA_13B_HF_MODEL` 指向集群上的本地模型路径。

## 预下载模型与数据集（在 mg01 上一次性完成）

```bash
ssh mg01-out
cd /home/lexiangrui/vauq-repro
source configs/vauq.env
bash scripts/fetch_assets.sh
```

`fetch_assets.sh` 把模型与数据集都集中下到**同一个资产根目录** `VAUQ_ASSETS_DIR`（默认 `/opt/lexiangrui/vauq_assets/`），其下分 `models/` 与 `datasets/`：

- 模型 → `$VAUQ_ASSETS_DIR/models/`：默认下载 `llava-1.5-7b-hf/`、`llava-1.5-13b-hf/`（`huggingface_hub.snapshot_download`，断点续传）。已有 `Qwen2.5-VL-7B-Instruct/`、`Qwen3-VL-8B-Instruct/`、`InternVL3_5-8B-HF/` 权重目录保留，但不再作为推理 backend 使用。
- 数据集 → `$VAUQ_ASSETS_DIR/datasets/`：CV-Bench 2D/3D、MM-Vet、ViLP parquet。

不依赖已废弃的 `huggingface-cli`。之后推理作业即可在 `HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1 / TRANSFORMERS_OFFLINE=1` 下纯离线运行（slurm 脚本已默认设好）。

## 小样本调试

```bash
salloc --partition=batch --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gres=gpu:1 --mem=48G --time=00:30:00
source configs/vauq.env
"$PYTHON_BIN" scripts/run_vauq.py \
    --backend llava --benchmark cvbench --judge letter \
    --model-path "$LLAVA_HF_MODEL" \
    --limit 4 --output results/cvbench_llava_debug.jsonl
```

## Slurm 批量运行

```bash
sbatch slurm/run_llava_cvbench.sbatch
squeue -u lexiangrui
tail -f logs/vauq-llava-cvbench_<job_id>.out
```

冒烟默认可设置 `LIMIT=4`；跑全量在提交前 `export LIMIT=0` 或不设置 `LIMIT`。

论文主表当前复现组合（LLaVA-1.5-7B/13B × ViLP/MMVet/CVBench，已排除 VisualCoT）：

```bash
sbatch slurm/run_paper_grid.sbatch
# 只做冒烟：
LIMIT=4 sbatch slurm/run_paper_grid.sbatch
```

`run_paper_grid.sbatch` 使用论文 Appendix A 的生成设置：greedy decoding，`max_new_tokens=128`。CVBench 用 `letter` 正则判断字母是否相同；ViLP/MMVet 先输出 `correct=null` 的 VAUQ 结果，随后用本地 Qwen 裁判补标。

## Free-form Judge

free-form 数据集（ViLP/MMVet）不强制复刻 GPT-5 judge，默认使用集群已有开源模型 `Qwen2.5-3B-Instruct` 替代。为避免同一进程同时加载 LLaVA-13B 和 Qwen，流程分成两步：

1. GPU 节点跑 VAUQ：`--judge none`，输出 `correct=null`。
2. GPU 节点跑 Qwen 本地裁判：读取已有 JSONL，输出 `.qwen_judged.jsonl` 和 summary。

提交 Qwen 裁判作业：

```bash
sbatch slurm/run_qwen_judge_paper_v2.sbatch
```

也可以手动补标单个文件：

```bash
source configs/vauq.env
"$PYTHON_BIN" scripts/apply_llm_judge.py \
    --judge qwen_local \
    --input results/mmvet_llava-1.5-7b-hf_vauq.jsonl \
    --output results/mmvet_llava-1.5-7b-hf_vauq.qwen_judged.jsonl \
    --resume
```

API judge 仍保留为可选路径，但不作为当前论文版复现默认配置。

## 输出

每条 JSONL 记录：

- `prediction`：模型答案。
- `correct`：判分器结果；CVBench 是字母正则匹配，free-form 原始 VAUQ 文件为 `null`，补标后的 `.qwen_judged.jsonl` 为本地 Qwen judge 结果。
- `scores.entropy`：原图预测熵。
- `scores.entropy_masked`：核心视觉 token 掩码后的预测熵。
- `scores.is_score`：`entropy_masked - entropy`。
- `scores.vauq`：`entropy - alpha * is_score`。
- `config`：本次运行的超参。

`.summary.json` 含 accuracy 与 `vauq / entropy / is_score` 的 AUROC、AUPR。

## 复现核查（2026-07-07）

核查基准：

- 论文：arXiv `2602.21054v2` / ACL Findings 2026 版本。
- 官方代码：`deeplearning-wisc/vauq`，当前核查 commit 为 `72d3a55`。
- 集群结果：`/home/lexiangrui/vauq-repro/results/*paper_v2*.summary.json`。
- Slurm 作业：主网格 `34769`（array 0-5）已完成；Qwen judge `34770`（array 0-3）已完成。
- 日志：`/home/lexiangrui/vauq-repro/logs/vauq-paper-grid_34769_*.{out,err}` 与 `logs/vauq-qwen-judge_34770_*.{out,err}`。

论文主表（Table 1）和当前集群结果的 VAUQ AUROC 对比：

| 模型 | 数据集 | 当前结果 | 论文结果 | 差值 | 说明 |
|---|---:|---:|---:|---:|---|
| LLaVA-1.5-7B | ViLP | 63.59 | 77.00 | -13.41 | `Qwen2.5-3B-Instruct` 本地 judge；论文为 GPT-5 三次多数投票。 |
| LLaVA-1.5-7B | MMVet | 80.75 | 81.50 | -0.75 | `Qwen2.5-3B-Instruct` 本地 judge；数值接近论文。 |
| LLaVA-1.5-7B | CVBench | 68.77 | 73.20 | -4.43 | 跑 2D+3D 共 2638 条；2D 子集 72.65，3D 子集 63.05。 |
| LLaVA-1.5-13B | ViLP | 65.45 | 69.50 | -4.05 | `Qwen2.5-3B-Instruct` 本地 judge；case2 accuracy 很低，整体 AUROC 受标签分布影响明显。 |
| LLaVA-1.5-13B | MMVet | 84.96 | 88.60 | -3.64 | `Qwen2.5-3B-Instruct` 本地 judge。 |
| LLaVA-1.5-13B | CVBench | 65.47 | 68.30 | -2.83 | 跑 2D+3D 共 2638 条；2D 子集 71.40，3D 子集 58.25。 |
| LLaVA-1.5-7B/13B | VisualCoT | 未复现 | 77.80 / 80.20 | - | 本工程当前排除 VisualCoT。 |

当前 `paper_v2` 已修复的结果影响项：

- 生成设置已对齐论文 Appendix A：`run_paper_grid.sbatch` 使用 greedy decoding，`max_new_tokens=128`；`LlavaBackend.generate_with_ids` 不再硬编码 64。
- CVBench 继续跑 2D+3D 全集 2638 条，判分采用字母正则匹配。这里不强制复刻论文未说明的 exact matching 归一化细节，只要求能正确判断预测字母和金标字母是否相同。
- free-form 判分不强制复刻 GPT-5；当前采用集群本地开源 `Qwen2.5-3B-Instruct` 作为替代 judge，并在输出中标记为 `qwen_local`。
- 当前仍只提交单次结果；论文 Appendix A 写“averaged over three random seeds”。如需完全对齐统计口径，需要再补 3-seed 重复实验。

当前 summary 文件：

| 数据集 | 7B summary | 13B summary |
|---|---|---|
| ViLP | `results/vilp_llava-1.5-7b-hf_paper_v2.qwen_judged.summary.json` | `results/vilp_llava-1.5-13b-hf_paper_v2.qwen_judged.summary.json` |
| MMVet | `results/mmvet_llava-1.5-7b-hf_paper_v2.qwen_judged.summary.json` | `results/mmvet_llava-1.5-13b-hf_paper_v2.qwen_judged.summary.json` |
| CVBench | `results/cvbench_llava-1.5-7b-hf_paper_v2.summary.json` | `results/cvbench_llava-1.5-13b-hf_paper_v2.summary.json` |

论文原文与官方代码仓库不一致/不完整的地方：

- 生成设置不一致：论文 Appendix A 写 greedy decoding、最大生成长度 128；官方代码仓库当前实现为 `do_sample=True, temperature=0.1, max_new_tokens=64`，且 `max_new_tokens` 在 LLaVA wrapper 里硬编码为 64。
- CVBench 数据范围不一致：论文数据集说明写 CVBench 共 2638 条，主表应对应 2D+3D 全集；官方代码 `benchmark/cvbench.py` 只加载 `load_dataset("nyu-visionx/CV-Bench", "2D")["test"]`，即只跑 2D 子集。
- 论文 Appendix F 的 LLaVA-1.5-7B 超参与官方 GitHub 默认值不一致：论文给 ViLP `alpha=0.6, K=60`、CVBench `alpha=1.2, K=30`；官方代码默认 ViLP `topk_ratio=1.0`、CVBench `topk_ratio=0.1, alpha=1.5`。本工程按论文 Appendix F，而不是按官方代码默认值。
- 官方仓库当前只内置 LLaVA-1.5-7B 的默认超参，且 benchmark 目录没有 VisualCoT loader；论文主表却报告 LLaVA-1.5-13B、Qwen、InternVL 和 VisualCoT。
- 官方仓库不发布论文使用的 answer cache / GPT-5 多数投票标签，只说明 AUROC/AUPR 需要外部提供 `label`。因此 free-form 数据集无法仅凭官方仓库完全复现论文标签。
- 论文写 CVBench 使用 exact answer matching，但没有说明是否对 `(C)` 与 `C` 做归一化；这个细节足以把 accuracy 从正常水平变成 0%，应在任何结果表中显式说明。

下一步建议：

1. 优先复核 ViLP：当前与论文差距最大，且 case2 标签分布极端，建议抽样人工看 Qwen judge 是否偏保守。
2. 如需完全对齐论文统计口径，再补 3 个 seed 的重复实验并报告均值。
3. 保留当前 Appendix F 超参作为论文基准；如果要验证官方 GitHub，则另开一组“official-code-defaults”结果，避免两套基准混在同一张表里。

## 已知限制

- 论文使用 GPT-5 三次多数投票标注 free-form 回答正确性；本工程当前用集群本地 `Qwen2.5-3B-Instruct` 作为开源替代 judge。
- VisualCoT 因数据体积过大已从本次复现范围中移除。
- LLaVA backend 强制 `attn_implementation="eager"`，否则 `output_attentions` 不可用。
