# Uncertainty Quantification of MLLM

本仓库研究多模态大模型（MLLM）的回答不确定性量化与低不确定性幻觉改进，覆盖结构化回答生成、UQ 计算、正确性/幻觉评判、人类对齐、指标分析和 ERA（Early Rationale Attribution）。

## 实验设置

| 类别 | 配置 |
| --- | --- |
| 模型 | LLaVA-1.5-7B、Qwen2.5-VL-7B-Instruct、InternVL3.5-8B |
| 数据集 | ViLP、HallusionBench、MM-Vet |
| 回答格式 | `<vision>...</vision><reasoning>...</reasoning><answer>...</answer>` |
| UQ 方法 | Perplexity、Semantic Entropy、UMPIRE |
| 改进方法 | ERA |

三个模型均使用 XML 格式 LoRA。正式生成采用两阶段流程：vLLM 批量生成回答并保存精确 token ID，随后由 Hugging Face 模型对相同 token 做 teacher forcing，提取答案概率、隐藏状态和 ERA 所需信息。

```text
数据集 + Prompt + 模型/LoRA
            │
            ▼
       vLLM 批量生成
            │
            ▼
      Hugging Face 回放
            │
      ┌─────┼─────────┐
      ▼     ▼         ▼
     UQ   双模型 Judge  ERA
             │
             ▼
          人类盲裁
             │
             ▼
        指标与结果分析
```

## 仓库结构

```text
LoRA/                  LoRA 数据处理、训练与验证
baseline/              UQ 基线复现
docs/                  方法、实验与工程说明
prompts/               生成、LoRA 与 Judge Prompt
scripts/               各阶段命令行入口与分析脚本
slurm/                 集群作业入口
src/                   数据、模型、生成、UQ、Judge 与 ERA 实现
tests/                 回归测试
report/                实验报告源码与成图脚本
```

数据集、模型权重、API 凭据及大体量张量产物不提交到 Git。`results/` 版本化生成 JSONL、正式对齐标签、最终指标、实验 manifest 和复现索引，其中生成 JSONL 由 Git LFS 管理；token/hidden sidecar、图片、人工标注工作区及模型 adapter 仍只保存在集群。

## 环境与测试

建议使用 Python 虚拟环境安装根目录依赖：

```bash
git lfs install
git lfs pull
python -m pip install -r requirements.txt
pytest -q
```

LoRA 与各基线的额外依赖和运行方式分别见 `LoRA/README.md` 与 `baseline/*/README.md`。

## 正式运行

公共集群路径和 Python 环境由 `slurm/common.sh` 管理，可通过环境变量覆盖。运行前需在计算环境中准备好模型、数据集和 LoRA adapter。

单模型生成：

```bash
sbatch --export=ALL,MODEL=llava slurm/generation/generate.sbatch
```

三模型生成并自动衔接 UQ：

```bash
bash slurm/submit_full_pipeline.sh
```

也可指定模型或单独运行后续阶段：

```bash
bash slurm/submit_full_pipeline.sh internvl
sbatch --export=ALL,MODEL=llava slurm/uq/compute_uq.sbatch
sbatch --export=ALL,MODEL=llava slurm/judging/judge.sbatch
sbatch --export=ALL,MODEL=llava slurm/improvement/run_era.sbatch
```

Judge 会消耗远程 API 配额，ERA 需要额外 GPU，建议在生成结果稳定后单独提交。GPT 与 Gemini 标签不一致的样本须经人类盲裁，才能写入正式标签目录 `results/judging/`。

## 产物约定

| 产物 | 路径 |
| --- | --- |
| vLLM 原始回答与 token sidecar | `results/generation/<model>/vllm_raw/` |
| HF 回放后的回答 | `results/generation/<model>/{greedy,samples}/` |
| UMPIRE hidden sidecar | `results/hidden/<model>/<dataset>/` |
| UQ | `results/uq/<model>/<dataset>.jsonl` |
| 原始 Judge 结果 | `results/judging_<judge>/<model>/<dataset>.jsonl` |
| 人类对齐后的正式标签 | `results/judging/<model>/<dataset>.jsonl` |
| 指标与分析 | `results/metrics/`、`results/analysis/` |
| ERA 分量 | `results/era_components/<model>/<dataset>.jsonl` |

生成、回放、UQ、Judge 和 ERA 均按 `sample_id` 支持断点续跑，并在输出中保存运行配置。若模型、Prompt、采样参数或上游输入发生变化，应删除对应阶段及其下游产物后重算；不要因下游失败清空仍可验证的上游结果。

## 进一步说明

- [工程实现](docs/工程实现.md)
- [不确定性量化主实验](docs/不确定性量化主实验.md)
- [人类对齐流程](docs/人类对齐流程.md)
- [ERA 方法](docs/ERA早期推理归因不确定性量化方法.md)
- [统一大模型裁判](src/llm_judge/README.md)
- [LoRA 训练](LoRA/README.md)
