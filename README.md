# Uncertainty Quantification of MLLM

本仓库实现多模态大模型（MLLM）的结构化回答生成、不确定性量化、正确性/幻觉评判和 ERA 改进方法。当前正式推理链路是 **vLLM 批量生成 + Hugging Face（HF）精确 token 回放**：vLLM 负责吞吐，HF 负责与三个模型兼容的 token 概率、隐藏状态和注意力提取。

## 实验范围

| 类别 | 当前配置 |
| --- | --- |
| 模型 | LLaVA-1.5-7B、Qwen2.5-VL-7B-Instruct、原始 InternVL3.5-8B |
| 数据集 | ViLP（900）、HallusionBench（1,129）、MM-Vet（218） |
| 回答协议 | `<vision>...</vision><reasoning>...</reasoning><answer>...</answer>` |
| 生成 | 1 条 greedy + 10 条随机采样；sample 最多 50 次 XML 拒绝重采样 |
| UQ | Perplexity、Semantic Entropy、UMPIRE |
| 改进方法 | ERA（Early Rationale Attribution） |

三个模型分别挂载 XML 格式 LoRA。InternVL 使用原始 OpenGVLab `InternVL3_5-8B` checkpoint 和匹配的 `adapter-original`，vLLM 生成与 HF 内部状态回放共用同一套 token 与聊天模板。

## 正式推理架构

```text
dataset + XML prompt + base model/LoRA
                 │
                 ▼
       persistent vLLM engine
  批量生成 greedy / K=10 samples
                 │
                 ├── raw JSONL
                 └── exact generated token IDs
                              │
                              ▼
                persistent HF replay model
           对完全相同的 token 做 teacher forcing
                              │
                              ├── answer log probabilities
                              ├── sample final-token hidden states
                              └── final generation JSONL
                                           │
                         ┌─────────────────┼───────────────┐
                         ▼                 ▼               ▼
                   PPL / SE / UMPIRE      Judge            ERA
```

- `scripts/generation/run_vllm_pipeline.py` 在三个数据集及 greedy/samples 两个阶段之间复用同一个 vLLM engine，不重复加载权重。
- `scripts/generation/run_hf_replay_pipeline.py` 对全部阶段复用同一个 HF 模型，也不在 greedy 与 samples 之间重复加载。
- 显存档位由 `src/models/runtime.py` 根据当前可见 GPU 自动选择。32 GiB GPU 默认使用 `max_num_seqs=8`、HF replay batch size `5`；HF 遇到 OOM 时递归拆分当前 batch 后重试。
- HallusionBench 的有图与无图记录在 HF replay 时分组批处理。原始 InternVL 对无图记录不构造视觉 token、`pixel_values` 或 `image_flags`，而是直接调用其语言模型前向。
- 解析失败的少量异常回答保留原文并标记 `sections_valid=false`；它们不会静默修复，也不会进入后续 UQ 计算。

## 目录与正式产物

```text
LoRA/                                  XML 格式数据、训练和验证
baseline/                              PPL、Semantic Entropy、UMPIRE 与保留复现
docs/                                  实验规范、工程实现和分析说明
prompts/                               版本化生成、LoRA 和 Judge prompt
scripts/generation/                    vLLM 生成与 HF replay 入口
scripts/uq/                            UQ 入口
scripts/judging/                       Judge 入口
scripts/evaluation/                    指标入口
scripts/improvement/                   ERA 特征提取入口
slurm/                                 集群正式作业入口
src/                                   数据、模型、生成、Judge、UQ、ERA 公共实现
tests/                                 回归测试
```

数据集、基础模型权重、API 凭据和 `results/` 不提交到 Git。正式结果结构如下：

| 产物 | 路径 |
| --- | --- |
| vLLM 原始回答 | `results/generation/<model>/vllm_raw/{greedy,samples}/<dataset>.jsonl` |
| vLLM 精确 token sidecar | 与上述 JSONL 相邻的 `<dataset>.tokens/*.pt` |
| HF replay 后 greedy | `results/generation/<model>/greedy/<dataset>.jsonl` |
| HF replay 后 samples | `results/generation/<model>/samples/<dataset>.jsonl` |
| UMPIRE hidden sidecar | `results/hidden/<model>/<dataset>/*.pt` |
| UQ | `results/uq/<model>/<dataset>.jsonl` |
| Judge | `results/judging/<model>/<dataset>.jsonl` |
| 指标 | `results/metrics/<model>/<dataset>.json` |
| ERA 分量 | `results/era_components/<model>/<dataset>.jsonl` |
| 分析 | `results/analysis/` |

正式模型与 adapter 映射由 `slurm/generation/generate.sbatch` 和 `slurm/improvement/run_era.sbatch` 统一维护：

| `MODEL` | family | 基础模型 | adapter |
| --- | --- | --- | --- |
| `llava` | `llava_1_5` | `$MODEL_ROOT/llava-1.5-7b-hf` | `results/lora/llava/adapter` |
| `qwen` | `qwen2_5_vl` | `$MODEL_ROOT/Qwen2.5-VL-7B-Instruct` | `results/lora/qwen/adapter` |
| `internvl` | `internvl3_5_original` | `$MODEL_ROOT/InternVL3_5-8B` | `results/lora/internvl/adapter-original` |

## 集群运行

`slurm/common.sh` 提供可覆盖的公共路径。默认模型和数据根目录分别为 `/opt/$USER/models` 与 `/opt/$USER/datasets`，HF 环境为 `$HOME/.venvs/vlm-transformers/bin/python`；vLLM 环境默认是 `$HOME/.venvs/vllm-0.25.1/bin/python`。计算节点按离线模式运行，模型和数据需预先在登录节点准备完成。

单模型正式 generation：

```bash
sbatch --export=ALL,MODEL=llava slurm/generation/generate.sbatch
```

三模型 generation，并在各自完成后自动提交 UQ：

```bash
bash slurm/submit_full_pipeline.sh
```

也可以只提交指定模型：

```bash
bash slurm/submit_full_pipeline.sh internvl
```

只计算某个模型的 UQ：

```bash
sbatch --export=ALL,MODEL=llava slurm/uq/compute_uq.sbatch
```

Judge 和 ERA 不在上述 DAG 中自动提交：Judge 会消耗远程 API 配额，ERA 需要额外 GPU，应在正式 generation 产物稳定后单独运行。

## 断点续跑与清理边界

vLLM raw、HF replay、UQ、Judge 和 ERA 均以 `sample_id` 断点续跑。每个 JSONL 首行保存完整 `run` 配置；已有文件的模型、数据、Prompt、参数或上游输入与当前运行不一致时，程序会拒绝追加，防止混合实验。

恢复作业时遵循以下边界：

1. 配置一致且 token/hidden sidecar 完整时，直接重提同一作业，已完成样本会跳过。
2. 只修改 HF replay 时，保留 `vllm_raw/`，删除对应 final `greedy/`、`samples/` 与其下游 UQ/ERA 后重新 replay。
3. 修改生成模型、adapter、Prompt 或采样参数时，重新生成受影响的 `vllm_raw/`，并清理其 downstream 产物。
4. 不因下游失败清空可验证的上游结果；只有运行配置不兼容或 sidecar 缺失时才重做对应阶段。

## UQ、Judge 与 ERA

`scripts/uq/compute_uq.py` 只读取 HF replay 后的 greedy/samples：Perplexity 使用 greedy 最终答案的 HF log probability；Semantic Entropy 使用 samples 的最终答案与 mean log probability；UMPIRE 额外读取 sample 最终答案末 token 的末层 hidden sidecar。

`scripts/judging/judge_responses.py` 只评价 greedy 主回答。闭源 Judge 的服务地址和密钥只从 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 读取，Prompt 哈希保存在输出的 `run` metadata 中。

ERA 读取 HF replay 后的 greedy JSONL 及其精确 token sidecar，并通过 HF eager attention 计算浅层归因分量：

```bash
sbatch --export=ALL,MODEL=llava slurm/improvement/run_era.sbatch
```

详细说明见：

- [工程实现](docs/工程实现.md)
- [不确定性量化主实验](docs/不确定性量化主实验.md)
- [LoRA 说明](LoRA/README.md)
- [ERA 方法](docs/ERA早期推理归因不确定性量化方法.md)
- [统一大模型裁判](src/llm_judge/README.md)

## 测试

安装依赖后在仓库根目录运行：

```bash
pytest -q
```

测试覆盖数据适配、XML 解析、vLLM/HF 后端、显存批处理、HF replay（含 HallusionBench 无图记录）、断点续跑、UQ、Judge、LoRA 和 ERA。
