# 三模型 XML 格式 LoRA

本目录实现一个独立的最小闭环：从 VQAv2 train2014 选择 2,000 条高一致性样本，由 `qwen3.7-plus` 读取图像并生成高质量的视觉证据与简短推理，程序严格校验后封装为单行 XML，最后分别对三个被测模型进行 LoRA 微调。

## 数据设计

- 来源：官方 VQAv2 train2014，仅使用训练集，避免污染 ViLP、HallusionBench 和 MM-Vet 评测集。
- 数量：1,600 train、200 validation、200 held-out format test。
- 筛选：10 位标注者中至少 7 位给出相同答案；答案简短；按 `question_type` 轮转抽样。
- 隔离：每张图片最多选择一个问题，因此三个 split 的 `image_id` 完全不重叠。
- 教师输入：真实图像、问题和 VQAv2 多数答案。教师不能自行改变最终答案。
- 教师输出：只返回 `vision`、`reasoning`、`answer` JSON；XML 标签由本地程序生成。
- 最终目标：`<vision>...</vision><reasoning>...</reasoning><answer>...</answer>`，不包含换行或标签外文本。

## 目录与路径

VQAv2 压缩包和图片放在服务器：

```text
/opt/lexiangrui/datasets/vqav2_xml_sft/
├── downloads/
├── candidates.jsonl
├── images/
├── teacher/
└── final/
```

LoRA checkpoint 与日志只写入 `/home/lexiangrui/results/lora/`。仓库不保存数据、权重、API 地址或密钥。

## 1. 配置教师 API

复制 `configs/qwen_teacher.env.example` 到服务器上的私有 `.env` 文件，再填写：

```bash
QWEN_TEACHER_BASE_URL=
QWEN_TEACHER_API_KEY=
```

代码使用 OpenAI SDK Chat Completions 协议，默认模型名是 `qwen3.7-plus`。若服务端使用不同的精确模型 ID，通过 `--model` 指定。模板故意保持为空；字段为空时脚本在发送请求前立即失败。

## 2. 登录节点准备 2,000 条候选数据

```bash
python LoRA/scripts/prepare_candidates.py \
  --questions-zip /opt/lexiangrui/datasets/vqav2_xml_sft/downloads/v2_Questions_Train_mscoco.zip \
  --annotations-zip /opt/lexiangrui/datasets/vqav2_xml_sft/downloads/v2_Annotations_Train_mscoco.zip \
  --output-root /opt/lexiangrui/datasets/vqav2_xml_sft
```

该步骤会从 COCO 官方地址只下载选中样本对应的图片，应在允许联网的登录或调试节点运行。

## 3. 调用 Qwen3.7-Plus

先在私有 shell 中加载环境变量，再运行：

```bash
python LoRA/scripts/generate_teacher_data.py \
  --candidates /opt/lexiangrui/datasets/vqav2_xml_sft/candidates.jsonl \
  --image-dir /opt/lexiangrui/datasets/vqav2_xml_sft/images \
  --accepted /opt/lexiangrui/datasets/vqav2_xml_sft/teacher/accepted.jsonl \
  --rejected /opt/lexiangrui/datasets/vqav2_xml_sft/teacher/rejected.jsonl \
  --model qwen3.7-plus \
  --workers 10
```

每个请求完成后立即追加一行。重新运行时按 `question_id` 跳过已接受数据；被拒绝样本可以再次生成。输出只记录教师模型、Prompt 版本、标注一致度、验证结论和必要内容，不记录 API 配置。

`--workers` 控制同时在途的 API 请求数。当前批量构造使用 10 并发；请求在线程池中执行，但 accepted/rejected JSONL 只由主线程逐行写入，避免并发写入破坏文件。服务端出现限流或网络错误时，该条记录进入 rejected，不用低质量内容替代，后续从断点重新请求。

首次建议加 `--limit 5`，人工查看五条结果后再生成全部数据。拒绝项不会自动用模板或低质量文本填补；需要重新调用教师，直到恰好 2,000 条全部通过。

## 4. 固化 XML 数据集

```bash
python LoRA/scripts/finalize_dataset.py \
  --accepted /opt/lexiangrui/datasets/vqav2_xml_sft/teacher/accepted.jsonl \
  --output-dir /opt/lexiangrui/datasets/vqav2_xml_sft/final
```

只有 1,600/200/200 三个 split 均完整且图片无交叉时才会写出最终数据。

## 5. 安装依赖与离线训练

在可联网的登录节点给现有环境安装 `peft`。计算节点不下载任何内容。三个正式配置分别为：

```bash
LoRA/configs/llava_inline_lora.json
LoRA/configs/qwen2_5_vl_inline_lora.json
LoRA/configs/internvl3_5_inline_lora.json
```

三个模型为 LLaVA-1.5-7B、Qwen2.5-VL-7B-Instruct 和 InternVL3.5-8B-HF。统一使用 LoRA rank 8、alpha 16、dropout 0.05、micro batch 1、梯度累积 16、1 epoch 和学习率 `2e-4`。仅语言模型的 `q_proj/v_proj` 注入 LoRA；视觉塔、多模态 projector 和全部原参数冻结。loss 只作用于 assistant XML 及官方 end-of-turn token。

LLaVA 和 Qwen2.5 的序列上限为 1024；InternVL3.5 的单图输入会展开为约 1792 个视觉 token，因此独立使用 4096。不对图像占位符截断，也不跳过图文 token 数不匹配的样本。

每条 assistant XML 目标都监督到模型官方 end-of-turn token：LLaVA 为 `</s>`，Qwen2.5 和 InternVL3.5 为 `<|im_end|>`。chat template 在 EOT 之后附加的排版 token 不进入 labels，保证最后一个有监督 token 是真实结束符。这样模型不仅学习三个 XML 块，还学习在 `</answer>` 后自主结束回答。

训练目录中的 `train_metrics.jsonl` 按每次 optimizer update 保存累计 micro-batch 的平均训练 loss、学习率、累计样本数和运行时间；`validation_metrics.jsonl` 按 epoch 保存验证 loss。两者均逐行即时写入，可直接用于后续 loss 曲线与实验报告，不以 Slurm 文本日志作为最终绘图数据源。

## 6. 正式训练结果保存位置

三个模型的正式训练结果分别保存在服务器用户目录：

```text
/home/lexiangrui/results/lora/official_inline/
├── llava-1.5-7b/
├── qwen2.5-vl-7b/
└── internvl3.5-8b-hf/
```

每个目录保存该模型训练得到的 LoRA adapter、processor 配置以及以下实验记录：

```text
train_metrics.jsonl
validation_metrics.jsonl
training_config.json
```

这里保存的是可挂载到原始基础模型上的 **LoRA adapter**，不是复制或合并后的完整模型权重。后续生成回答时，需要同时提供基础模型路径和对应 adapter 路径。例如 LLaVA 使用：

```text
基础模型：/opt/lexiangrui/models/llava-1.5-7b-hf
LoRA adapter：/home/lexiangrui/results/lora/official_inline/llava-1.5-7b
```

其余两个模型同样分别挂载上述目录中的 adapter。Slurm 标准输出和错误日志单独保存在：

```text
/home/lexiangrui/results/lora/logs/
```

日志仅用于排查运行问题；loss 绘图和论文实验记录应读取正式结果目录中的 JSONL 文件。

### LLaVA 数据量与 epoch 消融

用于比较“1,600 条训练样本训练 1 epoch”和“400 条训练样本训练 5 epochs”的独立消融配置为：

```text
LoRA/configs/llava_400train_100val_5epoch.json
```

该实验固定读取正式 `train.jsonl` 的前 400 条作为训练样本、正式 `validation.jsonl` 的前 100 条作为验证样本，不改变 held-out `test.jsonl`。五个 epoch 共曝光 2,000 样本次，产生 125 次 optimizer update 和 5 个 validation loss 点。结果独立保存到：

```text
/home/lexiangrui/results/lora/ablation_400train_100val_5epoch/llava-1.5-7b/
```

该目录不会覆盖正式的 1,600 条、1 epoch adapter。两个 adapter 已使用 `LoRA/scripts/evaluate_llava_heldout.py` 在同一份 200 条 held-out test 上完成配对比较。每题使用 1 条 greedy 和 10 条 `temperature=1.0` sampling 回答，固定 Prompt、seed 和 `max_new_tokens=256`。结果保存在：

```text
/home/lexiangrui/results/lora/heldout_comparison/
├── llava_1600train_1epoch.jsonl
└── llava_400train_5epoch.jsonl
```

| 配置 | Greedy 严格 XML | Greedy 答案准确率 | Sampling 严格 XML | 每题 10 条 sampling 全部合法 | EOS 停止率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,600 train × 1 epoch | 200/200（100%） | 135/200（67.5%） | 1,988/2,000（99.4%） | 188/200（94%） | 2,200/2,200（100%） |
| 400 train × 5 epochs | 200/200（100%） | 127/200（63.5%） | 1,968/2,000（98.4%） | 170/200（85%） | 2,200/2,200（100%） |

两种配置都能稳定约束 greedy XML 和 EOS；但减少样本并重复训练没有改善格式或答案质量，sampling 格式稳定性反而下降。因此正式实验继续使用 1,600 条训练样本、1 epoch 的 adapter。答案准确率采用当前 VQAv2 规范化精确匹配，只作为该 held-out 对比的内部指标。

### Sampling reject & resample

自由采样在 `temperature=1.0` 下仍有约 0.6% 严格 XML 失败。评测脚本默认启用 **reject & resample**：

- 每个最终保留的 sample 最多重采 `K=10` 次（`--reject-resample-k 10`）
- 仅当 `strict_xml_valid=True` 时接受；否则丢弃并换 seed 重抽
- 若 10 次仍全部非法，保留最后一次并标记 `reject_resample.accepted=false`
- 每条 sample 记录 `attempts_used` / `rejected_count`；整题记录 `reject_resample_summary`

在登录节点提交正式 adapter 的 reject-resample held-out：

```bash
export PROJECT_ROOT=/home/lexiangrui/<your-repo>
export ADAPTER_PATH=/home/lexiangrui/results/lora/official_inline/llava-1.5-7b
export OUTPUT=/home/lexiangrui/results/lora/heldout_comparison/llava_1600train_1epoch_reject_resample_k10.jsonl
export REJECT_RESAMPLE_K=10
export TEMPERATURE=1.0
sbatch slurm/lora/evaluate_llava_heldout.sbatch
```

## 7. 测试

```bash
pytest -q LoRA/tests
```

单元测试不访问真实 API，也不下载模型或数据。正式训练前还应对 held-out test 的 XML 合法率、标签顺序、标签重复率和答案准确率做推理评测。
