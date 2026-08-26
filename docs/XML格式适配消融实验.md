# XML 格式适配消融实验

## 1. 实验目的

本实验检验 XML 格式 LoRA 是否改变三个被测模型的答案正确率与幻觉率。比较对象不是两种输出字符串的格式有效性本身，而是在相同题目、相同基础权重和相同 greedy 解码条件下：

- **XML-LoRA 条件**：加载当前格式 LoRA adapter，使用主实验 XML 回答要求，输出 `<vision>/<reasoning>/<answer>` 三段内容；
- **原生提示条件**：使用原生基础模型，不加载 adapter，只通过独立 Prompt 要求依次输出 `Visual Observation`、`Reasoning` 和 `Final Answer` 三段内容。

原生提示文件为 [`prompts/generation/native_three_part_zero_shot.md`](../prompts/generation/native_three_part_zero_shot.md)。该文件只正向描述回答结构。裁判继续使用主实验已有的 [`prompts/judge/closed_source_judge.md`](../prompts/judge/closed_source_judge.md)，本消融不修改 Judge Prompt。

## 2. 配对抽样与成本控制

先检查三个模型主实验 XML greedy 回答的 `sections_valid` 字段，仅保留 LLaVA、Qwen 和 InternVL 均能完整解析出 `<vision>`、`<reasoning>` 与 `<answer>` 的问题实例。在这一共同合格抽样框内，以固定随机种子 42 进行简单随机不放回抽样，选取 500 条样本。三个模型和两种条件共享完全相同的 500 个 `sample_id`，因此模型间与条件间的数据集构成都一致，且不会把 XML 格式失败混入内容质量比较。

为避免重复计算与重复裁判，XML-LoRA 条件直接复用主实验已经生成的 greedy 回答及其 Gemini-3.7-Flash 原始 Judge 标签，只生成原生提示条件的新回答，并仅将原生提示回答提交 Gemini-3.7-Flash。每个模型最多需要新增 500 次基础模型 greedy 生成，理论上最多产生

\[
3\ \text{个模型}\times 500\ \text{条样本}=1500
\]

次 Judge 请求。所有原生回答均提交 Judge；格式是否完整只作为独立的指令遵循指标，不决定是否裁判。

抽样清单保存在 `results/ablation/xml_format/sample_manifest.json`，其中记录随机种子、各数据集抽中数量、完整样本 ID 与样本集合 SHA-256。后续程序会校验该哈希，避免不同阶段使用了不同的 500 条样本。

## 3. 回答生成协议

两种条件均使用 greedy decoding：`do_sample=false`、`temperature=0`、`max_new_tokens=512`。图像、问题、模型 checkpoint 和数据集版本保持一致，唯一的实验处理是是否加载 XML 格式 LoRA，以及与之配套的回答结构 Prompt。

原生提示条件的回答格式为：

```text
Visual Observation: ...
Reasoning: ...
Final Answer: ...
```

原生回答在 Judge 阶段不经过重组。程序将模型生成的完整 `raw_response` 原样作为 `candidate_response` 提交，由 Gemini 自行完成判断；与此同时，本地解析器记录回答能否完整分离为 `Visual Observation`、`Reasoning` 和 `Final Answer` 三段。Judge Prompt 本身保持不变，审计记录保存的输入也只有原始回答，不生成回退字段或修复后的版本。

## 4. Judge 与统计口径

XML-LoRA 条件从 `results/judging_gemini_3_7_flash` 提取主实验已有的 Gemini 标签；原生提示条件同样使用 Gemini-3.7-Flash。两种条件使用相同的 Judge 模型和 Judge Prompt，且裁判任一条件时均看不到另一条件的回答或标签。XML 回答沿用主实验的三段结构化输入，原生回答则将未经重组的完整 `raw_response` 直接提交。

Judge 在同一次调用中输出：

- `correct`：仅根据正式答案与参考答案的语义一致性判断；
- `rating`：按既有 0–6 量表评价视觉观察与推理中的幻觉；
- `hallucination = (rating < 3)`：由程序确定性派生；
- `hallucination_types`：视觉幻觉和/或推理幻觉。

正文内容指标比较以初始 500 条为候选集合，并在每个模型内仅保留原生回答能够完整分离为三段、且两种条件 Judge 标签均有效的样本。由于三种原生模型的格式遵循情况不同，最终配对数允许随模型变化。对每个模型分别报告：

1. 配对样本上的正确率和幻觉率；
2. 差值 `XML-LoRA − 原生提示`；
3. 以 `group_id` 为聚类单位的 1,000 次 bootstrap 95% 置信区间；
4. 配对二元结果的精确 McNemar 检验。

正确率差值大于 0 表示 XML-LoRA 条件正确率更高；幻觉率差值小于 0 表示 XML-LoRA 条件幻觉率更低。

## 5. 运行流程

所有集群 Python 命令继续使用 `MLLM-UQ` 虚拟环境。

### Step 1：生成共享抽样清单

```bash
sbatch slurm/ablation/xml_format_prepare.sbatch
```

该步骤只执行一次。若清单已经用于生成或 Judge，不应重新抽样覆盖。

### Step 2：生成三个模型的原生提示回答

```bash
sbatch slurm/ablation/xml_format_generate.sbatch
```

Slurm 数组的三个任务分别处理 LLaVA、Qwen 和 InternVL。每个任务先从主实验结果中提取相同 500 条 XML-LoRA greedy 回答，再用未加载 adapter 的基础模型生成原生提示回答。输出目录为：

```text
results/ablation/xml_format/generation/
├── xml_lora/<model>/<dataset>.jsonl
└── native_prompt/<model>/<dataset>.jsonl
```

### Step 3：复用 XML Gemini 标签并裁判原生回答

在能够访问 API 的登录节点运行：

```bash
nohup bash scripts/ablation/run_xml_format_judge_all.sh \
  > logs/ablation/xml-format-judge.out 2>&1 &
```

如实际 API 模型 ID 不同，可显式覆盖：

```bash
JUDGE_MODEL='<实际 Gemini Flash 模型 ID>' \
  nohup bash scripts/ablation/run_xml_format_judge_all.sh \
  > logs/ablation/xml-format-judge.out 2>&1 &
```

脚本先物化 XML-LoRA 的 Gemini 标签子集，再仅对原生提示回答调用 Gemini。输出按模型、数据集和条件分别保存，原生条件支持从已完成样本继续运行。

### Step 4：重新汇总结果

Judge 脚本完成后会自动汇总；也可单独运行：

```bash
source slurm/common.sh
"$HF_PYTHON" scripts/analysis/xml_format_ablation.py \
  --judge-model gemini-3.7-flash
```

最终生成：

- `results/ablation/xml_format/analysis/paired_performance.csv`；
- `results/ablation/xml_format/analysis/结果.md`；
- `results/ablation/xml_format/analysis/manifest.json`。

## 6. 完整性检查

分析程序会拒绝以下情况：抽样清单哈希不一致、任一条件的生成样本集合不等于共享 500 条、同一输出中出现重复 `sample_id`、有效 Judge 记录缺少布尔标签，或筛选后没有任何共同有效样本。结果写入实验报告附录 A.4 前，还应核对三个模型的原生三段完整数、最终配对数与 Judge 状态，避免采用不完整输出。
