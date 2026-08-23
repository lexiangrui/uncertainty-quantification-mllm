# 统一大模型裁判

项目对外保留四类 Judge：

1. `ClosedSourceJudge`：通过 OpenAI Python SDK 调用远程闭源/托管模型 API；
2. `OpenSourceJudge`：本地加载 Qwen 模型，通过 `model_name` 在文本 Qwen 和 Qwen-VL 两种实现之间切换；
3. `RuleJudge`：不调用模型的确定性规则判断，支持选择题和 Yes/No。
4. `NLIJudge`：本地 DeBERTa 的前提-假设蕴含判断，用于 Semantic Entropy 的语义聚类，不负责最终答案或幻觉评分。

正式第一阶段 Judge 使用 `ClosedSourceJudge`，默认裁判模型为
`gpt-5.6-terra`，通过 OpenAI Python SDK 的 Responses API 调用兼容服务。
连接信息优先从环境变量读取：

```bash
export OPENAI_BASE_URL=
export OPENAI_API_KEY=
```

两项为空时会读取项目根目录下不提交 Git 的 `.ven`；两处均无有效配置时
程序拒绝启动。仓库中的 `configs/judges/openai.env.example` 只保留空变量名，
不保存服务地址或密钥。

调用入口：

```bash
python scripts/judging/judge_responses.py \
  --dataset vilp \
  --dataset-source /server/datasets/vilp/ViLP.parquet \
  --greedy-input /server/results/generation/llava/greedy/vilp.jsonl \
  --output /server/results/judging/llava/vilp.jsonl \
  --max-tokens 4096 \
  --timeout 300
```

`--greedy-input` 应指向 HF replay 完成后的正式 `results/generation/<model>/greedy/<dataset>.jsonl`，不能使用 `vllm_raw/` 中尚未完成内部信号回放的中间文件。

裁判只处理 greedy 主回答，一次请求同时给出最终答案正确性和视觉/推理
幻觉评分。图片以内嵌 data URL 放入 user message；无图样本只发送文本。
模型返回 `analysis`、`correct`、`rating` 和 `hallucination_types`，程序根据
`rating < 3` 确定性推导 `hallucination`。JSON 字段、类型、评分范围以及评分
与幻觉类型的一致性均严格验证。网络或服务异常最多重试三次；无法分出三部分的
greedy 回答不调用 API，而是写入 `judge.valid=false` 的审计记录。

输出按样本写入 JSONL，并根据样本 ID 断点续跑；已有输出的裁判模型、
Prompt 版本、生成输入或运行配置不同则拒绝续写。

闭源 Judge 的 system prompt 位于
`prompts/judge/closed_source_judge.md`。运行 JSONL 的 `run`
头记录 `judge_prompt_sha256` 确定实际调用的 Prompt 内容。

## 开源 Judge

项目只保留这一套统一 Judge 包；VAUQ、VL-Uncertainty 和 Semantic Uncertainty 均从项目根目录的 `src.llm_judge` 导入所需实现。各方法不得自行维护另一份正则表达式、Judge prompt、宽松 fallback 或 verdict 解析器。

- `RuleJudge` / `extract_choice`：用于 CVBench 等封闭选择题。`extract_choice` 解析字母或从 0 开始的数字选项标识并返回统一的零基索引；`extract_choice_letter` 是返回字母的便捷封装；`RuleJudge.judge(prediction, gold_index, choices, mode)` 直接给出对错，不做语义判断。
- `OpenSourceJudge(model_name=...)`：由模型名称选择 Qwen 后端。`Qwen2.5-3B-Instruct` 和 `Qwen3-4B-Instruct-2507` 使用文本正确性 Judge；`Qwen3.6-VL` 使用多模态 MMHal Judge。需要将 `model_path` 指向本地 checkpoint 时单独传入该参数。
- 文本后端用 greedy 生成严格 JSON verdict，只返回正确性；Qwen-VL 后端输出 MMHal-Bench 0--6 评分和独立 correctness，幻觉由 `rating < 3` 确定性导出。`fp8_kernel` 和 `AutoModelForMultimodalLM` 仍然懒加载。

示例：

```python
from src.llm_judge import ClosedSourceJudge, OpenSourceJudge, RuleJudge

# 封闭选择题
correct = RuleJudge().judge("(C) 1", gold_index=2, choices=["3", "2", "1", "0"])

# 开放式纯文本
judge = OpenSourceJudge(
    "Qwen3-4B-Instruct-2507",
    model_path="/opt/lexiangrui/models/Qwen3-4B-Instruct-2507",
)
correct = judge.judge(question, references, prediction)

# 多模态幻觉 + 正确性
judge = OpenSourceJudge(
    "Qwen3.6-VL",
    model_path="/opt/lexiangrui/models/qwen3.6-vl",
)
result = judge.judge(image, question, reference, prediction, dataset="mmvet")
# result == {"analysis": ..., "correct": bool, "rating": int, "hallucination": bool}

# 远程闭源/托管模型
judge = ClosedSourceJudge("gpt-5.6-terra")
```

## NLI Judge

`NLIJudge` 从本地 sequence-classification checkpoint 加载 DeBERTa NLI 模型，自动识别其 entailment 标签。它提供单对 `judge(premise, hypothesis)` 和批量 `check_pairs(pairs)` 两个接口。

Semantic Entropy 对非完全相同的回答与当前语义簇代表做双向 NLI；只有两个方向均为 entailment 时才合并为同一语义簇。调用入口是 `scripts/uq/compute_uq.py`，模型路径由 `--entailment-model-path` 指定。
