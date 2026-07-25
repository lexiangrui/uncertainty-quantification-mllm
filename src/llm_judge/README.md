# 统一大模型裁判

高置信度幻觉子集流水线使用 `OpenAIChatJudge`，通过 OpenAI Python SDK
的 Chat Completions 协议调用兼容服务。连接信息只从环境变量读取：

```bash
export OPENAI_BASE_URL=
export OPENAI_API_KEY=
```

两项为空时程序直接拒绝启动。仓库中的
`configs/judges/openai.env.example` 只保留空变量名，不保存服务地址或密钥。

调用入口：

```bash
python scripts/judging/judge_responses.py \
  --dataset vilp \
  --dataset-source /server/datasets/vilp/ViLP.parquet \
  --generation-input /server/results/generation/llava_vilp.jsonl \
  --output /server/results/judging/llava_vilp.jsonl \
  --model YOUR_JUDGE_MODEL
```

裁判只处理 greedy 主回答，一次请求同时给出最终答案正确性和视觉/推理
幻觉评分。图片以内嵌 data URL 放入 user message；无图样本只发送文本。
模型返回 `analysis`、`correct`、`rating` 和 `hallucination_types`，程序根据
`rating < 3` 确定性推导 `hallucination`。JSON 字段、类型、评分范围以及评分
与幻觉类型的一致性均严格验证，不修复输出、不重试。无法分出三部分的
greedy 回答不调用 API，而是写入 `judge.valid=false` 的审计记录。

输出按样本写入 JSONL，并根据样本 ID 断点续跑；已有输出的裁判模型、
Prompt 版本、生成输入或运行配置不同则拒绝续写。

## 旧有本地裁判

项目只保留三种 judge，GASP、MALP、VAUQ、VL-Uncertainty 和 Semantic Uncertainty 均从项目根目录的 `judge` 包导入实现。各方法不得自行维护另一份正则表达式、Judge prompt、宽松 fallback 或 verdict 解析器。

- `RegexChoiceJudge` / `extract_choice`：用于 CVBench 等封闭选择题。`extract_choice` 解析字母或从 0 开始的数字选项标识并返回统一的零基索引；`extract_choice_letter` 是返回字母的便捷封装；`RegexChoiceJudge.judge(prediction, gold_index, choices, mode)` 直接给出对错，不做语义判断。
- `QwenLLMJudge`：用于 MM-Vet、ViLP 等开放式回答（纯文本）。固定使用问题、参考答案和模型回答，greedy 生成严格 JSON verdict；格式错误直接报错。
- `QwenMultimodalHallucinationJudge`：多模态 MMHal 判定（Qwen3.6 VL，需 `fp8_kernel`）。直接观察图像，输出官方 MMHal-Bench 0--6 评分与独立 correctness；幻觉由 `rating < 3` 确定性导出，不单独询问模型。重依赖（`fp8_kernel`、`AutoModelForMultimodalLM`）懒加载，import `judge` 包本身不要求多模态环境，只有实例化该 judge 才需要。

示例：

```python
from judge import QwenLLMJudge, QwenMultimodalHallucinationJudge, RegexChoiceJudge

# 封闭选择题
correct = RegexChoiceJudge().judge("(C) 1", gold_index=2, choices=["3", "2", "1", "0"])

# 开放式纯文本
judge = QwenLLMJudge("/opt/lexiangrui/models/Qwen3-4B-Instruct-2507")
correct = judge.judge(question, references, prediction)

# 多模态幻觉 + 正确性
judge = QwenMultimodalHallucinationJudge("/opt/lexiangrui/models/qwen3.6-vl")
result = judge.judge(image, question, reference, prediction, dataset="mmvet")
# result == {"analysis": ..., "correct": bool, "rating": int, "hallucination": bool}
```
