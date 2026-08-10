# VAUQ 有效性验证

## 1. 研究目的

本文针对 VAUQ（Vision-Aware Uncertainty Quantification）提出两方面的有效性质疑，并设计对应的对照实验：

1. VAUQ 的注意力核心区域遮蔽是否真的提供了超越简单视觉消融的有效信息；
2. 论文报告的 AUROC 是否真正衡量视觉幻觉检测能力，还是主要衡量模型回答正确性的区分能力。

验证对象为 LLaVA-1.5-7B。实验一使用 ViLP、MMVet 和 CVBench；实验二只使用 ViLP 和 MMVet。所有实验尽量复用同一批模型回答、相同评分公式、相同标签和相同超参数，以减少非目标变量的影响。

---

## 2. 文章存在的问题

### 2.1 注意力核心区域的有效性缺少充分对照

VAUQ 首先根据生成回答对视觉 token 的注意力选出 top-K 核心区域，然后遮蔽这些 token，计算遮蔽前后的预测熵差：

$$
\operatorname{IS}_{\mathrm{core}}
=H(\mathbf y\mid\mathbf v_{\mathrm{masked}},\mathbf t)
-H(\mathbf y\mid\mathbf v,\mathbf t).
$$

最终分数为：

$$
s_{\mathrm{VAUQ}}
=H(\mathbf y\mid\mathbf v,\mathbf t)
-\alpha\operatorname{IS}_{\mathrm{core}}.
$$

该方法隐含的核心主张是：基于注意力选出的视觉 token 比任意视觉 token 更能代表支撑回答的关键视觉证据。因此，移除这些 token 所产生的熵变化应当比简单的整图移除或等面积随机移除更有助于判断回答是否可靠。

但是，仅仅证明遮蔽核心区域后的分数能够区分正确和错误回答，还不能排除以下替代解释：

- VAUQ 的增益可能只来自移除了一定数量的视觉 token，而不是注意力定位准确；
- 任意较大范围的视觉信息破坏都可能产生相似的熵变化；
- 完全移除视觉输入这一更简单的方法可能已经达到相同或更好的效果；
- 注意力 top-K 的表现可能与等面积随机遮蔽没有稳定差异。

因此，需要在同一模型回答、同一数据集和同一遮蔽面积下，将注意力核心区域与 blank 和 random 基线进行严格配对比较。

### 2.2 论文将“回答错误”等同于“视觉幻觉”

论文对 ViLP、MMVet 和 VisualCoT 的标签构造方式是：使用语言模型判断模型回答是否与标准答案语义等价；如果回答正确，则标记为非幻觉，否则标记为幻觉。CVBench 则直接使用答案精确匹配。

因此，其标签实际上是：

$$
E=\mathbb 1[\text{模型回答与标准答案不一致}],
$$

相应 AUROC 实际衡量：

$$
\operatorname{AUROC}(s_{\mathrm{VAUQ}},E),
$$

即 VAUQ 对正确回答和错误回答的排序能力。

但视觉幻觉通常是指回答中出现了被图像否定、缺乏图像依据或与视觉内容冲突的具体陈述。回答错误与视觉幻觉并不等价：

| 类型 | 最终答案是否正确 | 是否包含视觉幻觉 | 示例性质 |
|---|---:|---:|---|
| A | 是 | 否 | 视觉观察与最终答案均正确 |
| B | 是 | 是 | 最终答案正确，但附带了错误视觉描述 |
| C | 否 | 否 | 视觉观察正确，但计算、推理、格式或知识运用错误 |
| D | 否 | 是 | 回答中包含错误视觉事实，最终答案也错误 |

论文当前的标签会把 B 类标记为非幻觉，把 C 类标记为幻觉，从而混淆“正确性”和“视觉幻觉”两个不同构念。

因此，论文的 AUROC 计算在其自定义标签下并没有数值错误，但其标签不足以支撑“VAUQ 检测视觉幻觉”的强结论。更准确地说，这是一个构念效度问题：实验明确验证了回答正确性区分能力，却未充分验证视觉幻觉特异性。

### 2.3 现有数据集对视觉幻觉的可识别性不同

- **MMVet**：回答相对丰富，任务包含识别、OCR、空间、知识、语言和数学推理，适合区分感知错误与非视觉推理错误。
- **ViLP**：适合研究语言先验与反事实视觉证据之间的冲突，但当前答案通常较短，可供判断的视觉信息有限。
- **CVBench**：模型通常只输出一个选项字母。错误字母只能证明答案错误，无法判断错误来自视觉幻觉、空间推理错误还是选项映射错误，因此不适合作为严格的生成式视觉幻觉检测证据。

---

## 3. 总体实验原则

1. 使用同一 LLaVA-1.5-7B 模型和相同数据集版本；
2. 使用 greedy decoding，最大生成长度保持一致；
3. 同一条样本的不同遮蔽策略使用原图生成出的同一回答；
4. 在该回答上进行 teacher-forced 前向计算，不因遮蔽策略重新生成答案；
5. 保持 $\alpha$、层范围和数据集对应的 $K$ 不变；
6. random 与 core 遮蔽相同数量的视觉 token；
7. 所有随机选择按样本固定种子，保证结果可复现；
8. 正确性标签和幻觉标签分开构造和报告；
9. 幻觉 judge 不应看到 VAUQ、entropy、IS 或遮蔽策略等分数和信息；
10. 统计比较采用配对方法，并报告置信区间，而不只比较单次数值大小。

---

## 4. 实验一：核心视觉 token 遮蔽的有效性验证

### 4.1 实验问题

本实验回答以下问题：

1. VAUQ 的注意力核心区域遮蔽是否显著优于整图视觉 token 消融；
2. VAUQ 是否显著优于遮蔽相同数量 token 的随机策略；
3. VAUQ 的性能是否主要由原始预测熵贡献，而核心区域 Image-Information Score 只提供有限增益。

### 4.2 对比方法

#### 4.2.1 Core：论文现有方法

聚合指定层、全部注意力头和全部回答 token 对视觉 token 的注意力，从 576 个视觉 token 中选择注意力最高的 top-K%，并将这些位置的 attention mask 置零。

LLaVA-1.5-7B 沿用论文超参数：

| 数据集 | 遮蔽比例 $K$ | $\alpha$ | 层范围 |
|---|---:|---:|---:|
| ViLP | 60% | 0.6 | 10–25 |
| MMVet | 40% | 0.6 | 10–25 |
| CVBench | 30% | 1.2 | 10–25 |

#### 4.2.2 Blank：全部视觉 token 消融

将全部 576 个视觉 token 的 attention mask 置零：

$$
\operatorname{IS}_{\mathrm{blank}}
=H(\mathbf y\mid\varnothing,\mathbf t)
-H(\mathbf y\mid\mathbf v,\mathbf t).
$$

这里的 blank 指视觉 token 消融，而不是把原图像素替换为黑色图像。这样可以避免黑色图像本身仍产生视觉 embedding，确保实验真正测量“无视觉 token”条件。

#### 4.2.3 Random：等数量随机视觉 token 消融

在 576 个视觉 token 中无放回随机选择与 core 完全相同数量的 token：

$$
|\mathbf v_{\mathrm{random}}|=|\mathbf v_{\mathrm{core}}|.
$$

ViLP、MMVet 和 CVBench 分别随机遮蔽 60%、40% 和 30%。随机种子使用全局种子与样本 ID 共同确定，避免恢复运行或样本顺序改变 mask。

建议对 random 至少使用 3 个随机种子，并报告均值、标准差和置信区间；如计算预算允许，可扩展至 5–10 个种子，以降低单次随机 mask 的偶然性。

#### 4.2.4 Entropy：不使用视觉消融分数

将原始预测熵作为额外基线：

$$
s_{\mathrm{entropy}}=H(\mathbf y\mid\mathbf v,\mathbf t).
$$

该基线用于判断 VAUQ 的效果究竟来自核心视觉信息分数，还是主要来自原始语言模型不确定性。

### 4.3 控制变量

对每条样本，先用原图生成一次回答 $\mathbf y$。随后 core、blank 和 random 均使用同一 $\mathbf y$ 计算熵。由此应满足：

- 三种策略的 `prediction` 完全相同；
- 三种策略的正确性标签完全相同；
- 三种策略的原图 entropy 完全相同；
- 只有遮蔽后的 entropy、Image-Information Score 和最终 VAUQ 分数不同。

若上述前三项不一致，说明实验混入了重新生成、数据顺序或判分器变化等非目标变量，不能进行严格比较。

### 4.4 评价指标

分别报告：

- Accuracy：只用于核对三种策略使用了同一批回答，不用于评价遮蔽策略；
- VAUQ AUROC、AUPR；
- 原始 entropy AUROC、AUPR；
- Image-Information Score AUROC、AUPR；
- random 多种子均值与标准差；
- core 相对 blank、random 和 entropy 的 AUROC 差值。

AUROC 的正负方向必须统一，使更高风险分数对应更可能错误的回答。

### 4.5 统计检验

由于不同方法作用于完全相同的样本，应采用配对统计方法：

1. 以样本为单位进行 paired bootstrap；
2. ViLP 的事实/反事实图像成对来源于同一原始问题，建议以原始问题为重采样单位；
3. 计算 AUROC 差值：

$$
\Delta_{\mathrm{core-random}}
=\operatorname{AUROC}_{\mathrm{core}}
-\operatorname{AUROC}_{\mathrm{random}},
$$

$$
\Delta_{\mathrm{core-blank}}
=\operatorname{AUROC}_{\mathrm{core}}
-\operatorname{AUROC}_{\mathrm{blank}};
$$

4. 报告 95% bootstrap 置信区间；
5. random 多种子实验同时报告种子间波动。

### 4.6 判定标准

- 如果 core 在三个数据集上稳定优于 random 和 blank，且差值置信区间不跨 0，则支持注意力核心区域选择具有额外价值；
- 如果 core 与 random 相近，说明效果可能主要来自遮蔽一定数量的视觉 token，而非核心区域定位；
- 如果 blank 与 core 相近或更优，说明复杂的注意力区域选择未必必要；
- 如果 VAUQ 与原始 entropy 相近，说明 Image-Information Score 的边际贡献有限；
- 如果优势只出现在单个数据集或单个随机种子上，则不能支持方法具有稳定普适性。

---

## 5. 实验二：正确性区分与视觉幻觉检测的解耦验证

### 5.1 实验问题

本实验回答以下问题：

1. VAUQ 对视觉幻觉的 AUROC 是否与论文报告的错误回答 AUROC 相同；
2. 在回答正确性固定后，VAUQ 是否仍能区分有幻觉和无幻觉回答；
3. VAUQ 与幻觉的相关性是否主要由“错误回答更容易包含幻觉”这一混杂关系产生。

本实验不再使用 CVBench。CVBench 只输出选择题字母，无法可靠判断错误来源是视觉感知、推理还是选项映射，不适合作为生成式视觉幻觉检测数据。

### 5.2 数据与模型设置

| 项目 | 设置 |
|---|---|
| 被评估模型 | LLaVA-1.5-7B |
| 数据集 | MMVet 全集、ViLP 全集 |
| 回答生成 | greedy decoding，`max_new_tokens=128` |
| 幻觉 judge | Qwen3.6-27B，多模态输入，直接查看原图 |
| Judge 解码 | 单次确定性解码，`do_sample=false`、`temperature=0` |
| Judge 输入 | 原图、原问题、可接受参考答案、LLaVA 回答、MMHal-Bench 评价规则 |
| Judge 不可见信息 | VAUQ、entropy、IS、正确性旧标签、Core/Blank/Random 策略 |

Qwen3.6-27B 必须以视觉—语言模式直接输入原图。不能只给它问题、标准答案和模型回答，否则它仍然只能判断答案匹配，无法独立检查视觉依据。

本阶段按既定范围只使用一次 Qwen3.6-27B 评审，不加入人工校验、多次独立评审或分歧裁决。因此，最终结论应明确写作“基于 Qwen3.6-27B judge 标签”，不能将其表述为人工确认的客观幻觉真值。

本实验 Prompt 和运行规范采用以下官方实现：

- 回答生成直接遵循 [LLaVA 官方 MM-Vet 推理脚本](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/mmvet.sh) 与 [官方 `model_vqa_loader.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_loader.py) 的原始问题输入方式；
- Judge system prompt 直接以 [MMHal-Bench 官方 `eval_gpt_mmhal.py`](https://github.com/RLHF-V/RLHF-V/blob/main/eval/eval_gpt_mmhal.py) 中的 `template` 为主体，完整保留 0–6 rating；只额外要求独立给出回答正确性。

### 5.3 LLaVA 回答生成 Prompt

原复现 Prompt 要求：

```text
NOTE: Provide only the final answer. Do not provide unrelated details.
```

这会人为压缩回答，并改变开放式 VQA 的自然生成行为。实验二改为采用 [LLaVA 官方 MM-Vet 推理脚本](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/mmvet.sh) 和 [官方 `model_vqa_loader.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_loader.py) 的设置：**只输入数据集原始问题，不添加任何回答指令**。

因此实际 Prompt 为：

```text
{original_dataset_question}
```

其中不追加“只输出最终答案”“解释视觉证据”“逐步推理”或“避免幻觉”等文字。该方案也与 RLHF-V/MMHal-Bench 的官方生成流程一致：生成器直接读取原问题，使用模型自身的自然回答风格。

选择这一设置的理由是：

- 最接近 LLaVA 和 MM-Vet 的官方评测协议；
- 不通过要求解释而人为延长回答，避免回答长度改变幻觉机会；
- 不通过“基于图像”“不要猜测”等指令显式抑制幻觉；
- 不要求 chain-of-thought，避免把隐藏推理过程引入被评估文本；
- MMVet 和 ViLP 采用相同原则，均保留各自原始问题，不增加数据集特定后缀。

修改 Prompt 后必须重新生成回答并重新计算 VAUQ，不能沿用实验一的短答案分数，因为回答 token 序列和对应 entropy 已发生变化。每条结果应保存 `prediction` 和精确的 `generated_ids`。

### 5.4 两个评价标签

Qwen3.6-27B 在同一次评审中输出 `correct` 和 MMHal-Bench `rating`。实验使用 `correct` 作为正确性标签，并严格按 MMHal-Bench 官方阈值从 `rating` 派生 `hallucination` 标签，不再要求 Judge 直接输出幻觉布尔值。

#### 5.4.1 `correct`

$$
E=1 \iff \texttt{correct=false}.
$$

`correct=true` 表示模型回答正确回答了问题，且与图像和可接受参考答案一致；否则为 `false`。该标签用于复现论文实际测量的错误回答检测 AUROC。

#### 5.4.2 `rating` 与派生的 `hallucination`

参考 MMHal-Bench 的官方定义：视觉幻觉是回答中包含图像或上下文没有呈现、暗示或支持的具体信息，包括不存在或错误的对象、属性、数量、动作、情绪、文字、位置和关系。

MMHal-Bench 的官方 rating 定义完整保留：

| Rating | 官方含义 | 派生幻觉标签 |
|---:|---|---:|
| 6 | very informative with good analysis or reasoning, no hallucination | 0 |
| 5 | very informative, no hallucination | 0 |
| 4 | somewhat informative, no hallucination | 0 |
| 3 | not informative, no hallucination | 0 |
| 2 | very informative, with hallucination | 1 |
| 1 | somewhat informative, with hallucination | 1 |
| 0 | not informative, with hallucination | 1 |

官方 `eval_gpt_mmhal.py` 的实现为：

$$
H=\mathbb 1[R<3],\qquad R\in\{0,1,2,3,4,5,6\}.
$$

因此 `rating` 为 0、1、2 时记为 `hallucination=true`；为 3、4、5、6 时记为 `hallucination=false`。这一阈值固定，不根据 MMVet 或 ViLP 调整，也不在结果出来后重新选择。

Judge 直接按 MMHal-Bench 规则判断，不再拆分其他子标签，也不输出错误类型、置信度等额外标签：

- 回答遗漏参考答案中的信息、回答不够详细或直接表达不确定，不自动构成幻觉；
- 参考答案可能不完整，某个细节未出现在参考答案中不代表它是幻觉，必须以图像为首要证据；
- 与图像事实一致的合理推断或通用知识不自动构成幻觉；
- 关于图像的具体但无依据的对象、属性、动作、数量、位置和关系构成幻觉；
- 回答冗长本身既不加分也不扣分；
- 推理或计算错误可以使 `correct=false`，但只有其同时声称错误视觉事实时才应得到低于 3 的 rating；
- 对直接询问视觉事实的问题，错误对象、属性、OCR、计数或空间答案本身属于错误视觉陈述，可同时得到 `correct=false` 和低于 3 的 rating；
- 拒答或无法确定通常为 `correct=false`，但若没有虚假视觉陈述，应得到 rating 3，而不是幻觉分数；
- ViLP 的反事实图像中，当前图像证据必须优先于语言常识；judge 不查看同一问题的另一张配对图像。

### 5.5 Qwen3.6-27B Judge Prompt

#### 5.5.1 System Prompt 来源与结构

System Prompt 直接以 [MMHal-Bench 官方 `eval_gpt_mmhal.py`](https://github.com/RLHF-V/RLHF-V/blob/main/eval/eval_gpt_mmhal.py) 中的 `template` 为基础，保留以下内容：

- “impartial and objective judge”的角色；
- 幻觉是图像或上下文中不存在或未暗示的信息；
- 对象、动作、情绪、属性、计数和其他视觉细节均属于检查范围；
- 官方五个示例及其分析，包括错误计数、合理补充、虚构位置、遗漏但无虚假陈述、通用分析；
- 参考答案可能不完整，judge 必须谨慎使用参考答案；
- 详细回答只要没有错误视觉陈述就不应被判为幻觉。

在官方 Prompt 末尾只做一项任务扩展和一项格式约束：

1. 在保留官方 0–6 rating 的基础上，增加 `correct`，并明确正确性与幻觉必须独立判断；
2. 将最终输出约束为结构化字段，便于稳定解析，但不改变官方 rating 的含义或阈值。

除这两项外，不增加其他标签或细粒度 rubric。

#### 5.5.2 User Prompt

在 system prompt 后依次提供原图和以下文本：

```text
[Dataset]
{MMVet or ViLP}

[Question]
{question}

[Accepted Reference Answer]
{reference_answer}

[Candidate Response]
{model_answer}

[Output]
Judge whether the response correctly answers the question, and assign the
official MMHal-Bench rating from 0 to 6. Return JSON only.
```

#### 5.5.3 结构化输出

```json
{
  "analysis": "Brief justification following the MMHal-Bench criteria.",
  "correct": true,
  "rating": 4
}
```

`correct` 必须是 JSON boolean，`rating` 必须是 0–6 的整数；`analysis` 只保留 MMHal-Bench 官方要求的简短解释，不参与任何标签或统计计算。程序随后确定性生成 `hallucination = (rating < 3)`。不得让 Judge 再输出一个可能与 rating 冲突的 `hallucination` 字段。输出无法解析或 rating 越界时，将该样本记为 judge failure，不通过重复调用改变标签。

### 5.6 Judge 运行与输出控制

1. 所有样本使用同一模型 checkpoint、图像预处理、MMHal-Bench system prompt 和输出 schema；
2. 使用 `do_sample=false`、`temperature=0`，每条样本只进行一次语义评审；
3. MMVet 和 ViLP 使用同一评价规则，不针对单个数据集调整幻觉定义；
4. Judge 输入中不包含 VAUQ、entropy、IS、预测风险排序或实验一结果；
5. 保存完整原始 judge 输出、`correct`、`rating`、按阈值派生的 `hallucination`、Prompt 版本哈希、模型路径和推理配置；
6. 不进行格式恢复、多数投票或再次裁决；解析失败样本从主分析排除并单独报告；
7. `correct` 与 `rating` 必须来自同一次 judge 输出；`hallucination` 只能由该 rating 按固定阈值派生，避免跨作业标签漂移或字段自相矛盾。

### 5.7 核心评价

#### 5.7.1 论文原始目标：错误回答区分

$$
\operatorname{AUROC}_{\mathrm{error}}
=\operatorname{AUROC}(s_{\mathrm{VAUQ}},E).
$$

#### 5.7.2 独立视觉幻觉区分

$$
\operatorname{AUROC}_{\mathrm{hall}}
=\operatorname{AUROC}(s_{\mathrm{VAUQ}},H).
$$

$$
\Delta_{\mathrm{construct}}
=\operatorname{AUROC}_{\mathrm{error}}
-\operatorname{AUROC}_{\mathrm{hall}}.
$$

#### 5.7.3 固定正确性后的幻觉检测

最关键的分析是只在错误回答内部计算：

$$
\operatorname{AUROC}
\left(s_{\mathrm{VAUQ}},H\mid E=1\right).
$$

这直接比较“错误且有幻觉”和“错误但无幻觉”。如果样本量允许，还应计算：

$$
\operatorname{AUROC}
\left(s_{\mathrm{VAUQ}},H\mid E=0\right),
$$

比较“正确但有幻觉”和“正确且无幻觉”。

#### 5.7.4 四象限分数比较

分别报告以下四组 VAUQ 分数的分布：

$$
S_{E=0,H=0},\quad
S_{E=0,H=1},\quad
S_{E=1,H=0},\quad
S_{E=1,H=1}.
$$

重点进行两个固定正确性的比较：

$$
S_{E=1,H=1}\ \text{vs.}\ S_{E=1,H=0},
$$

$$
S_{E=0,H=1}\ \text{vs.}\ S_{E=0,H=0}.
$$

### 5.8 控制混杂变量

建立回归模型：

$$
\operatorname{logit}P(H=1)
=\beta_0+\beta_1s_{\mathrm{VAUQ}}
+\beta_2E+\beta_3L
+\text{dataset/task effects},
$$

其中 $L$ 为回答长度。必要时加入图像复杂度、问题类型和事实/反事实条件。

如果控制正确性和任务类型后，$\beta_1$ 接近 0 或不显著，而 $\beta_2$ 较强，则说明 VAUQ 与幻觉的表面关联主要由回答错误这一变量解释。

### 5.9 统计要求

- 使用 paired/clustered bootstrap 报告 AUROC、AUPR 与差值的 95% 置信区间；
- ViLP 按原始问题聚类重采样，避免事实/反事实配对被拆散；
- 报告四象限的样本数量，样本过少时不解释 AUROC；
- 对多个分组或多个指标进行显著性检验时说明多重比较校正；
- 除 AUROC 外报告 AUPR，因为幻觉标签可能明显不平衡。
- 主分析排除 judge JSON 解析失败的样本，并报告失败数量；
- AUROC 以错误或幻觉为正类，直接使用较高表示风险更高的 VAUQ 分数，避免符号方向混乱。

### 5.10 本阶段限制

本阶段按实验范围不加入人工校验，也不使用多个 judge 或多次独立评审。因此需要显式承认：

- 标签可能包含 Qwen3.6-27B 自身的视觉识别偏差；
- 单一 judge 的结果适合检验“在更合理的多模态幻觉定义下，论文结论是否仍成立”，但不能证明标签等同于人工真值；
- 后续若结论对 judge 标签高度敏感，再另行加入人工或第二模型验证，不在本阶段执行。

### 5.11 判定标准

若出现以下结果，则支持论文存在 correctness–hallucination 构念混淆：

1. 存在足够数量的 B 类和 C 类样本，证明错误与幻觉并不等价；
2. $\operatorname{AUROC}_{\mathrm{error}}$ 较高，但 $\operatorname{AUROC}_{\mathrm{hall}}$ 显著更低；
3. 在 $E=1$ 的错误回答内部，VAUQ 对幻觉的 AUROC 接近 0.5；
4. 控制正确性、回答长度和任务类型后，VAUQ 分数对幻觉的独立解释能力很弱；
5. VAUQ 无法稳定区分“错误但无幻觉”和“错误且有幻觉”。

如果 VAUQ 在 MMVet 和 ViLP 上、固定正确性后仍能稳定区分 Qwen3.6-27B 所标注的视觉幻觉，则说明论文虽然原标签定义不够严格，但其方法可能确实捕获了视觉幻觉相关信号。

---

## 6. 两项实验的联合解释

两项实验分别检验不同层面的有效性：

| 实验 | 主要问题 | 被验证的主张 |
|---|---|---|
| Core / Blank / Random | 注意力核心区域是否有额外价值 | 方法机制有效性 |
| Correctness / Hallucination 双标签 | VAUQ 测量的是错误还是视觉幻觉 | 评价构念有效性 |

可能出现以下联合结论：

| 遮蔽实验 | 双标签实验 | 解释 |
|---|---|---|
| Core 优于基线 | 固定正确性后仍能检测幻觉 | 同时支持机制与幻觉检测主张 |
| Core 优于基线 | 只能检测正确性 | 核心区域机制可能有效，但“幻觉检测”结论被夸大 |
| Core 不优于基线 | 能检测幻觉 | 分数可能包含幻觉信号，但注意力核心定位不是必要机制 |
| Core 不优于基线 | 只能检测正确性 | 方法机制和评价构念均缺乏充分支持 |

最终结论应依据置信区间、效应量、跨数据集稳定性和标注可靠性，而不能仅依据某个数据集上的单一 AUROC 排名。

---

## 7. 实验结果

### 7.1 实验一：Core、Blank、Random 与 Entropy

#### 7.1.1 主结果

本轮对每条样本只生成一次回答、调用一次 judge，并在同一进程中依次计算 Core、Blank 和 Random。三种方法的分数与唯一的 `prediction`、`correct` 和 `judge_result` 保存在同一条记录中，因此不存在跨作业回答或标签漂移。三种方法还共用同一次原图前向计算得到的 entropy。

| 数据集 | 方法 | Seed | Accuracy | VAUQ AUROC | VAUQ AUPR | Entropy AUROC | IS AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| ViLP | Core | — | 55.33 | 65.71 | 69.31 | 66.07 | 58.63 |
| ViLP | Blank | — | 55.33 | **67.11** | **71.32** | 66.07 | **63.91** |
| ViLP | Random | 42–46 | 55.33 | 65.96 ± 0.10 | 69.70 ± 0.38 | 66.07 | 55.69 ± 0.89 |
| MMVet | Core | — | 23.39 | **85.31** | 68.47 | 83.01 | 65.66 |
| MMVet | Blank | — | 23.39 | 82.26 | 61.22 | 83.01 | 56.21 |
| MMVet | Random | 42–46 | 23.39 | 83.83 ± 0.56 | **68.92 ± 0.99** | 83.01 | **67.06 ± 2.72** |
| CVBench | Core | — | 61.71 | **68.77** | **78.85** | 68.74 | **64.09** |
| CVBench | Blank | — | 61.71 | 66.22 | 76.79 | 68.74 | 57.24 |
| CVBench | Random | 42–46 | 61.71 | 68.55 ± 0.10 | 78.76 ± 0.10 | 68.74 | 55.91 ± 0.19 |

注：Random 报告 seeds 42–46 五次随机遮蔽结果的均值 ± 样本标准差；Core 和 Blank 为确定性结果。每个数据集内部 Accuracy 和 Entropy AUROC 相同，是因为所有方法使用完全相同的回答、标签和原图 entropy。粗体标记确定性方法或 Random 均值中的最高值。

#### 7.1.2 配对差值

| 数据集 | $\Delta_{\mathrm{core-random}}$ | 95% CI | $\Delta_{\mathrm{core-blank}}$ | 95% CI | 是否支持 Core 的注意力定位优势 |
|---|---:|---|---:|---|---|
| ViLP | -0.26 | — | -1.41 | — | 否；Blank 最好，Core 还略低于 Random 均值 |
| MMVet | +1.48 | — | +3.05 | — | 有限；Core 最好，但只比 Random 均值高约 1.5 点 |
| CVBench | +0.23 | — | +2.55 | — | 有限；Core 与 Random 均值几乎相同 |

#### 7.1.3 实验一观察

统一实验得到以下结果：

1. **五随机种子后，Core 与等面积 Random 的 VAUQ AUROC 仍然非常接近。** ViLP 上 Core 比 Random 均值低 0.26 个百分点；MMVet 上高 1.48 个百分点；CVBench 上只高 0.23 个百分点。Random AUROC 的种子间样本标准差分别只有 0.10、0.56 和 0.10 个百分点，因此“二者接近”不是单个随机种子的偶然结果。
2. **Core 相对 Blank 的优势不稳定。** MMVet 和 CVBench 上 Core 分别高 3.05 和 2.55 个百分点，但 ViLP 上 Blank 反而高 1.41 个百分点，并同时取得最高 VAUQ AUPR 和 IS AUROC。
3. **CVBench 上 VAUQ 几乎没有超过原始 Entropy。** Core VAUQ AUROC 为 68.77，Entropy AUROC 为 68.74，仅相差约 0.04 个百分点；Random 五种子均值为 68.55。
4. **MMVet 上 Random 的部分指标均值仍优于 Core。** Core 的 VAUQ AUROC 最高，但 Random 五种子的 VAUQ AUPR 均值为 68.92，高于 Core 的 68.47；Random IS AUROC 均值为 67.06，也高于 Core 的 65.66。
5. **视觉信息消融本身可能有效，但注意力定位的额外价值有限。** Core 通常优于完全遮蔽视觉 token 的 Blank，但没有表现出相对等面积 Random 的稳定明显优势。

综合而言，本次严格共享基础回答和 judge 标签的实验不能证明注意力选出的核心视觉 token 明显优于随机视觉 token。结果更符合以下解释：遮蔽一定数量的视觉信息本身就能产生大部分效果，而注意力 top-K 定位只带来很小且依赖数据集的边际变化。

Random seeds 42–46 的方差很小，且五种子均值没有改变单种子实验的排序模式。因此可以更稳健地得出：Core 相对等面积随机遮蔽没有跨数据集稳定且显著的优势；MMVet 上存在有限增益，但 ViLP 上方向相反，CVBench 上差异接近于零。

### 7.2 实验二：正确性与视觉幻觉双标签

#### 7.2.1 标签分布

| 数据集 | A：正确无幻觉 | B：正确有幻觉 | C：错误无幻觉 | D：错误有幻觉 |
|---|---:|---:|---:|---:|
| MMVet |  |  |  |  |
| ViLP |  |  |  |  |

#### 7.2.2 Judge 输出质量与 MMHal-Bench 评分

| 数据集 | 总样本数 | JSON 解析成功数 | JSON/Rating 失败数 | 平均 Rating | Hallucination Rate（Rating < 3） |
|---|---:|---:|---:|---:|---:|
| MMVet | 218 |  |  |  |  |
| ViLP | 600 |  |  |  |  |

平均 Rating 和 Hallucination Rate 均按 MMHal-Bench 官方统计方式报告；其中 Hallucination Rate 是 rating 为 0、1、2 的样本比例。

#### 7.2.3 AUROC/AUPR 结果

| 数据集 | 标签目标 | 条件 | AUROC | AUPR | 95% CI |
|---|---|---|---:|---:|---|
| MMVet | Error | 全部样本 |  |  |  |
| MMVet | Hallucination | 全部样本 |  |  |  |
| MMVet | Hallucination | 仅错误回答 |  |  |  |
| MMVet | Hallucination | 仅正确回答 |  |  |  |
| ViLP | Error | 全部样本 |  |  |  |
| ViLP | Hallucination | 全部样本 |  |  |  |
| ViLP | Hallucination | 仅错误回答 |  |  |  |
| ViLP | Hallucination | 仅正确回答 |  |  |  |

#### 7.2.4 四象限 VAUQ 分数

| 数据集 | 分组 | 样本数 | VAUQ 均值 | VAUQ 中位数 | 四分位距 |
|---|---|---:|---:|---:|---:|
| MMVet | A：正确无幻觉 |  |  |  |  |
| MMVet | B：正确有幻觉 |  |  |  |  |
| MMVet | C：错误无幻觉 |  |  |  |  |
| MMVet | D：错误有幻觉 |  |  |  |  |
| ViLP | A：正确无幻觉 |  |  |  |  |
| ViLP | B：正确有幻觉 |  |  |  |  |
| ViLP | C：错误无幻觉 |  |  |  |  |
| ViLP | D：错误有幻觉 |  |  |  |  |

#### 7.2.5 回归分析

| 变量 | 系数 | 标准误 | $p$ 值 | 95% CI | 解释 |
|---|---:|---:|---:|---|---|
| VAUQ score |  |  |  |  |  |
| Error |  |  |  |  |  |
| Answer length |  |  |  |  |  |
| Dataset/task effects |  |  |  |  |  |

#### 7.2.6 实验二观察

> 待填写。

### 7.3 联合结论

> 待实验完成后填写。建议区分以下三个层次：
>
> 1. VAUQ 是否能区分正确与错误回答；
> 2. 注意力核心区域是否比简单视觉消融更有效；
> 3. VAUQ 在控制回答正确性后是否仍能检测视觉幻觉。

---

## 8. 预期的规范表述

在结果尚未完成前，不预设论文一定错误。若实验支持上述质疑，建议使用如下表述：

> VAUQ 对回答正确性或一般预测不确定性具有一定区分能力，但现有 ground-truth labeling 将所有错误回答定义为幻觉，无法充分区分视觉幻觉与计算、推理、格式等非幻觉错误。若注意力核心遮蔽未稳定优于等面积随机遮蔽或全部视觉 token 消融，则其性能也不能完全归因于核心视觉证据定位。因此，现有实验不足以同时证明 VAUQ 的核心区域选择机制有效且其分数具有视觉幻觉特异性。

若实验不支持质疑，则应如实报告 core 的稳定优势，以及 VAUQ 在固定正确性后对独立幻觉标签仍具有显著区分能力。
