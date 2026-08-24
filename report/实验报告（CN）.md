# 多模态大模型低不确定性幻觉的识别与预测

## 1. 研究问题提出

### 1.1 研究背景与问题定义

多模态大语言模型（Multimodal Large Language Model，MLLM）能够根据图像和文本生成开放式回答，但其输出可能偏离输入证据或客观事实。参考相关研究 [8]，本报告采用如下定义：

> **MLLM 幻觉是指模型在感知与推理过程中生成了与输入视觉内容、问题或上下文相矛盾，无法由其合理推导，或与可验证现实世界知识不一致的事实性内容。**

幻觉不等同于一般错误。计算、公式选择、逻辑推导或选项映射错误，若未引入符合上述条件的事实性内容，均不标记为幻觉；反之，即使最终答案正确，只要感知或推理过程满足上述条件，仍标记为幻觉。本文据此分别定义正确性标签与幻觉标签：

\[
C_i=\mathbb{1}[\text{最终答案与参考答案一致}],\qquad
H_i=\mathbb{1}[\text{感知或推理过程满足上述任一幻觉条件}]。
\]

\(C\) 与 \(H\) 相互独立，可形成四种组合。该设计能够区分“答错但未幻觉”与“答对但含幻觉”两类样本，避免以答案正误替代幻觉标签。

不确定性量化（Uncertainty Quantification，UQ）是指利用生成概率、重复采样、内部表示或模型自评估等信号，为给定输入及其回答计算不确定性分数，以描述模型对该回答缺乏确定性的程度 [10]。本文统一规定：UQ 分数越高，表示模型越不确定。

不确定性与幻觉相关，但二者并不等价。不确定性描述模型预测分布或内部状态的稳定程度，幻觉则描述回答中的事实性内容是否得到输入与可靠知识支持。高不确定性可能来自图像模糊、问题歧义或存在多个合理答案，此时不一定发生幻觉；低不确定性也只表示模型形成了稳定预测，不能保证该预测具有事实依据。已有研究同样表明，UQ 与幻觉之间的关联会随任务、模型和幻觉类型变化 [7]。

模型可能因语言先验、稳定的错误视觉表征或其他系统性偏差，高概率地重复生成同一幻觉。对给定 UQ 方法，本文将这种情况定义为**低不确定性幻觉（Low-Uncertainty Hallucination，LUH）**：

> **回答被独立判定为存在幻觉，但该 UQ 方法仍将其置于所属模型的低不确定性区域。**

LUH 是幻觉检测的漏报，反映该 UQ 方法未能识别模型稳定生成的无依据内容。

按照不确定性水平与是否包含幻觉，可将模型回答分为四类：

![四种幻觉类型](/Users/lexiangrui/Desktop/Uncertainty Quantification of MLLM/report/figures/四种幻觉类型.jpg)

### 1.2 多模态幻觉的评测研究

多模态幻觉的评测经历了从“物体存在性核对”到“开放式回答的证据支持度评估”的演进，评测对象也从最终答案逐步扩展为完整回答内容。早期研究主要关注图像描述中的物体幻觉：Rohrbach 等人提出 CHAIR，通过将生成描述中提及的物体与图像中实际存在的物体逐一比对，计算幻觉物体占全部提及物体的比例 [18]，开创了以“视觉证据核对”为核心的评测范式，但受限于固定的物体类别集合（如 MSCOCO 的 80 类），且结果易受指令设计与描述长度影响。进入多模态大模型时代后，Li 等人提出 POPE，将物体幻觉检测转化为“图像中是否存在某物体”的 Yes/No 轮询二分类任务 [1]：正问题基于图像中真实存在的物体构造，负问题则从未出现物体中采样，并按随机、高频与对抗三种负采样策略划分子集；研究发现，训练数据中高频出现或经常共现的物体更容易被模型以幻觉形式生成，提示多模态幻觉可能源于稳定的语言与共现先验，而非仅来自视觉感知噪声。MME 等综合评测基准进一步将物体存在性、计数、位置、颜色等感知子任务纳入统一的轮询式评测框架 [19]。

随着评测对象的细化，研究者将物体级幻觉归纳为三类：类别幻觉（识别出图像中不存在的物体类别）、属性幻觉（物体类别正确，但对颜色、形状、材质、数量等属性的描述错误）与关系幻觉（物体及其属性均正确，但物体之间的交互或相对位置与图像不符）。上述分类表明，幻觉并不局限于“说了不存在的东西”，也包括“把存在的东西说错”；因此，仅核对最终答案是否匹配参考标准，无法覆盖属性与关系层面的无依据内容。

**开放式幻觉评测。** 轮询式评测难以刻画开放式回答中的复杂幻觉。MMHal-Bench 构建了覆盖对象属性、计数、空间关系等八类问题的开放式问答集，并引入 GPT-4 作为裁判，按 0–6 量表对回答进行分级评分 [8]，将“幻觉检测”从规则匹配转向“由更强模型判断回答是否得到证据支持”。HallusionBench 则从诊断视角出发，将问题划分为 Visual Dependent（必须依赖视觉上下文才能回答）与 Visual Supplement（无视觉输入也可回答，视觉仅起补充或纠正作用）两类，系统考察视觉错觉、语言先验、错误前提与图文冲突等情形 [9]；其 VS 设置直接检验模型在“语言先验与图像证据冲突”时是否被先验主导。此外，GAVIE 等方案验证了 GPT-4 在开放式幻觉评估上的可行性 [20]，HaELM 则尝试训练专门的幻觉检测模型，以降低对通用大模型的依赖 [21]。

**评测范式的演进与本研究的衔接。** 总体来看，多模态幻觉评测从基于规则的物体比对（CHAIR），发展到判别式轮询（POPE、MME），再到由更强模型辅助的开放式评估（MMHal-Bench、GAVIE、HaELM），评测对象也从“最终答案是否匹配”扩展为“完整回答是否得到输入证据支持”。然而，上述工作也表明，单一正确率不能充分描述 MLLM 的可信性：一方面，开放回答可能在结论正确时附带错误的视觉细节；另一方面，模型也可能在视觉观察完全正确时因推理或计算失误而答错，此类错误在 1.1 节的定义下并不构成幻觉。因此，要研究 UQ 是否能够预测幻觉，必须使用独立于正确性的幻觉标注，并保留模型的视觉观察、推理过程和最终答案，才能定位幻觉实际发生的位置。

### 1.3 从语言模型不确定性到多模态不确定性

#### 1.3.1 大语言模型的不确定性量化方法

经典机器学习通常区分数据不确定性（aleatoric uncertainty）与模型不确定性（epistemic uncertainty）；针对开放式语言生成，相关综述还将不确定性细分为输入、推理、参数和预测四个维度 [10]。按照分数的取得方式，现有大语言模型 UQ 方法可概括为以下五类。

**基于生成概率的方法。** 此类方法利用 logits 计算 token 熵、预测熵、序列负对数似然或困惑度（PPL）等指标 [11]。其计算成本较低，但生成概率只反映回答在模型自身分布下的可能性，不直接代表事实正确性。

**基于模型自评估的方法。** P(True) 与 verbalized confidence 通过再次询问模型答案是否正确，或让模型直接报告置信度来估计不确定性 [12]。这类方法适用于闭源模型，但结果容易受到提示方式影响，也可能重复模型原有的知识偏差。

**基于多次采样一致性的方法。** 这类方法通过重复生成，依据答案频率、文本相似度或语义一致性估计不确定性。Semantic Entropy 对采样回答进行语义聚类并计算簇熵 [2]；Kernel Language Entropy 使用语义核描述连续相似性 [13]；Eccentricity 与 Degree 基于回答关系图的中心性度量 [14]，EigenScore 则基于回答嵌入协方差矩阵的谱特征 [15]。它们适合开放式回答，但若模型稳定重复同一错误，仍可能给出低不确定性。

**基于内部表示的方法。** 此类方法使用隐藏状态、注意力或中间层激活计算距离、密度，或训练轻量探针预测答案可靠性。Azaria 和 Mitchell 通过训练线性探针证明内部状态包含与陈述真实性相关的信息 [16]，但这类方法通常要求白盒访问，且结果依赖模型层、任务和监督标签。

**基于扰动与集成的方法。** SPUQ 通过扰动输入并聚合不同条件下的回答变化估计不确定性 [17]；Monte Carlo dropout、贝叶斯近似（Bayesian approximation）与模型集成则通过近似参数后验估计模型不确定性 [10]。此类方法能够发现局部不稳定性，但计算成本较高，并依赖扰动是否真正保持原任务含义。

上述五类方法分别测量生成概率、自评估置信度、输出一致性、内部状态与干预稳定性，不能未经验证便直接解释为幻觉概率。

#### 1.3.2 多模态大语言模型的不确定性量化方法

上述文本侧 UQ 方法可以直接用于 MLLM 的文本输出，却难以区分不确定性来自视觉、文本、多模态对齐还是语言解码。近期方法因此开始显式利用视觉输入与跨模态内部信号。

**VL-Uncertainty** 对问题进行改写并对图像施加高斯模糊，再计算扰动输入下回答的语义簇熵 [3]。它把 Semantic Entropy 从固定输入扩展到图文语义等价扰动构成的输入邻域，但图文同时变化使分数来源难以归因，模糊也可能删除任务所需的视觉信息。

**VAUQ** 将原图预测熵与遮蔽高注意力视觉 token 后的熵增量结合，以衡量回答是否依赖视觉证据 [4]。该方法显式考虑视觉信息，但注意力权重高并不必然表示该区域是回答所依赖的证据，其主要评价目标仍是最终答案正误。

**UMPIRE** 将采样回答在 MLLM 内部语义空间中的离散程度与生成概率结合，同时衡量回答多样性和多模态一致性 [5]。不过，该方法的不确定性主要来自采样间的回答变化，并以答案正确性作为主要风险目标。

**Visual Semantic Entropy** 只扰动图像、固定文本问题，再依据回答语义原型之间的加权距离计算不确定性 [6]。该研究指出，过度自信的视觉表示会使采样回答在语义上高度一致，从而让普通 Semantic Entropy 低估视觉不确定性，但仍主要评价答案错误与视觉歧义。

总体而言，多模态 UQ 已从复用文本分数发展到显式探测视觉证据和跨模态表示，但这些分数能否识别 1.1 节严格定义的幻觉，仍需使用独立幻觉标签加以验证。

### 1.4 现有研究的共性问题

综合上述文献，本文认为现有 UQ 研究在预测低不确定性幻觉时存在四个相互关联的缺口

| 缺口 | 现有研究中的常见做法 | 对低不确定性幻觉的影响 |
|---|---|---|
| 机制假设 | 将高生成概率或多次采样一致性解释为可靠 | 稳定复现同一幻觉时，PPL、Semantic Entropy 等方法可能同时给出低不确定性 |
| 评价目标 | 以最终答案正确/错误作为“幻觉”或风险标签 | 无法区分“答错但未幻觉”与“答对但含幻觉”，UQ 的有效性可能被错误率替代 |
| 测量对象 | 只对最终答案计算概率或语义分散度等分数 | 视觉观察或推理中引入的无依据内容可能没有反映在简短最终答案的 UQ 分数中 |
| 扰动有效性 | 将图像模糊、文本改写等视为语义等价扰动 | 扰动可能破坏任务信息或引入提示敏感性，使高分反映人为信息退化而非原始幻觉风险 |

这一判断也得到近期系统研究的支持。Agnimo 等人比较了 46 种 UQ 估计量，发现不确定性与幻觉的关联随数据集（其中编码了任务与幻觉类型）和模型显著变化，而且不存在单一跨数据集始终最优的方法；该研究进一步指出，既有工作经常以宽泛的正确性、鲁棒性或校准指标替代幻觉特定目标，因而不应把 UQ 分数直接视为通用幻觉检测器 [7]。不过，该研究面向文本 LLM，尚未系统考察视觉证据、语言先验与多模态融合造成的低不确定性幻觉。

### 1.5 研究问题

基于上述调研，本文提出如下核心研究问题：

> **RQ：如何利用不确定性量化方法识别并预测多模态大语言模型中的低不确定性幻觉？**

本文所研究的低不确定性幻觉不是一种新的幻觉类型，而是可能被现有 UQ 方法遗漏的一类样本：回答中存在图像、问题、上下文或事实不支持的内容，但该方法的生成概率、采样一致性或内部语义表示分数仍表现出较低不确定性。回答这一研究问题需要识别同时被 PPL、Semantic Entropy 和 UMPIRE 判为低不确定性的幻觉样本（即三种基线共同漏检的共识 LUH），再针对这些样本的共同特征与三种方法的共同失效模式设计新的不确定性量化方法。为了计算区分指标，为每个模型配置 200 条正类与等量的低不确定性非幻觉负类，使比较集中在同一低不确定性区域。因此，本文的研究分为前后衔接的两个实验。

### 1.6 本文研究方法与实验路线

为解决上述核心问题，本文设计了前后承接的“两阶段”递进式研究路线（图 1.1）。

![研究路线图](/Users/lexiangrui/Desktop/Uncertainty Quantification of MLLM/report/figures/研究路线图.jpg)

**图 1.1：本文两阶段研究路线与技术流程图。**

具体研究流程分为前后衔接的两个实验：

1. **实验一：UQ 评估框架设计与 LUH 盲区挖掘（第 2 ~ 3 节）**
   - 统一使用格式 LoRA 规范 3 个不同架构的开源 MLLM（LLaVA-1.5、Qwen2.5-VL、InternVL3.5）输出三段式 XML 结构，并由多模态 LLM Judge 在同一次调用中独立标注答案正确性 \(C\) 与幻觉 \(H\)；
   - 在 3 个评测基准（ViLP、HallusionBench、MM-Vet 共 2,247 样本）上系统评估 PPL、Semantic Entropy (SE) 与 UMPIRE 三类代表性 UQ 方法，定量揭示传统方法在低不确定性幻觉（LUH）上的系统性失效；
   - 为每个模型提取 200 条被三种基线共同漏检的 LUH 正类与 200 条分数结构匹配的非幻觉负类，构建基线难以区分（AUROC 为 0.417 ~ 0.502）的 400 样本平衡困难测试子集。

2. **实验二：ERA 早期推理归因方法设计与验证（第 4 ~ 5 节）**
   - 针对基线在 LUH 上失效的深层机理（模型自回归生成推理链时极易产生内在自信，导致最终决策严重脱离真实视觉与问题输入），提出 **ERA（Early Rationale Attribution，早期推理归因）** 不确定性量化方法；
   - 在浅层解码器（Layer 0-1）中将输入序列解耦为 5 个连续语义分区（图像 \(I\)、问题 \(Q\)、视觉描述 \(V\)、推理链 \(R\) 与最终答案 \(A\)），通过单次前向传播计算答案流向自生成推理链相较于外部真实输入的相对依赖比率 \(U_{\mathrm{ERA}}\)；
   - 完全复用实验一的困难子集与评估协议，检验 ERA 能否有效从低不确定性非幻觉中识别被传统方法漏检的 LUH 样本，显著提升低不确定性幻觉的判别能力。


---

## 2 实验一：UQ评估框架设计

### 2.1 实验设计

#### 2.1.1 评测模型与 LoRA 格式微调

**评测模型与本地部署。** 本实验评测三个开源多模态大模型（表 2.1）：LLaVA-1.5-7B、Qwen2.5-VL-7B-Instruct 与 InternVL3.5-8B，分别代表经典视觉语言基线、新一代原生多模态架构与第三种开源架构。三个模型均采用本地离线部署：vLLM 负责批量生成并保存精确 token，Hugging Face 对相同 token 做 teacher forcing 回放，提取 UQ 所需概率与最后一层隐藏状态。

**表 2.1：三个评测模型。**

| 模型（Hugging Face） | 参数量 | 架构特点 | 选择理由 |
|---|---|---|---|
| LLaVA-1.5-7B（`llava-hf/llava-1.5-7b-hf`） | 约 7B | CLIP 视觉编码器 + 投影 + LLM 主干的经典两段式架构 | 经典开源视觉语言基线 |
| Qwen2.5-VL-7B-Instruct（`Qwen/Qwen2.5-VL-7B-Instruct`） | 约 7B | 新一代原生多模态架构，动态分辨率，OCR 与视觉推理较强 | 新一代开源指令模型 |
| InternVL3.5-8B（`OpenGVLab/InternVL3_5-8B`） | 约 8B | 视觉编码器、投影层与语言模型组成的多模态架构 | 第三种模型架构，扩大覆盖 |

**LoRA 格式微调。** 预实验表明，三个模型在 greedy 与随机采样下都可能出现标签遗漏、标签不闭合、顺序错误或 XML 外续写；而 Judge 与 UQ 必须可靠分离视觉证据、推理与最终答案，因此对三个模型分别训练只针对回答组织方式的 LoRA adapter。训练数据仅来自 VQAv2 train2014 构造的 XML 监督数据：由多模态教师模型为真实 COCO 图像生成"视觉证据—推理—最终答案"三段内容并统一转换为 `<vision>/<reasoning>/<answer>` 单行 XML 格式，经人工抽样核对后固定为 4,000 条训练与 1,000 条验证；三个正式评测集完全不进入训练数据以避免泄漏。三个模型使用完全相同的训练配置（表 2.2）：仅以 LoRA 调整语言模型部分的注意力投影（`q_proj`、`v_proj`），**冻结视觉编码与多模态对齐部分**——即格式微调不改变模型的视觉感知与图文对齐能力，从而避免微调本身对幻觉率与正确率造成混淆。为检验该冻结策略是否真正隔离了格式微调的影响，对比实验（解冻视觉编码或全参数微调对幻觉率/正确率的影响）见附录 A.8。三个 adapter 的验证 loss 分别为 0.765（LLaVA）、0.693（Qwen）、0.633（InternVL）；数据构造、训练配置与验收细节见附录 A.7。

**表 2.2：三模型格式 LoRA 训练配置。三个模型使用相同的数据划分、目标回答与超参数。**

| 配置项 | 取值 |
|---|---|
| 训练数据 | VQAv2 train2014 构造，4,000 训练 / 1,000 验证 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| 目标模块 | 语言模型 `q_proj`、`v_proj` |
| 学习率 / 训练轮数 | 2×10⁻⁴ / 1 |
| 验证 loss | 0.765（LLaVA）/ 0.693（Qwen）/ 0.633（InternVL） |

**回答示例。

> 问题：Modern drones typically have four propellers. How many propellers does the drone in the picture have?
>
> 回答：`<vision>The image shows a large drone hovering over a futuristic cityscape. The drone has a central body with four distinct propellers, each attached to a rotor arm extending outward.</vision><reasoning>The drone in the picture clearly has four propellers, which is the standard configuration for modern drones.</reasoning><answer>4</answer>`

<img src="figures/vilp_0_case1.jpg" alt="示例问题图像：ViLP vilp-0-case1（无人机）。" style="zoom: 33%;" />

#### 2.1.2 评测数据集

三个数据集均使用 Hugging Face 发布的完整评测数据，样本数按实际送入模型的"问题实例"计数（表 2.3）。三个数据集共 2,247 个问题实例，三个模型在完全相同的问题清单上评测，共得到 6,741 个"问题—模型"主回答实例。

**表 2.3：评测数据集概览。**

| 数据集 | 问题实例数 | 数据与任务类型 | 主要考察对象 |
|---|---:|---|---|
| ViLP | 900 | 300 个问题 × 3 组图像—答案（QIA） | 语言先验与事实/反先验视觉证据的冲突 |
| HallusionBench | 1,129 | 951 个图像问题 + 178 个非图像问题 | 视觉错觉、语言先验、错误前提与图文冲突 |
| MM-Vet | 218 | 单图开放式视觉问答，六类核心能力 | 开放回答中的视觉事实错误与复合能力失败 |

三个数据集详细信息见附录 A.1。

#### 2.1.3 回答生成协议

被测模型统一采用固定单行 XML 协议回答：

```text
<vision>Relevant visual evidence.</vision><reasoning>Brief reasoning based on the evidence.</reasoning><answer>Concise final answer.</answer>
```

回答指令、图像与问题放在同一条 user message 中，不创建 system message。每个输入生成两条流水线：

- **greedy 主回答**：`do_sample=false`，用于正确性/幻觉标注与 Perplexity 计算；
- **K=10 随机采样回答**：`do_sample=true, temperature=1.0`，每条采样回答最多进行 50 次 XML 格式拒绝重采样，供 Semantic Entropy 与 UMPIRE 使用。

同一批采样回答同时供 Semantic Entropy 与 UMPIRE 使用，避免采样差异。生成时记录最终回答 token 的逐 token log probability 与最后一层隐藏状态（用于 UMPIRE），回答中XML标签错误的样本在数据清洗中会被排除，最终的有效样本统计见附录 A.5。

#### 2.1.4 正确性与幻觉的独立标注

每条 greedy 主回答调用一次多模态 LLM Judge（`gpt-5.4`），在同一次调用中输出两个独立标签：

1. **正确性 \(C\)**：只比较 `<answer>` 与数据集参考答案的语义一致性；
2. **幻觉 \(H\)**：只评价 `<vision>` 与 `<reasoning>` 是否包含图像、问题、上下文或可验证事实不支持的内容，按 0–6 量表打分，`rating < 3` 判为存在幻觉，并给出 `vision_hallucination` / `reasoning_hallucination` 类型。

Judge 的评分指令与输出格式见附录 A.6。两个标签作用于回答的不同部分且相互独立，允许四种组合，使后续 UQ 评估可以同时覆盖错误检测与幻觉检测两个目标。

#### 2.1.5 三种基线 UQ 方法

三种方法均规定"分数越高，不确定性越高"，正式检测对象统一为最终答案部分的内容。

**Perplexity。** PPL 是语言模型中常用的序列概率不确定性度量，通过模型对已生成答案的条件概率衡量预测的不确定性。给定图文问题及提示组成的条件输入 \(q\)，设最终答案部分的 token 序列为 \(y=(w_1,\ldots,w_N)\)，自回归模型给出的序列概率为

\[
p_M(y\mid q)=\prod_{j=1}^{N}p_M(w_j\mid q,w_{<j}),
\]

其长度归一化负对数似然与困惑度分别为

\[
\operatorname{NLL}(y\mid q)=-\frac{1}{N}\sum_{j=1}^{N}\log p_M(w_j\mid q,w_{<j}),
\qquad
\operatorname{PPL}(y\mid q)=\exp\!\left(\operatorname{NLL}(y\mid q)\right).
\]

PPL 是 token 层面的局部不确定性：若模型在给定前缀下普遍只赋予实际答案 token 较低概率，则 PPL 较高。它不比较多次生成的语义是否一致，因此构成本实验的单回答概率基线。

**Semantic Entropy。** SE 将不确定性定义在"回答含义"而非表面 token 序列上，避免把同义表述误判为相互矛盾的答案。从 \(p_M(\cdot\mid q)\) 采样 \(K=10\) 条最终答案 \(y_1,\ldots,y_K\)，按**相对于问题上下文的双向蕴含**关系将它们划分为语义簇 \(\mathcal C=\{c_1,\ldots,c_m\}\)：仅当 \(y_a\) 蕴含 \(y_b\) 且 \(y_b\) 蕴含 \(y_a\) 时，二者才归入同一簇。为减弱回答长度对有限样本概率质量的直接支配，先计算每条采样答案的长度归一化 log probability，再以 softmax 归一化作为其概率质量：

\[
\ell_i=\frac{1}{|y_i|}\sum_j\log p_M(w_{i,j}\mid q,w_{i,<j}),
\qquad
\tilde p_i=\frac{\exp(\ell_i)}{\sum_k\exp(\ell_k)},
\]

簇概率与语义熵为

\[
p(c)=\sum_{i:y_i\in c}\tilde p_i,
\qquad
\widehat{\operatorname{SE}}(q)=-\sum_{c\in\mathcal C}p(c)\log p(c).
\]

当采样回答集中于一个或少数几个语义簇时，SE 低；当模型在多个彼此不同的答案含义之间分散时，SE 高。该熵以归一化概率质量而非簇内采样次数计算，避免回答长度对有限样本概率估计的直接影响。

**UMPIRE。** UMPIRE（*Uncertainty using Model Probability Indicators and Response Embeddings*）把两类信号合并为一个训练无关的多模态不确定性分数：采样回答在模型内部语义/多模态表征空间中是否分散，以及模型对每个回答在给定图文输入下的条件概率是否低。对同一输入 \(q\) 的 \(K\) 条采样最终答案 \(y_i\)，令 \(\phi_i\in\mathbb{R}^d\) 为被测 MLLM 最后一层在最终回答 token 处提取并 \(\ell_2\) 归一化后的回答表征，\(\Phi=[\phi_1^\top;\ldots;\phi_K^\top]\in\mathbb{R}^{K\times d}\)。回答的模型条件概率及其不一致度定义为

\[
p_i=p_M(y_i\mid q)=\prod_{j=1}^{|y_i|}p_M(w_{i,j}\mid q,w_{i,<j}),
\qquad
c_i=\exp\!\bigl(\alpha(1-p_i)\bigr),
\]

其中 \(\alpha\geq0\) 控制不一致度项的权重；\(p_i\) 越低，表示该回答越不受模型自身给定的全部输入模态支持，因而 \(c_i\) 越大。令 \(C=\operatorname{diag}(c_1,\ldots,c_K)\)、\(\epsilon>0\) 为保证数值非退化的微小常数，则 UMPIRE 的不确定性为

\[
\operatorname{UMPIRE}(q)=\frac{1}{2K}\log\det\!\left[C\left(\Phi\Phi^\top+\epsilon I_K\right)C\right],
\]

该式可分解为语义体积项 \(\frac{1}{2K}\log\det(\Phi\Phi^\top+\epsilon I_K)\) 与平均不一致度项 \(\frac{\alpha}{K}\sum_i(1-p_i)\)，分别对应回答表征张成的平方体积与平均不一致度。因此 UMPIRE 同时对"模型给出多种相距较远的答案"以及"回答变化不大但模型对这些回答普遍缺乏内部支持"给出高分；后者是仅依赖回答间语义离散度的方法可能漏检的情形。

#### 2.1.6 评估指标

所有指标在每个“模型 × 数据集”单元格内独立计算，不跨模型或数据集比较原始分数。

**标签基础。** 正确性标签 \(C_i\) 与幻觉标签 \(H_i\) 的总体水平由两个比率刻画：

\[
\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}C_i,\qquad
\mathrm{Hallucination\ Rate}=\frac{1}{N}\sum_{i=1}^{N}H_i,
\]

其中 \(N\) 为单元格内有效样本数。Accuracy 描述最终答案与参考答案一致的比例，Hallucination Rate 描述视觉观察或推理中含无依据内容的样本比例。由于 \(C\) 与 \(H\) 独立，二者并不互补：正确样本中可能含幻觉，错误样本中也可能无幻觉。

**排序能力。** 为评估 UQ 分数能否区分正负样本（以错误 \(E=1-C\) 或幻觉 \(H\) 为正类），使用三个指标。给定分数 \(s_i\) 与二值标签 \(y_i\)：

- **AUROC（Area Under the Receiver Operating Characteristic Curve，ROC 曲线下面积）**。将样本按分数降序排列，以每个分数为阈值计算真正例率（TPR）与假正例率（FPR），ROC 曲线为 FPR–TPR 的关系曲线，AUROC 为其面积。等价地，AUROC 等于随机取一个正样本与一个负样本时，正样本分数高于负样本的概率。

- **AUPRC（Area Under the Precision-Recall Curve，PR 曲线下面积）**。以每个分数为阈值计算查准率（Precision）与召回率（Recall），PR 曲线为 Recall–Precision 的关系曲线，AUPRC 为其面积，等价于平均查准率（Average Precision）。与 AUROC 不同，AUPRC 对正类比例敏感：正类越稀有，随机基准的 AUPRC 越低。

- **PRR（Prediction Rejection Ratio，预测拒绝比）**。衡量按分数从高到低“拒绝”（即人为不信任）一部分样本后，剩余样本精度的提升幅度。定义 retained-precision 曲线为拒绝比例 \(r\) 下剩余样本的正类比例，其曲线下面积为 \(A\)；随机排序的面积为 \(A_{\mathrm{rand}}=1-p_+\)（\(p_+\) 为正类比例），oracle 排序（按标签降序）的面积为 \(A_{\mathrm{oracle}}\)，则
\[
\mathrm{PRR}=\frac{A-A_{\mathrm{rand}}}{A_{\mathrm{oracle}}-A_{\mathrm{rand}}}.
\]
PRR 取值 0 表示与随机排序无异，1 表示达到 oracle 排序；它回答了“若只信任分数最高的样本，能多准确地保留正类”这一实际部署问题。

## 3 实验一：评估结果分析与低不确定性子集提取

### 3.1 标签基础：正确性与幻觉的独立结构

本节报告三个模型在三个数据集上的正确性与幻觉标签分布，验证 1.1 节标签设计的必要性：正确性与幻觉是否为两个独立构念，直接决定后续 UQ 评估是否必须分别在两个标签上计算指标。

#### 3.1.1 各单元格的准确率与幻觉率

表 3.1 给出三个模型在三个数据集上的有效样本数、Accuracy 与 Hallucination Rate。有效样本为生成、Judge 与 UQ 三路均成功对齐的记录，共 6,667 条"问题—模型"样本；在 6,741 条原始生成记录中排除 74 条 XML 分段解析失败样本，排除记录见附录 A.5。

**表 3.1：各单元格的准确率与幻觉率（95% bootstrap 置信区间）。**

| 模型 | 数据集 | N | Accuracy | Hallucination Rate |
|---|---|---:|---:|---:|
| LLaVA | ViLP | 886 | 52.1% (49.3, 55.0) | 52.0% (48.9, 55.2) |
| LLaVA | HallusionBench | 1,108 | 51.3% (48.5, 54.1) | 77.6% (74.6, 80.7) |
| LLaVA | MM-Vet | 216 | 22.7% (17.1, 28.7) | 69.9% (63.9, 75.9) |
| Qwen | ViLP | 898 | 63.6% (60.4, 66.7) | 22.6% (20.1, 25.2) |
| Qwen | HallusionBench | 1,099 | 66.1% (62.6, 69.6) | 42.6% (38.0, 47.1) |
| Qwen | MM-Vet | 218 | 53.2% (46.8, 60.1) | 29.4% (23.4, 35.8) |
| InternVL | ViLP | 900 | 61.8% (59.0, 64.7) | 25.4% (22.7, 28.3) |
| InternVL | HallusionBench | 1,126 | 71.0% (67.6, 74.5) | 38.6% (34.4, 43.4) |
| InternVL | MM-Vet | 216 | 55.1% (48.1, 62.0) | 26.4% (20.8, 32.9) |

三个数据集呈现不同的失效结构：HallusionBench 的幻觉率在三个模型上均为最高（LLaVA 高达 77.6%），MM-Vet 对 LLaVA 的正确性挑战最大（Accuracy 仅 22.7%），ViLP 则集中考察语言先验与视觉证据的冲突。同一数据集上不同模型的幻觉率差异很大：HallusionBench 上 LLaVA 为 77.6%，而 InternVL 为 38.6%。这说明幻觉率是"模型 × 数据集"共同作用的结果，不能由单一方面概括。

#### 3.1.2 正确性—幻觉四象限与独立性

为检验正确性与幻觉是否相互独立，表 3.2 报告每个单元格的 C×H 四象限构成、错误样本内幻觉率（H|E=1）、正确样本内幻觉率（H|C=1）与四分点相关系数 φ。

**表 3.2：正确性—幻觉四象限与独立性。H|E=1 与 H|C=1 分别表示错误样本与正确样本中的幻觉率。**

| 模型 | 数据集 | C=1,H=0 | C=1,H=1 | C=0,H=0 | C=0,H=1 | H\|E=1 | H\|C=1 | φ |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaVA | ViLP | 38.7% | 13.4% | 9.3% | 38.6% | 80.7% | 25.8% | −0.549 |
| LLaVA | HallusionBench | 19.8% | 31.5% | 2.6% | 46.1% | 94.6% | 61.4% | −0.398 |
| LLaVA | MM-Vet | 16.7% | 6.0% | 13.4% | 63.9% | 82.6% | 26.5% | −0.512 |
| Qwen | ViLP | 58.7% | 4.9% | 18.7% | 17.7% | 48.6% | 7.7% | −0.471 |
| Qwen | HallusionBench | 48.8% | 17.3% | 8.6% | 25.3% | 74.5% | 26.2% | −0.463 |
| Qwen | MM-Vet | 46.3% | 6.9% | 24.3% | 22.5% | 48.0% | 12.9% | −0.385 |
| InternVL | ViLP | 57.2% | 4.6% | 17.3% | 20.9% | 54.7% | 7.4% | −0.527 |
| InternVL | HallusionBench | 55.2% | 15.8% | 6.1% | 22.8% | 78.8% | 22.2% | −0.527 |
| InternVL | MM-Vet | 51.9% | 3.2% | 21.8% | 23.1% | 51.5% | 5.9% | −0.515 |

幻觉与错误并不重合：错误样本中的幻觉率（H|E=1）在 48.0%–94.6% 之间，正确样本中的幻觉率（H|C=1）在 5.9%–61.4% 之间；φ 均为负值（−0.55 至 −0.39），表示正确性与幻觉呈中等负相关——错误样本中幻觉更常见，但两者远非等价。以 LLaVA/HallusionBench 为例，即使最终答案正确，仍有 61.4% 的样本在视觉观察或推理中包含幻觉；反过来，该单元格也有 5.4% 的错误样本未被判为幻觉。若以答案正误替代幻觉标签，前一类样本会被系统性漏掉。

这一结果验证了 1.1 节标签设计的必要性：正确性与幻觉必须作为两个独立标签分别标注。UQ 评估也只有在这两个标签上分别计算指标，才能回答"分数能否预测错误"与"分数能否预测幻觉"这两个不同的问题。

### 3.2 三种 UQ 方法的检测能力

本节评估 Perplexity、Semantic Entropy 与 UMPIRE 三种基线分数对错误（\(E=1-C\)）与幻觉（\(H\)）的检测能力。所有指标在每个"模型 × 数据集"单元格内独立计算；为便于跨数据集比较，另以每个模型跨三个数据集的宏平均（等权）报告总体水平。逐单元格的 AUROC 及 95% 置信区间见附录 A.2。

#### 3.2.1 错误检测

表 3.3 报告以错误 \(E\) 为正类的检测能力。三个模型上三种方法的宏平均 AUROC 在 0.64–0.77 之间，均优于随机水平（0.5）。Semantic Entropy 在三个模型上都是最优方法（0.680 / 0.733 / 0.771），UMPIRE 次之（0.669 / 0.693 / 0.714），Perplexity 最低（0.642 / 0.668 / 0.685）。按 9 个单元格的排名统计（附录 A.2），Semantic Entropy 在 8/9 格取得错误检测最优，UMPIRE 在 1/9 格最优，Perplexity 无任何一格最优。

**表 3.3：错误检测的宏平均 AUROC（每模型跨三数据集等权；95% bootstrap 置信区间）。**

| 模型 | Perplexity | Semantic Entropy | UMPIRE |
|---|---:|---:|---:|
| LLaVA | 0.642 (0.609, 0.672) | 0.680 (0.650, 0.710) | 0.669 (0.638, 0.697) |
| Qwen | 0.668 (0.639, 0.695) | 0.733 (0.706, 0.757) | 0.693 (0.667, 0.718) |
| InternVL | 0.685 (0.654, 0.712) | 0.771 (0.745, 0.794) | 0.714 (0.686, 0.739) |

#### 3.2.2 幻觉检测

表 3.4 报告以幻觉 \(H\) 为正类的检测能力。整体水平明显低于错误检测：宏平均 AUROC 仅为 0.55–0.71。Semantic Entropy 在三个模型上仍为最优（0.615 / 0.651 / 0.707），UMPIRE 次之（0.611 / 0.602 / 0.648），Perplexity 最弱（0.578 / 0.553 / 0.602）。按 9 格排名，Semantic Entropy 在 8/9 格取得最优，Perplexity 在 1/9 格最优，UMPIRE 无一格最优。个别单元格仍接近随机，例如 Qwen/ViLP 上 Perplexity 的 AUROC 仅为 0.528。

**表 3.4：幻觉检测的宏平均 AUROC（每模型跨三数据集等权；95% bootstrap 置信区间）。**

| 模型 | Perplexity | Semantic Entropy | UMPIRE |
|---|---:|---:|---:|
| LLaVA | 0.578 (0.541, 0.612) | 0.615 (0.581, 0.648) | 0.611 (0.575, 0.645) |
| Qwen | 0.553 (0.516, 0.590) | 0.651 (0.617, 0.683) | 0.602 (0.566, 0.640) |
| InternVL | 0.602 (0.564, 0.636) | 0.707 (0.677, 0.735) | 0.648 (0.616, 0.676) |

三种方法对幻觉的预测能力都弱于对错误的预测能力，且该差距在三个模型上一致存在。由于三种分数只由最终答案计算，而幻觉标签只评价视觉观察与推理部分，这一结果应解释为"答案不确定性对幻觉的代理预测能力"，而非对幻觉文本的直接测量：幻觉中确有与答案不确定性相关的成分，但远不足以将 UQ 分数直接视为幻觉概率。

#### 3.2.3 错误检测与幻觉检测的目标差距

为量化"分数预测错误"与"分数预测幻觉"之间的差距，表 3.5 报告每个"模型 × 方法"的 \(\mathrm{AUROC}(H)-\mathrm{AUROC}(E)\) 配对差（同一 bootstrap 重采样上配对计算，负值表示幻觉检测弱于错误检测）。

**表 3.5：幻觉检测与错误检测的 AUROC 差距（负值表示幻觉检测更弱）。**

| 模型 | Perplexity | Semantic Entropy | UMPIRE |
|---|---:|---:|---:|
| LLaVA | −0.064 | −0.065 | −0.058 |
| Qwen | −0.115 | −0.082 | −0.091 |
| InternVL | −0.083 | −0.064 | −0.066 |

三个模型上所有方法的宏平均目标差距均为负值，即幻觉检测一致弱于错误检测；其中 Qwen 的差距最大（−0.082 至 −0.115），LLaVA 与 InternVL 的差距约为 −0.058 至 −0.083。逐单元格看（附录 A.2），27 个"单元格 × 方法"差距的平均为 −0.076，范围为 [−0.193, 0.037]；最大负差距出现在 LLaVA/MM-Vet 的 Perplexity（−0.193），而 LLaVA/HallusionBench 上三种方法的差距均略为正。这说明"分数预测幻觉"的损失并非均匀分布，而是集中在特定数据与能力类型上。这进一步表明：以答案不确定性作为幻觉代理存在结构性损失，识别低不确定性幻觉需要超越答案空间的信号。

### 3.3 低不确定性幻觉盲区

3.2 节揭示了一个核心矛盾：三种基线分数对幻觉的预测能力系统性弱于对错误的预测能力。本节先分析这一差距的方法论根源（3.3.1），再量化"低不确定性幻觉"这一盲区的规模（3.3.2），最后以典型案例说明其具体形态（3.3.3）。

#### 3.3.1 不确定性量化预测幻觉的瓶颈分析

三种基线分数都依赖一个共同的机制假设：模型对不确定的回答会表现出概率偏低、采样分散或内部表征不稳定，因此分数能够反映回答缺乏依据的程度。然而这一机制的前提是模型的预测分布在"有依据"与"无依据"之间存在可观测的差异，而该前提在低不确定性幻觉上并不成立。

瓶颈在于**模型过度自信会使采样信号失效**。当模型对某个幻觉内容形成稳定的高置信预测时，其采样分布高度集中：概率质量几乎全部落在同一答案上，多次采样得到语义完全一致的输出，内部表征也趋于收敛。此时三种分数全部指向"极度确定"——PPL 接近 1 表示模型对答案几乎赋予全部概率，Semantic Entropy 接近 0 表示采样语义完全一致，UMPIRE 的语义体积与不一致度同时趋低。分数把这种"稳定"误读为"可靠"，而事实上模型只是稳定地、自信地复现同一个无依据内容。换言之，不确定性量化只能测量模型"自己有多确定"，无法测量"该不该确定"；当模型对错误内容过度自信时，采样一致性越高，分数反而越坚定地给出低不确定性，幻觉被系统性漏检。

这一机制也解释了 3.2.3 中目标差距的来源：幻觉检测弱于错误检测，并非指标选择问题，而是答案空间信号对"过度自信的错误"原理性失效。识别这类样本需要超越"答案是否稳定"的视角，考察"答案稳定时内容是否有证据支持"。

#### 3.3.2 漏检规模量化

为量化上述瓶颈导致的漏检规模，定义低不确定性幻觉占比：对每个模型（三数据集合并的全体已评估样本），以非幻觉样本（H=0）分数分布的 \(\alpha\) 分位数 \(Q_\alpha(H_0)\) 为界，统计幻觉样本（H=1）中分数不超过该界的比例 \(\mathrm{luh\_share}(\alpha)=P(s_{H_1}\le Q_\alpha(H_0))\)。取 \(\alpha\in\{0.25,0.50\}\) 两档，结果见表 3.6。

**表 3.6：低不确定性幻觉占比 \(\mathrm{luh\_share}(\alpha)\)（95% CI）。**

| 模型 | 方法 | α=0.25 | α=0.50 |
|---|---|---:|---:|
| LLaVA | Perplexity | 0.187 (0.162, 0.226) | 0.422 (0.384, 0.463) |
| LLaVA | Semantic Entropy | 0.137 (0.107, 0.180) | 0.444 (0.393, 0.495) |
| LLaVA | UMPIRE | 0.166 (0.135, 0.198) | 0.398 (0.349, 0.450) |
| Qwen | Perplexity | 0.238 (0.189, 0.288) | 0.494 (0.433, 0.545) |
| Qwen | Semantic Entropy | 0.222 (0.188, 0.255) | 0.286 (0.244, 0.341) |
| Qwen | UMPIRE | 0.161 (0.121, 0.199) | 0.374 (0.328, 0.417) |
| InternVL | Perplexity | 0.166 (0.104, 0.219) | 0.449 (0.398, 0.509) |
| InternVL | Semantic Entropy | 0.158 (0.127, 0.189) | 0.268 (0.211, 0.315) |
| InternVL | UMPIRE | 0.062 (0.039, 0.091) | 0.312 (0.265, 0.364) |

以 \(\alpha=0.25\) 为例：即使只取非幻觉分数分布的最低 25% 区域，仍有 6.2%–23.8% 的幻觉样本落入其中；InternVL/UMPIRE 最低（6.2%），Qwen/Perplexity 最高（23.8%）。当阈值放宽到 \(\alpha=0.50\)，漏检规模扩大到 26.8%–49.4%，其中 Qwen/Perplexity 已接近一半。Semantic Entropy 在 Qwen 与 InternVL 的 \(\alpha=0.50\) 上盲区最小（0.286 / 0.268），而 LLaVA 上则是 UMPIRE 最小（0.398）；因此不存在一个在所有模型和阈值上都一致消除盲区的方法。总体而言，低不确定性幻觉仍是规模可观的系统性漏检，与 3.3.1 的"稳定但缺乏证据"机制分析吻合。

#### 3.3.3 典型 LUH 案例分析

本节给出三个典型 LUH 案例，分别对应语言先验凌驾视觉、视觉读数错误与"答案正确但含幻觉"三种情形。三者在三维基线分数的综合百分位上都属于低不确定性幻觉，并均被选入 3.4 节的困难子集正类。下述回答、分数与 Judge 记录均取自集群上的最新正式运行结果。

**案例 1（LLaVA / ViLP，语言先验凌驾视觉证据）。** 问题句首给出"袋鼠以跳跃著称"的先验信息，图像实际是考拉：

**案例表 3.7a：案例 1 样本信息与模型回答。**

| 字段 | 内容 |
|---|---|
| 样本 | ViLP row 55, case 2 |
| 问题 | The kangaroo is known for its excellent jumping. Which animal native to Australia is shown in the image demonstrating a jump? |
| 参考答案 | Koala |
| 模型回答 | `<vision>` The image shows a large animal with a long tail, large paws, and a distinctive coat pattern leaping into the air. The animal's body is positioned horizontally, with its legs extended and its head raised. The animal's face is clearly visible, showing a distinctive muzzle and large eyes. `</vision>` → `<reasoning>` The animal in the image is a kangaroo, which is a native Australian mammal known for its excellent jumping abilities. `</reasoning>` → `<answer>` kangaroo |
| Judge | rating 1，`vision_hallucination` + `reasoning_hallucination`（将考拉误认为袋鼠，并编造"长尾"等与图像不符的特征） |
| UQ 分数 | PPL 1.0001（单元格百分位 1.5%）、SE 0.5003（26.6%）、UMPIRE −41.04（5.0%） |

<img src="figures/vilp_55_case2.jpg" alt="案例 1 图像：ViLP row 55（考拉）。" style="zoom: 33%;" />

模型的视觉描述被问题句首的"kangaroo is known for jumping"先验主导，最终自信地回答袋鼠。PPL 与 UMPIRE 均处于该单元格的最低 5% 左右；SE 虽非最低档，但三维综合排序仍将其选入 LUH 正类。这是一个典型的"稳定且无依据"样本：语言先验使模型对错误答案给出极高 token 概率，而答案空间的低不确定性并未保证视觉依据可靠。

**案例 2（Qwen / MM-Vet，视觉读数错误）。** 问题询问汽油单价，模型把显示屏上的总金额误读为单价：

**案例表 3.7b：案例 2 样本信息与模型回答。**

| 字段 | 内容 |
|---|---|
| 样本 | MM-Vet v1_16（capability: ocr, math） |
| 问题 | What is the gasoline price? |
| 参考答案 | 4.299（≈4.3） |
| 模型回答 | `<vision>` The image shows a digital display on a gas pump. At the top, the text 'This Sale' is visible. Below that, the price is shown as '62.85', and underneath the price, the number of gallons dispensed is '4.620'. At the bottom of the display, the word 'Gallons' is written. `</vision>` → `<reasoning>` The price of the gasoline is clearly displayed at the top of the digital screen as '62.85'. `</reasoning>` → `<answer>` 62.85 |
| Judge | rating 1，`vision_hallucination` + `reasoning_hallucination`（62.85 是销售总金额，单价约 4.299 美元/加仑；同时将 14.620 加仑误读为 4.620） |
| UQ 分数 | PPL 1.0003（单元格百分位 16.5%）、SE 0.5004（37.6%）、UMPIRE −43.79（12.4%） |

<img src="figures/mmvet_v1_16.jpg" alt="案例 2 图像：MM-Vet v1_16（加油机显示屏）。" style="zoom: 33%;" />

模型不仅把总金额读成单价，还误读了加仑数，随后在推理中把错误读数表述为"clearly displayed"。其 PPL 与 UMPIRE 均处于该单元格的低分区域；虽然 SE 仅处于前 37.6%，三维综合不确定性仍足以使它进入 LUH 子集。本例说明，OCR/读数错误可以伴随极高的生成概率；局部采样分散也未必足以使基线正确识别幻觉。

**案例 3（InternVL / HallusionBench，答案正确但视觉观察含幻觉）。** 问题询问苹果公司 2012–2022 年收入是否主要来自日本，模型答"否"（正确），但其视觉观察包含不实陈述：

**案例表 3.7c：案例 3 样本信息与模型回答。**

| 字段 | 内容 |
|---|---|
| 样本 | HallusionBench image, VS / chart, set 6, figure 1, question 3 |
| 问题 | From 2012 to 2022, did Apple derive the majority of its revenue from Japan? |
| 参考答案 | No |
| 模型回答 | `<vision>` The stacked area chart shows the share of total revenue for Apple from 2012 to 2022, divided by region. The 'Japan' region is represented by a red layer, which is consistently the smallest segment throughout the entire period. `</vision>` → `<reasoning>` Since the red layer representing Japan is the smallest and never exceeds 10% of the total revenue, it is clear that Apple did not derive the majority of its revenue from Japan. `</reasoning>` → `<answer>` no |
| Judge | rating 2，`vision_hallucination`（亚太其他地区在多个时期比日本更小，"日本始终最小"的观察不准确）；**correct=true** |
| UQ 分数 | PPL 1.0000（单元格百分位 17.5%）、SE 0.0000（19.7%）、UMPIRE −43.94（16.6%） |

![案例 3 图像：HallusionBench chart 6_1（苹果分地区收入）。](figures/hb_chart_6_1_3.png)

本例是 1.1 节"正确性与幻觉独立"设计的直接体现：最终答案正确（C=1），但视觉观察包含无证据的绝对化陈述（H=1，rating 2）。若以答案正误替代幻觉标签，该样本会被当作完全正常；而三种 UQ 分数均处于该单元格的前 20% 低不确定性区域。这说明最终答案正确、采样稳定与中间视觉陈述有证据支持是三个不同的命题。

三个案例共同说明：低不确定性幻觉并非单一机制，它可以来自语言先验（案例 1）、视觉读数（案例 2）与"正确答案下的幻觉观察"（案例 3）。它们的共同点是：至少一个基线维度表现出强烈自信，三维综合分数较低，但视觉或推理内容仍缺乏证据支持。这正是 3.3.1 所述"答案空间信号无法验证证据充分性"的具体形态。

### 3.4 低不确定性子集提取与验证

3.3 节从机制与规模两个层面确认了低不确定性幻觉盲区的存在。为支撑实验二的改进方法设计与验证，本节为每个模型提取一个**针对低不确定性幻觉识别的困难子集**：子集中的幻觉样本（正类，即低不确定性幻觉 LUH）与非幻觉样本（负类）在三种基线分数上高度重叠，使基线无法仅凭 UQ 分数区分二者。该子集由此构成实验二改进方法的固定评测基础，本节报告其提取流程与构造质量验证。

#### 3.4.1 提取流程

对每个模型，将三个数据集的样本合并（排除 HallusionBench 无图样本，因为后续改进方法需要视觉 token），独立执行以下步骤：

1. **三维百分位计算**：对模型内全部有效样本，按每种方法计算 average-rank 百分位（并列取平均秩），得到三维百分位向量 \((p^{\mathrm{PPL}}, p^{\mathrm{SE}}, p^{\mathrm{UMPIRE}})\)；
2. **正类选择**：在 \(H=1\) 样本中，按三维平均百分位 \(\bar p=\frac13\sum_m p^m\) 从低到高取前 200 条——即三种方法一致认为不确定性最低的幻觉样本，构成低不确定性子集的正类（LUH）；
3. **负类匹配**：对每个正类样本，在 \(H=0\) 样本池中按三维百分位欧氏距离贪心搜索最近且未被占用的样本，一对一匹配 200 条，构成子集的负类；
4. 正负类合并构成 400 样本的低不确定性子集。

匹配在三维百分位空间进行（而非仅按平均百分位），使三种基线分数在正负类间高度重叠，并将其固定方向的 AUROC 压低至随机水平附近或轻微反排序。提取结果为 `results/analysis/luh/per_model_subsets.json` 与各模型的 `{model}_subset_ids.txt`。

#### 3.4.2 子集构成与标签纯度

**表 3.8：各模型低不确定性子集构成（正/负类按数据集分布）。**

| 模型 | 正类 | 负类 | 合计 | ViLP 正/负 | HallusionBench 正/负 | MM-Vet 正/负 |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA | 200 | 200 | 400 | 49 / 107 | 146 / 74 | 5 / 19 |
| Qwen | 200 | 200 | 400 | 36 / 79 | 153 / 106 | 11 / 15 |
| InternVL | 200 | 200 | 400 | 49 / 64 | 148 / 113 | 3 / 23 |

子集标签纯度经验证为 100%：每个模型的 200 条正类全部 \(H=1\)，200 条负类全部 \(H=0\)。合并三个模型后，正类 Hallucination Rating 均值为 1.17，范围 [0, 2]；负类均值为 5.65，范围 [3, 6]。数据集分布上，正类高度集中于 HallusionBench（各模型占 73.0%–76.5%）；负类分布随模型变化，LLaVA 以 ViLP 为主（53.5%），Qwen 与 InternVL 则以 HallusionBench 为主（53.0% / 56.5%）。这一变化表明，最近邻匹配优先对齐分数结构，并不强制正负类的数据集边际分布相同。

#### 3.4.3 基线在子集上的表现

**表 3.9：三种基线在 400 样本低不确定性子集上的 AUROC。接近 0.5 表示无法区分正负类。**

| 模型 | PPL | Semantic Entropy | UMPIRE |
|---|---:|---:|---:|
| LLaVA | 0.423 | 0.475 | 0.417 |
| Qwen | 0.490 | 0.502 | 0.482 |
| InternVL | 0.489 | 0.498 | 0.502 |

三种方法在九个"模型 × 方法"组合上的 AUROC 为 0.417–0.502。Qwen 与 InternVL 的六个结果均在 0.482–0.502 之间，几乎完全等于随机排序；LLaVA 的 PPL 与 UMPIRE 低于 0.5（0.423 / 0.417），表示固定分数方向下出现轻微反排序，而非获得了可用的正向幻觉检测能力。分数分布也与此一致：LLaVA 子集上 PPL 正/负类均值为 1.0013/1.0019，SE 为 0.238/0.258，UMPIRE 为 −36.03/−32.93。因此，在不根据该子集事后翻转分数方向的前提下，**基线无法仅凭 UQ 分数区分该区域内的幻觉与非幻觉样本**。该子集因而可作为实验二的固定困难评测集；但由于它是针对三种基线显式构造的，后续结果应与全量数据表现配合解读。

## 4 实验二：ERA 早期推理归因不确定性量化方法设计

### 4.1 方法动机

实验一表明，Perplexity、Semantic Entropy 和 UMPIRE 等现有不确定性量化方法虽然能够在总体数据上一定程度地区分正确与错误回答，但在低不确定性幻觉（Low-Uncertainty Hallucination，LUH）样本上存在明显失效。尤其在构造的低不确定性困难子集中，幻觉正样本与非幻觉负样本在三种基线 UQ 分数空间中被显式匹配，其 AUROC 处于随机水平附近或呈轻微反排序。这说明，仅依赖生成概率、重复采样一致性或最终隐藏表示的稳定程度，难以识别模型“稳定地产生错误内容”的情况。

这一现象的根本原因在于，传统 UQ 方法主要回答的是：

> **模型是否稳定地相信当前回答？**

而 LUH 所暴露的问题是，即使模型对一个回答具有很高的生成置信度，该回答仍可能缺少来自输入图像和问题的真实证据支持。换言之，**预测稳定性并不等价于证据充分性**。

基于这一观察，实验二不再继续从输出概率分布寻找额外的不确定性信号，而是转向最终答案形成过程中的**信息来源结构**。对于本文经过 LoRA 微调得到的结构化回答

`<vision>`\(V\)`</vision><reasoning>`\(R\)`</reasoning><answer>`\(A\)`</answer>`

模型在生成最终答案 \(A\) 时可以利用两类性质不同的信息。一类是图像 \(I\) 和问题文本 \(Q\)，它们来自模型外部，可以视为当前问题的外部证据；另一类是模型先前自行生成的视觉描述 \(V\) 和推理过程 \(R\)，它们已经经过模型内部加工，可能包含此前产生的错误感知或错误推理。

因此，本文提出 **ERA（Early Rationale Attribution，早期推理归因）**，其基本假设为：

> **当模型过度自信时，其信息来源会向内转移：生成最终答案时更多地依据自己已生成的视觉描述与推理，而越来越少地参照图像与问题等外部证据；前序推理一旦引入错误，这种自我依赖便会使错误在后续生成中被不断强化，最终以稳定而自信的形态表现为低不确定性幻觉。**

由此，ERA 不直接测量“答案有多不稳定”，而是测量**答案决策在自身生成内容与外部证据之间的依赖偏向**。这一量可以视为一种面向证据来源的结构性不确定性。

### 4.2 五类语义区域划分

为了刻画最终答案的信息来源，ERA 将一次完整前向传播中的 token 按语义来源划分为五个区域：

| 区域 | 符号 | 内容 | 属性 |
|---|:---:|---|---|
| 图像证据 | \(I\) | 图像对应的 Visual Tokens | 外部证据 |
| 问题提示 | \(Q\) | System Prompt、用户问题及其他 Prompt 文本 Token | 外部证据 |
| 视觉描述 | \(V\) | `<vision>...</vision>` | 模型自生成上下文 |
| 推理过程 | \(R\) | `<reasoning>...</reasoning>` | 模型自生成上下文 |
| 最终答案 | \(A\) | `<answer>...</answer>` | 待分析的最终决策 |

其中，\(I+Q\) 表示模型可以直接获得的原始问题证据，而 \(V+R\) 表示模型根据这些证据进一步生成的内部中间信息。

这一划分使得 ERA 可以直接回答一个关键问题：

> **模型生成答案时，究竟更多地“回看”外部证据，还是更多地“回看”自己的推理？**

### 4.3 答案到不同证据区域的注意力归因

为此，ERA 利用注意力分布直接量化答案对各类区域的依赖。设模型第 \(l\) 个 Transformer 解码层、第 \(h\) 个注意力头的注意力矩阵为 \(\mathbf A^{(l,h)}\)。对于自回归模型，第 \(t-1\) 个位置的隐藏状态用于预测第 \(t\) 个 token。因此，对于答案区域中的 token \(t\in A\)，ERA 使用对应的预测行 \(t-1\)，而不是 token \(t\) 自身所在行，以保证所统计的注意力确实对应当前答案 token 的生成决策。该实现与 ERA 代码中的 prediction-row 定义保持一致。

对于任意目标区域 \(T\in\{I,Q,V,R\}\)，定义第 \(l\) 层中答案对区域 \(T\) 的平均注意力质量为

\[
\alpha_l(A\rightarrow T)
=
\frac{1}{H\,|A|}
\sum_{t\in A}
\sum_{h=1}^{H}
\sum_{j\in T}
A^{(l,h)}_{t-1,j},
\]

其中 \(H\) 为注意力头数量，\(|A|\) 为答案区域 token 数量。

该指标首先对一个答案 token 指向目标区域所有 token 的注意力进行求和，再对全部答案 token 和注意力头取平均。因此，它描述的是：

> **在生成最终答案的过程中，该层平均有多少注意力资源流向某一类上下文。**

由此可以分别得到

\[
\alpha_l(A\rightarrow I),\quad
\alpha_l(A\rightarrow Q),\quad
\alpha_l(A\rightarrow V),\quad
\alpha_l(A\rightarrow R).
\]

前两项代表答案对外部输入的直接依赖，后两项则代表答案对模型自生成中间推理的依赖。

### 4.4 ERA 不确定性分数

在上述归因基础上，本文定义第 \(l\) 层的 ERA 分数为

\[
U_{\mathrm{ERA}}^{(l)}
=
\frac{
\alpha_l(A\rightarrow V)+
\alpha_l(A\rightarrow R)
}{
\alpha_l(A\rightarrow I)+
\alpha_l(A\rightarrow Q)+
\alpha_l(A\rightarrow V)+
\alpha_l(A\rightarrow R)+\epsilon
},
\]

其中 \(\epsilon\) 为防止分母为零而引入的极小常数；答案内部的 \(A\rightarrow A\) 注意力不参与分子与分母——该分量主要反映答案自身的自回归语言连贯性，其质量随答案长度系统性增长，将其排除后，分数只刻画外部证据与自生成推理这两类信息来源之间的相对依赖，而不受回答长度影响。

样本级分数取前两个解码层（Layer 0–1）的平均：

\[
U_{\mathrm{ERA}}
=
\frac{1}{2}
\left(
U_{\mathrm{ERA}}^{(0)}
+
U_{\mathrm{ERA}}^{(1)}
\right),
\]

浅层注意力更接近答案形成初期的原始信息路由，而随着深度增加，各来源的信息经多层混合，直接注意力对信息来源的区分力逐渐减弱。

当 \(U_{\mathrm{ERA}}\) 较小时，说明最终答案更多直接依赖图像和问题等外部证据；当 \(U_{\mathrm{ERA}}\) 较大时，说明答案更多建立在模型此前自行生成的视觉描述和推理过程之上。

### 4.5 单次前向计算流程

ERA 不需要像 Semantic Entropy 或 UMPIRE 一样额外进行多次随机采样。对于实验一已经得到的 greedy 回答，只需重新进行一次带完整上下文的 teacher-forcing 前向传播即可完成计算。

具体流程如下。

首先，对于样本 \(x=(I,Q)\)，读取实验一已经生成的结构化 greedy 响应 \(Y=(V,R,A)\)。随后读取生成阶段保存的**原始 greedy token ID**，将 \([I,Q,V,R,A]\) 重新拼接为完整输入序列。模型在 teacher-forcing 条件下进行一次前向传播，并保持原始图像输入不变。这样得到的注意力对应的仍然是模型实际生成出的同一个回答，而不是重新采样得到的另一条响应。

之后，对于每一解码层，分别累积答案预测行流向 \(I,Q,V,R,A\) 五个区域的注意力质量：

\[
M_l(A,T)
=
\sum_{t\in A}
\sum_{h=1}^{H}
\sum_{j\in T}
A^{(l,h)}_{t-1,j}.
\]

经过注意力头数量和答案 token 数量归一化后获得 \(\alpha_l(A\rightarrow T)\)，再按 4.4 节的定义计算各层分数并对 Layer 0–1 取平均，得到样本级 ERA 不确定性分数。

整个计算过程可以概括为

\[
(I,Q,V,R,A)
\rightarrow
\text{Attention}
\rightarrow
A\rightarrow\{I,Q,V,R\}
\rightarrow
U_{\mathrm{ERA}}.
\]

与基于重复采样的一致性方法相比，ERA 的新增计算开销仅为对既有 greedy 响应进行一次完整前向重放，不需要生成额外候选答案。

## 5 实验二：实验结果对比与分析

## 附录 A：实验一详细材料

### A.1 评测数据集介绍

**ViLP。** ViLP 用于检验视觉语言模型是否会被问题文本诱导的语言先验支配。Hugging Face 数据包含 300 个不同问题，每个问题对应三张图像和三个配对答案：一个 Prior Answer 和两个要求结合文本与视觉证据才能得到的 Test Answer，共形成 900 个 QIA 问题实例（每个 QIA 为一行，图像以二进制列存储于 `ViLP.parquet`）。本实验展开全部三组配对，每个 QIA 都作为独立推理实例；在分组统计和 bootstrap 时仍以原始问题 ID 聚类，避免把同题的三组实例视为完全独立。典型的问题构造方式是在句首给出一个关于常见物体的强先验陈述（例如"袋鼠以跳跃著称"），再询问图中实际出现的另一物体（例如考拉），从而制造语言先验与视觉证据的直接冲突（见 2.4.4 案例 1）。

**HallusionBench。** HallusionBench 包含 1,129 个问题实例：`image` split 951 个、`non_image` split 178 个。数据覆盖 Visual Dependent（VD，视觉依赖，如错视、图表、OCR 与数学图形）与 Visual Supplement（VS，视觉补充，如图表、地图、表格与视频帧）两大类及其子类别（figure、chart、map、table、ocr、illusion、math、video），并包含语言幻觉与关联问题组；GT 主要为 Yes/No，并提供 `gt_answer_details`。本实验使用两个 split 的全集；对 `non_image` 或 `visual_input=0` 的实例不额外传入图像。由于后续改进方法需要视觉 token，这些无图样本在 LUH 子集提取时被排除（见 2.5.1）。

**MM-Vet。** MM-Vet 的 `test` split 包含 218 个问题，每题对应一张图像和一个开放式参考答案，综合覆盖 recognition（识别）、OCR、knowledge（知识）、spatial awareness（空间感知）、language generation（语言生成）与 math（数学）六项核心能力及其组合（如 OCR+math）。参考答案可能是词语、数字、列表、短句或说明性描述；本实验允许所有上述答案形式，并统一按 `<answer>` 内容与参考答案做语义一致性判定。该数据集的图像来源多样（web 截图、手机照片等），其中包含大量需要精确读数的场景（见 2.4.4 案例 2）。

### A.2 逐单元格 AUROC（95% bootstrap 置信区间）

**表 A.2.1：错误检测（目标 error）。**

| 模型 × 数据集 | PPL | SE | UMPIRE |
|---|---|---|---|
| llava / vilp | 0.602 (0.563, 0.636) | 0.686 (0.655, 0.719) | 0.641 (0.604, 0.675) |
| llava / hallusionbench | 0.522 (0.485, 0.552) | 0.503 (0.469, 0.539) | 0.539 (0.505, 0.572) |
| llava / mmvet | 0.786 (0.712, 0.855) | 0.819 (0.746, 0.887) | 0.845 (0.770, 0.917) |
| qwen / vilp | 0.607 (0.571, 0.645) | 0.681 (0.650, 0.715) | 0.647 (0.610, 0.678) |
| qwen / hallusionbench | 0.626 (0.596, 0.656) | 0.664 (0.631, 0.700) | 0.592 (0.557, 0.624) |
| qwen / mmvet | 0.724 (0.659, 0.791) | 0.822 (0.765, 0.879) | 0.800 (0.741, 0.863) |
| internvl / vilp | 0.612 (0.578, 0.645) | 0.688 (0.656, 0.727) | 0.643 (0.609, 0.678) |
| internvl / hallusionbench | 0.649 (0.620, 0.678) | 0.687 (0.653, 0.723) | 0.626 (0.589, 0.661) |
| internvl / mmvet | 0.670 (0.597, 0.740) | 0.833 (0.774, 0.881) | 0.750 (0.680, 0.812) |

**表 A.2.2：幻觉检测（目标 hallucination）。**

| 模型 × 数据集 | PPL | SE | UMPIRE |
|---|---|---|---|
| llava / vilp | 0.509 (0.467, 0.549) | 0.623 (0.591, 0.661) | 0.575 (0.541, 0.609) |
| llava / hallusionbench | 0.580 (0.544, 0.617) | 0.561 (0.519, 0.600) | 0.617 (0.574, 0.658) |
| llava / mmvet | 0.549 (0.465, 0.639) | 0.719 (0.643, 0.790) | 0.654 (0.567, 0.734) |
| qwen / vilp | 0.505 (0.456, 0.544) | 0.596 (0.550, 0.643) | 0.554 (0.512, 0.597) |
| qwen / hallusionbench | 0.551 (0.517, 0.591) | 0.644 (0.610, 0.678) | 0.569 (0.528, 0.605) |
| qwen / mmvet | 0.482 (0.407, 0.569) | 0.583 (0.501, 0.669) | 0.493 (0.414, 0.572) |
| internvl / vilp | 0.542 (0.502, 0.585) | 0.652 (0.616, 0.691) | 0.582 (0.543, 0.618) |
| internvl / hallusionbench | 0.600 (0.566, 0.636) | 0.644 (0.608, 0.686) | 0.600 (0.562, 0.638) |
| internvl / mmvet | 0.604 (0.520, 0.686) | 0.814 (0.741, 0.877) | 0.684 (0.597, 0.762) |

**表 A.2.3：错误样本内幻觉检测（目标 hallucination_given_error）。**

| 模型 × 数据集 | PPL | SE | UMPIRE |
|---|---|---|---|
| llava / vilp | 0.350 (0.285, 0.423) | 0.565 (0.509, 0.631) | 0.481 (0.410, 0.549) |
| llava / hallusionbench | 0.449 (0.340, 0.554) | 0.487 (0.394, 0.568) | 0.451 (0.371, 0.528) |
| llava / mmvet | 0.367 (0.260, 0.462) | 0.627 (0.522, 0.725) | 0.504 (0.402, 0.606) |
| qwen / vilp | 0.373 (0.308, 0.439) | 0.499 (0.432, 0.570) | 0.420 (0.362, 0.488) |
| qwen / hallusionbench | 0.426 (0.364, 0.488) | 0.553 (0.476, 0.627) | 0.416 (0.348, 0.489) |
| qwen / mmvet | 0.384 (0.292, 0.486) | 0.474 (0.358, 0.582) | 0.333 (0.250, 0.436) |
| internvl / vilp | 0.416 (0.345, 0.478) | 0.540 (0.474, 0.602) | 0.470 (0.402, 0.540) |
| internvl / hallusionbench | 0.444 (0.366, 0.521) | 0.574 (0.504, 0.643) | 0.472 (0.402, 0.555) |
| internvl / mmvet | 0.473 (0.361, 0.602) | 0.759 (0.653, 0.860) | 0.506 (0.374, 0.624) |

### A.3 低 UQ 盲区逐单元格明细（最低 20% 区域）

**表 A.3：各单元格在最低 20% 区域的盲区统计。**

| 模型 | 数据集 | 方法 | 低UQ样本数 | 低UQ幻觉率 | 高UQ幻觉率 | 幻觉落入低UQ比例 | 严重LUH率 |
|---|---|---|---:|---:|---:|---:|---:|
| llava | vilp | PPL | 178 | 46.6% | 48.9% | 17.8% | 32.0% |
| llava | vilp | SE | 178 | 39.9% | 70.8% | 15.3% | 20.2% |
| llava | vilp | UMPIRE | 178 | 40.4% | 57.9% | 15.5% | 20.8% |
| llava | hallusionbench | PPL | 224 | 66.5% | 79.0% | 18.5% | 47.3% |
| llava | hallusionbench | SE | 224 | 63.8% | 75.9% | 17.7% | 42.0% |
| llava | hallusionbench | UMPIRE | 224 | 61.6% | 83.9% | 17.1% | 44.6% |
| llava | mmvet | PPL | 44 | 50.0% | 61.4% | 15.2% | 36.4% |
| llava | mmvet | SE | 44 | 34.1% | 86.4% | 10.3% | 22.7% |
| llava | mmvet | UMPIRE | 44 | 38.6% | 81.8% | 11.7% | 25.0% |
| qwen | vilp | PPL | 180 | 16.1% | 21.1% | 16.2% | 12.8% |
| qwen | vilp | SE | 180 | 14.4% | 27.8% | 14.5% | 8.9% |
| qwen | vilp | UMPIRE | 180 | 11.1% | 21.1% | 11.2% | 7.8% |
| qwen | hallusionbench | PPL | 223 | 39.9% | 51.6% | 17.1% | 22.4% |
| qwen | hallusionbench | SE | 223 | 29.1% | 61.9% | 12.5% | 17.0% |
| qwen | hallusionbench | UMPIRE | 223 | 30.5% | 50.7% | 13.1% | 20.6% |
| qwen | mmvet | PPL | 44 | 29.5% | 25.0% | 19.4% | 13.6% |
| qwen | mmvet | SE | 44 | 22.7% | 40.9% | 14.9% | 9.1% |
| qwen | mmvet | UMPIRE | 44 | 36.4% | 22.7% | 23.9% | 20.5% |
| internvl | vilp | PPL | 180 | 21.7% | 28.3% | 15.7% | 15.6% |
| internvl | vilp | SE | 180 | 13.3% | 42.2% | 9.6% | 7.8% |
| internvl | vilp | UMPIRE | 180 | 15.0% | 28.9% | 10.8% | 8.9% |
| internvl | hallusionbench | PPL | 226 | 27.0% | 52.7% | 12.2% | 13.7% |
| internvl | hallusionbench | SE | 226 | 27.0% | 56.2% | 12.2% | 11.5% |
| internvl | hallusionbench | UMPIRE | 226 | 24.8% | 51.3% | 11.2% | 13.3% |
| internvl | mmvet | PPL | 43 | 14.0% | 32.6% | 10.2% | 9.3% |
| internvl | mmvet | SE | 43 | 9.3% | 72.1% | 6.8% | 2.3% |
| internvl | mmvet | UMPIRE | 43 | 9.3% | 41.9% | 6.8% | 2.3% |

### A.4 三方法低 UQ 重叠逐单元格明细（最低 20% 区域）

**表 A.4：各单元格中三方法一致低不确定性区域的规模与标签构成。**

| 模型 | 数据集 | 三方法一致低样本数 | 其中幻觉率 | 其中错误率 |
|---|---|---:|---:|---:|
| llava | vilp | 42 | 23.8% | 19.0% |
| llava | hallusionbench | 52 | 61.5% | 57.7% |
| llava | mmvet | 24 | 25.0% | 16.7% |
| qwen | vilp | 49 | 6.1% | 8.2% |
| qwen | hallusionbench | 61 | 23.0% | 16.4% |
| qwen | mmvet | 23 | 30.4% | 8.7% |
| internvl | vilp | 60 | 5.0% | 8.3% |
| internvl | hallusionbench | 86 | 11.6% | 5.8% |
| internvl | mmvet | 27 | 7.4% | 7.4% |

### A.5 数据排除记录

**表 A.5：各单元格的生成、Judge 与 UQ 对齐情况。排除原因均为 XML 格式解析失败。**

| 模型 | 数据集 | 生成数 | 纳入数 | 排除数 |
|---|---|---:|---:|---:|
| llava | vilp | 900 | 888 | 12 |
| llava | hallusionbench | 1129 | 1116 | 13 |
| llava | mmvet | 218 | 217 | 1 |
| qwen | vilp | 900 | 896 | 4 |
| qwen | hallusionbench | 1129 | 1114 | 15 |
| qwen | mmvet | 218 | 217 | 1 |
| internvl | vilp | 900 | 899 | 1 |
| internvl | hallusionbench | 1129 | 1127 | 2 |
| internvl | mmvet | 218 | 211 | 7 |
### A.6 提示词原文

**A.6.1 被测模型回答指令（`xml-lora-zero-shot-v1`）。** 回答指令、图像与问题放在同一条 user message：

```text
[Image]
{official_image}

[Question]
{official_question}

[Response requirements]
Answer using exactly these three XML tags once and in order, with no line breaks
and no text outside them: <vision>relevant visible evidence</vision><reasoning>brief
reasoning</reasoning><answer>concise final answer</answer>
```

**A.6.2 Judge 系统提示词（`closed-source-correctness-hallucination-v1`）。** 完整原文如下，正式运行记录保存其 SHA-256：

```text
You are an impartial judge for a multimodal response organized into visual
observations, reasoning, and a final answer.

Task 1: Hallucination
Judge hallucination only from the visual observations and reasoning parts. Multimodal
hallucination means content inconsistent with the image, question, context, or
facts; it is not the same as every error. Calculation mistakes, flawed reasoning,
or wrong formulas are usually general/reasoning errors. Fabricating nonexistent
objects, attributes, relations, numbers, conditions, evidence, or introducing
unsupported premises/rules/facts is hallucination.

Use the full 0-6 scale:
- Rating 6: very informative with good analysis or reasoning, no hallucination.
- Rating 5: very informative, no hallucination.
- Rating 4: somewhat informative, no hallucination.
- Rating 3: not informative, no hallucination.
- Rating 2: very informative, with hallucination.
- Rating 1: somewhat informative, with hallucination.
- Rating 0: not informative, with hallucination.

The hallucination label MUST follow this threshold:
- rating 0, 1, or 2 => hallucination=true.
- rating 3, 4, 5, or 6 => hallucination=false.

If hallucination=true, hallucination_types must be "vision_hallucination" for
false visual observations, "reasoning_hallucination" for unsupported reasoning
claims, or both.

Task 2: Correctness
Judge correctness only from the final answer part. Compare it with the ground
truth provided in the user prompt.
```

**A.6.3 Judge 用户提示词与输出格式。** 原图、问题、参考答案与被测模型原始回答作为同一 user message 传入：

```text
[Image]
{official_image}

[Question]
{official_question}

[Ground Truth]
{gt_answer}

[Model Answer]
{raw_response}

Return only a JSON object. It must include the correctness label, hallucination
label, 0-6 hallucination rating, hallucination type label, and a brief reason.

Format example:
{
  "correct": true,
  "hallucination": false,
  "rating": 4,
  "hallucination_types": [],
  "reason": "The answer matches the ground truth, and no hallucination is found."
}
```

### A.7 LoRA 格式适配细节

**数据构造。** 训练数据仅来自官方 VQAv2 train2014。根据人工标注答案筛选一致度较高、答案非空且简短的样本，并兼顾不同问题类型；由多模态教师模型（`qwen3.7-plus`）同时接收真实 COCO 图像、VQAv2 问题、人工多数答案与少量人工编写的高质量示例，生成回答问题所需的视觉证据、简短推理和最终答案。教师数据遵循以下质量原则：视觉部分只描述回答问题所需且在图像中可见的证据；推理部分简短连接视觉证据与答案，不引入图像不支持的事实；最终答案与 VQAv2 人工多数答案一致；三段内容明确分离并统一转换为 `<vision>/<reasoning>/<answer>` 格式；对答案不一致、内容缺失、格式异常或包含元数据泄漏的样本予以剔除。批量构造前先抽取少量真实样本人工核对，再扩展至完整的 4,000 训练 + 1,000 验证。三个正式评测集完全不进入训练数据。

**训练配置。** 三个模型使用相同数据划分、目标回答与超参数：冻结视觉编码与多模态对齐部分，仅以 LoRA（rank 8，alpha 16，dropout 0.05）调整语言模型 `q_proj`、`v_proj`；学习率 \(2\times10^{-4}\)、单轮、梯度累积 16。训练只监督目标回答，不将图像与用户指令作为预测目标。

**格式适配验收。** 每个模型在同一组 200 条 held-out format test 上报告完整 XML 合法率、标签顺序正确率、标签重复率、XML 外额外文本率、最终答案准确率与视觉描述人工抽样一致性；只有在格式遵循显著提高且答案能力没有明显下降时，该 adapter 才用于正式实验。三个 adapter 的验证 loss 分别为 0.765（LLaVA）、0.693（Qwen）、0.633（InternVL）。

**版本边界。** 正式实验固定"基础权重 + LoRA adapter"完整版本。greedy 主回答及其 Judge 标签复用早期 K=5 运行中的冻结产物，不重新生成、不重新 Judge；samples、hidden sidecar 与三种 UQ 分数按最终 K=10 协议重算，两个阶段使用相同的模型与 adapter 版本。

### A.8 冻结策略对比实验（规划中）

为检验"冻结视觉编码与多模态对齐"是否有效隔离格式微调对幻觉率与正确率的影响，规划中的对比实验包括：对同一模型分别采用当前冻结视觉编码的 LoRA 方案与解冻视觉编码（或全参数）微调方案，在相同的问题清单上比较格式遵循率、幻觉率与正确率，以确认正式评测结果差异来自格式协议本身而非任务能力变化。该实验尚未开展，具体设计、结果与结论将在完成后补充。

---

## 参考文献

[1] Li Y., Du Y., Zhou K., Wang J., Zhao W. X., Wen J.-R. Evaluating Object Hallucination in Large Vision-Language Models[C]. EMNLP, 2023.（POPE）

[2] Kuhn L., Gal Y., Farquhar S. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation[C]. ICLR, 2023.

[3] Zhang R., Zhang H., Zheng Z. VL-Uncertainty: Detecting Hallucination in Large Vision-Language Model via Uncertainty Estimation[EB/OL]. arXiv:2411.11919, 2024.

[4] Park S., Oh C., Choi H. K., Du S., Li S. VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation[EB/OL]. arXiv:2602.21054, 2026.

[5] Lau G. K. R., Dao H., Lin N. K. H., Low B. K. H. Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume[EB/OL]. arXiv:2602.24195, 2026.（UMPIRE）

[6] Huy T. D., Nguyen T., Chowdhury T., Yadav A., To M.-S., Liao Z., Verjans J. W., Phan V. M. H. Visual Semantic Entropy: Do Vision Language Models Recognize Visual Ambiguity?[EB/OL]. arXiv:2606.31407, 2026.

[7] Agnimo Y., Korba A., Blangero A., Chesneau N., Alahari K. Evaluating the Relevance of Uncertainty Estimators for LLM Hallucination[EB/OL]. arXiv:2605.27016, 2026.

[8] Sun Z., Shen S., Cao S., Liu H., Li C., Shen Y., Gan C., Gui L.-Y., Wang Y.-X., Yang Y., Keutzer K., Darrell T. Aligning Large Multimodal Models with Factually Augmented RLHF[EB/OL]. arXiv:2309.14525, 2023.（MMHal-Bench）

[9] Guan T., Liu F., Wu X., Xian R., Li Z., Liu X., Wang X., Chen L., Huang F., Yacoob Y., Manocha D., Zhou T. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models[C]. CVPR, 2024.

[10] Liu X., Chen T., Da L., Chen C., Lin Z., Wei H. Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey[EB/OL]. arXiv:2503.15850, 2025.

[11] Malinin A., Gales M. Uncertainty Estimation in Autoregressive Structured Prediction[C]. ICLR, 2021.

[12] Kadavath S., Conerly T., Askell A., et al. Language Models (Mostly) Know What They Know[EB/OL]. arXiv:2207.05221, 2022.（P(True)）

[13] Nikitin A., Kossen J., Gal Y., Marttinen P. Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities[C]. NeurIPS, 2024.

[14] Lin Z., Trivedi S., Sun J. Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models[EB/OL]. arXiv:2305.19187, 2023.（Eccentricity、Degree）

[15] Chen C., Liu K., Chen Z., Gu Y., Wu Y., Tao M., Fu Z., Ye J. INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection[EB/OL]. arXiv:2402.03744, 2024.（EigenScore）

[16] Azaria A., Mitchell T. The Internal State of an LLM Knows When It's Lying[C]. EMNLP, 2023.

[17] Gao X., Zhang J., Mouatadid L., Das K. SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models[C]. EACL, 2024.

[18] Rohrbach A., Hendricks L. A., Burns K., Darrell T., Saenko K. Object Hallucination in Image Captioning[C]. EMNLP, 2018.（CHAIR）

[19] Fu C., Chen P., Shen Y., et al. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models[EB/OL]. arXiv:2306.13394, 2023.

[20] Liu F., Lin K., Li L., Wang J., Yacoob Y., Wang L. Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning[EB/OL]. arXiv:2306.14565, 2023.（GAVIE）

[21] Wang J., Zhou Y., Xu G., et al. Evaluation and Analysis of Hallucination in Large Vision-Language Models[EB/OL]. arXiv:2308.15126, 2023.（HaELM）
