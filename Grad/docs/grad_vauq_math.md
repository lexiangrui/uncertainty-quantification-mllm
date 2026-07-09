# Grad-VAUQ Mathematical Model

本文档整理当前 `Grad/` 实现中从多模态输入到熵、不确定性分数、梯度选点与消融前向传播的数学过程。记号尽量保持与代码一致，便于后续扩展到其他视觉语言模型。

## 1. 输入与自回归生成

给定图像 \(I\) 和文本问题 \(q\)。视觉语言模型由三部分组成：

- 视觉编码器：\(\phi_v(I)\)
- 视觉到语言空间的投影器：\(P(\cdot)\)
- 语言模型：\(f_\theta(\cdot)\)

图像被编码并投影为视觉 token 序列：

\[
V = P(\phi_v(I)) = [v_1, v_2, \ldots, v_M], \quad v_i \in \mathbb{R}^d
\]

文本问题被 tokenizer 映射为文本 token：

\[
X = [x_1, x_2, \ldots, x_N]
\]

LLaVA 类模型会把 \(V\) 插入到语言模型输入序列中。本文统一记完整 prompt embedding 为：

\[
Z = \operatorname{merge}(X, V)
\]

模型自回归生成答案 token：

\[
Y = [y_1, y_2, \ldots, y_T]
\]

在 `Grad/scripts/run_grad_vauq.py` 中，`backend.generate_with_ids(...)` 先得到文本答案 `answer` 与 token id 序列 `generated_ids`。后续所有不确定性计算都固定这条 \(Y\)，不重新采样答案。

## 2. Teacher-Forced Forward 与响应分布

为了评估已生成答案 \(Y\) 的不确定性，代码把 prompt 与答案拼接为完整输入：

\[
S = [Z, y_1, y_2, \ldots, y_T]
\]

在 teacher-forced 前向传播中，第 \(t\) 个答案 token 的预测分布来自位置 \(t-1\) 的 logits：

\[
\ell_t = f_\theta(S)_{\operatorname{pos}(y_t)-1}
\]

\[
p_t(w \mid I,q,y_{<t}) =
\operatorname{softmax}(\ell_t)_w
\]

其中 \(w\) 遍历词表。代码中的切片是：

```python
response_logits = logits[0, prompt_len - 1 : -1]
```

即用 prompt 最后一个位置预测 \(y_1\)，用 \(y_{t-1}\) 的位置预测 \(y_t\)。

## 3. 原始响应熵

每个答案位置的词表熵定义为：

\[
H_t(I,q,Y) =
- \sum_{w \in \mathcal{V}} p_t(w)\log_2 p_t(w)
\]

整条答案的平均熵为：

\[
H_{\text{org}}(I,q,Y)
=
\frac{1}{T}\sum_{t=1}^{T} H_t(I,q,Y)
\]

代码映射：

- `Grad/grad_vauq/scoring.py::compute_response_entropy`
- `baseline/vauq-repro/src/vauq/metrics.py::OutputScoreInfo.compute_entropy`

该熵衡量模型在固定图像、问题和已生成答案前缀下，对下一个答案 token 的平均不确定性。

## 4. 梯度选取关键视觉 token

### 4.1 固定答案的负对数似然

为了找到哪些视觉 token 对答案概率贡献更直接，当前实现不使用 attention 权重，而是对固定答案的负对数似然求梯度：

\[
\mathcal{L}(I,q,Y)
=
-\frac{1}{T}\sum_{t=1}^{T}
\log p_t(y_t \mid I,q,y_{<t})
\]

代码映射：

```python
loss = response_nll_loss(grad_logits, generated_ids, grad_prompt_len)
```

### 4.2 捕获视觉 token 激活

LLaVA adapter 在 multimodal projector 输出处捕获视觉 token：

\[
V = [v_1,\ldots,v_M]
\]

这里的 \(v_i\) 已经处在语言模型 embedding 空间中，因此适合作为通用的视觉 token attribution 接口。对其他视觉模型，只要 adapter 能暴露“进入语言模型前的视觉 token 序列”，后续公式不需要改变。

### 4.3 Grad x Activation 分数

对每个视觉 token，计算：

\[
g_i = \frac{\partial \mathcal{L}}{\partial v_i}
\]

当前 selector 使用 Grad x Activation：

\[
a_i =
\left\lVert g_i \odot v_i \right\rVert_1
=
\sum_{j=1}^{d}
\left|
\frac{\partial \mathcal{L}}{\partial v_{i,j}}
\cdot v_{i,j}
\right|
\]

然后选取得分最高的 \(K\) 个视觉 token：

\[
\mathcal{S}_K =
\operatorname{TopK}(\{a_i\}_{i=1}^{M}),
\quad
K = \max(1, \lfloor rM \rfloor)
\]

其中 \(r\) 是 `topk_ratio`。CVBench + LLaVA-1.5-7B 当前默认：

\[
r=0.3,\quad \alpha=1.2
\]

代码映射：

- `Grad/grad_vauq/selectors.py::response_nll_loss`
- `Grad/grad_vauq/selectors.py::GradXActSelector`

直观上，attention 是“模型在序列混合时看向哪里”的统计关联，而 \(|\nabla V \odot V|\) 衡量的是“如果这个视觉 token 的激活发生局部扰动，固定答案似然的一阶变化幅度”。它仍不是严格因果效应，但比 attention 更接近对当前答案损失的局部敏感性。

## 5. 视觉 token 消融与 masked forward

得到关键集合 \(\mathcal{S}_K\) 后，再做一次 teacher-forced masked forward。当前实现只保留 attention mask knockout：

\[
\operatorname{mask}_{p_i}=0,\quad p_i=p_{\text{image start}}+i,\quad i\in\mathcal{S}_K
\]

attention mask knockout 不改 \(v_i\) 的数值，而是在语言模型 forward 时关闭被选视觉 token 的 attention 可见性。它和原 VAUQ 的 knockout 干预方式一致，区别在于 token 集合 \(\mathcal{S}_K\) 由梯度而不是 attention 权重选出。

消融后的响应熵为：

\[
H_{\text{mask}}(I,q,Y;\mathcal{S}_K)
=
\frac{1}{T}\sum_{t=1}^{T}
H_t(I,q,Y;\operatorname{mask}_{\mathcal{S}_K})
\]

代码映射：

- `Grad/grad_vauq/backends/llava.py::forward_logits_with_ablation`

## 6. IS 与 Grad-VAUQ 分数

沿用 VAUQ 的外层打分形式。图像敏感度分数定义为：

\[
\operatorname{IS}_{\text{grad}}
=
H_{\text{mask}} - H_{\text{org}}
\]

Grad-VAUQ 分数定义为：

\[
\operatorname{GradVAUQ}
=
H_{\text{org}} - \alpha \cdot \operatorname{IS}_{\text{grad}}
\]

代码映射：

```python
is_score = entropy_masked - entropy_org
vauq = entropy_org - alpha * is_score
```

解释：

- 若 mask 关键视觉 token 后熵升高，\(\operatorname{IS}_{\text{grad}}>0\)，说明答案对这些视觉信息敏感。
- 若 mask 后熵降低，\(\operatorname{IS}_{\text{grad}}<0\)，说明消融让模型更自信。这可能来自 baseline 的分布外效应、原答案本身并不依赖图像，或被替换区域引入了更强的语言先验。
- \(\alpha\) 控制视觉敏感度对最终不确定性分数的修正强度。

## 7. 为什么支持 Flash Attention 2 / SDPA

原 VAUQ 需要 `output_attentions=True` 来取 attention map。很多高性能 attention kernel，尤其 Flash Attention 2，不返回完整 attention matrix，因此原方法必须退回 `attn_implementation="eager"`。

Grad-VAUQ 不需要 attention matrix。它只需要：

1. 捕获视觉 token embedding \(V\)
2. 对固定答案 NLL 求 \(\partial \mathcal{L}/\partial V\)
3. 对选中 token 做一次 masked forward

因此模型前向可以使用：

- `flash_attention_2`：环境安装兼容 `flash-attn` 时使用
- `sdpa`：当前集群环境没有 `flash_attn` 时使用

## 8. 可缓存与需要重算的部分

对同一个样本、同一个模型、同一个生成配置：

### 可以缓存

- 生成答案 \(Y\)：`generated_ids`
- 原始熵 \(H_{\text{org}}\)
- 梯度选出的视觉 token 集合 \(\mathcal{S}_K\)：`grad.selected_indices`
- 正误标签与 judge 结果

### 更换 knockout 策略时只需重算

\[
H_{\text{mask}}(I,q,Y;\mathcal{S}_K)
\]

然后更新：

\[
\operatorname{IS}_{\text{grad}} =
H_{\text{mask}} - H_{\text{org}}
\]

\[
\operatorname{GradVAUQ}
=
H_{\text{org}} - \alpha \cdot \operatorname{IS}_{\text{grad}}
\]

注意：旧结果如果没有保存 `generated_ids`，无法走这个快速路径，需要用更新后的 runner 重新跑一次以生成缓存。

## 9. 通用扩展接口

迁移到其他视觉语言模型时，核心数学过程不变。只需要实现新的 adapter / backend：

1. `capture(model)`：捕获进入语言模型前的视觉 token \(V \in \mathbb{R}^{B\times M\times d}\)，并保持可求梯度。
2. `ablate(model, indices, baseline)`：在同一个视觉 token 层面对指定 indices 做替换。
3. `forward_logits(...)`：给定图像、问题和固定 `generated_ids`，返回 teacher-forced logits 与 `prompt_len`。
4. `generate_with_ids(...)`：生成答案并返回答案 token id。

只要这四个接口成立，`GradXActSelector`、熵计算、IS 与 Grad-VAUQ 分数都可以复用。
