# MALP-UQ：Modality-Attributed Latent Perturbation Uncertainty Quantification

> **当前实现说明（2026-07）**：当前实验已将旧版视觉 projector / 文本 embedding
> 分离的 `stage2` 扰动替换为 `fusion` 和 `reasoning` 两阶段。精确 hook 位置、mask、
> 15--20 层设置及当前输出格式以
> [03-融合与推理扰动实现.md](03-融合与推理扰动实现.md) 为准。本文后续模态分离与
> 四阶段内容保留为方法演进记录，不代表当前运行代码。

## 1. 方法定义

### 1.1 输入与输出

给定多模态大模型 $f_\theta$、图像 $I$ 和文本问题 $Q$。视觉编码器（CLIP 等）将图像编码为视觉 token 序列，经 Projector 对齐到 LLM 的 embedding 维度后，得到视觉 embedding：

$$
\mathbf{V} = \operatorname{Proj}(\operatorname{VisEnc}(I)) \in \mathbb{R}^{N_v \times d},
$$

其中 $d$ 为 LLM 的 hidden dimension。文本 embedding 由 LLM 的 token embedding 层获得：

$$
\mathbf{T} = \operatorname{Embed}(Q) \in \mathbb{R}^{N_t \times d}.
$$

模型以自回归方式生成回答 $\mathbf{y}=(y_1,\ldots,y_L)$：

$$
P_\theta(\mathbf{y}\mid\mathbf{V},\mathbf{T})
=
\prod_{l=1}^{L}P_\theta(y_l\mid y_{<l},\mathbf{V},\mathbf{T}).
$$

回答 $\mathbf y$ 由原始输入使用 greedy decoding 生成。后续计算固定该回答，并使用 teacher-forcing。

### 1.2 原始 NLL

定义回答的平均负对数似然：

$$
\mathcal{L}(\mathbf{V},\mathbf{T})
=
-\frac{1}{L}\sum_{l=1}^{L}\log P_\theta(y_l\mid y_{<l},\mathbf{V},\mathbf{T}).
$$

原始输入下的 NLL 为：

$$
\mathcal{L}_0=\mathcal{L}(\mathbf{V},\mathbf{T}).
$$

## 2. 模态分离 Embedding 扰动

### 2.1 扰动框架

对视觉和文本 embedding 分别施加独立扰动。设视觉 token 为 $\mathbf{v}_i \in \mathbb{R}^d\;(i=1,\ldots,N_v)$，文本 token 为 $\mathbf{t}_j \in \mathbb{R}^d\;(j=1,\ldots,N_t)$。给定扰动强度 $\sigma>0$，当前实现使用：

$$
\tilde{\mathbf{v}}_i = \mathbf{v}_i + \sigma \cdot \boldsymbol\delta_i^{(v)},
\qquad
\tilde{\mathbf{t}}_j = \mathbf{t}_j + \sigma\gamma \cdot \boldsymbol\delta_j^{(t)}.
$$

其中 $\gamma$ 暂时固定为 $1$，不做校准。当前保留三种扰动方法：

- `norm_isotropic`：基础随机扰动，扰动后执行范数保持，即保持 $\|\tilde{\mathbf{x}}_i\|_2=\|\mathbf{x}_i\|_2$；
- `directional`：采样高斯标量，沿 token 自身方向扰动，不额外做范数归一化；
- `adversarial`：先计算固定生成序列总对数概率关于 Stage-2 embedding 的梯度，再沿负梯度单位方向扰动；逐 token 扰动长度与 `directional` 保持一致。

三种方法的外层扰动强度统一由 $\sigma/\gamma$ 控制：视觉侧使用 $\sigma$，文本侧使用 $\sigma\gamma$。区别只在于内部的 $\boldsymbol\delta$ 如何生成。

### 2.2 方法 A：norm-preserving isotropic

先采样各向同性高斯方向，并将扰动方向缩放到与原 token 同长度：

$$
\mathbf{r}_i = 
\frac{\boldsymbol\epsilon_i}{\|\boldsymbol\epsilon_i\|_2+\varepsilon}
\cdot
\|\mathbf{x}_i\|_2,
\qquad
\boldsymbol\epsilon_i\sim\mathcal{N}(0,\mathbf{I}_d).
$$

先得到临时扰动结果：

$$
\mathbf{z}_i = \mathbf{x}_i + \sigma\mathbf{r}_i.
$$

然后投影回原 token 范数：

$$
\tilde{\mathbf{x}}_i
=
\frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2+\varepsilon}
\cdot
\|\mathbf{x}_i\|_2.
$$

因此该方法保持的是扰动后的 embedding 范数，而不是只控制扰动向量长度。它适合作为基础随机扰动版本，用于观察在不改变 token 向量长度尺度的前提下，随机方向变化会带来多大影响。

### 2.3 方法 B：directional noise

为每个 token 采样一个高斯标量，并沿 token 自身方向扰动：

$$
\boldsymbol\delta_i = \alpha_i\mathbf{x}_i,
\qquad
\alpha_i \sim \mathcal{N}(0,1).
$$

扰动后为：

$$
\tilde{\mathbf{x}}_i = \mathbf{x}_i + \sigma \alpha_i\mathbf{x}_i.
$$

该方法不再额外做范数归一化，也不强制保持 $\|\tilde{\mathbf{x}}_i\|_2$。它保留“沿原 embedding 语义方向连续放大或缩小”的语义保持特点，而不是只做二值的 $+1/-1$ 选择。

### 2.4 方法 C：adversarial gradient noise

对原始输入先生成固定回答序列 $\mathbf y=(y_1,\ldots,y_L)$，然后在 teacher-forcing 前向中计算整个生成序列的总对数概率：

$$
S(H)=\sum_{l=1}^{L}\log p(y_l\mid H, y_{<l}).
$$

这里 $H$ 表示当前要扰动的 Stage-2 embedding，可以是视觉 projector 输出，也可以是文本 input embedding 输出。然后计算：

$$
\mathbf g=\nabla_H S(H).
$$

扰动方向取梯度单位向量的反方向：

$$
\mathbf d_i=-\frac{\mathbf g_i}{\|\mathbf g_i\|_2+\varepsilon}.
$$

每个 token 采样一个非负高斯标量步长：

$$
\eta_i=|\epsilon_i|,\qquad \epsilon_i\sim\mathcal N(0,1).
$$

最终扰动为：

$$
\boldsymbol\delta_i=\eta_i\|\mathbf h_i\|_2\mathbf d_i,
\qquad
\tilde{\mathbf h}_i=\mathbf h_i+\lambda\boldsymbol\delta_i,
$$

其中视觉侧 $\lambda=\sigma$，文本侧 $\lambda=\sigma\gamma$。由于 $\|\boldsymbol\delta_i\|_2=\eta_i\|\mathbf h_i\|_2$，其逐 token 扰动长度分布与 `directional` 的 $\|\alpha_i\mathbf h_i\|_2=|\alpha_i|\|\mathbf h_i\|_2$ 完全相同；两种方法只在扰动方向上不同。取绝对值保证方向始终降低原生成序列总对数概率。零梯度 token 不施加扰动。

### 2.5 扰动后的 NLL

视觉扰动时保持文本 embedding 不变，反之亦然。定义：

$$
\mathcal{L}_v(\sigma)
:=
\mathbb{E}_{\{\boldsymbol\delta_i^{(v)}\}}
\left[
\mathcal{L}\big(\{\tilde{\mathbf{v}}_i\}_{i=1}^{N_v},\ \mathbf{T}\big)
\right],
$$

$$
\mathcal{L}_t(\sigma)
:=
\mathbb{E}_{\{\boldsymbol\delta_j^{(t)}\}}
\left[
\mathcal{L}\big(\mathbf{V},\ \{\tilde{\mathbf{t}}_j\}_{j=1}^{N_t}\big)
\right].
$$

期望使用 $K$ 次蒙特卡洛采样近似，每次对所有 token 独立采样扰动向量，默认 $K=5$ 或 $K=10$。

### 2.5 扰动影响分数 PIS

$$
\boxed{
\operatorname{PIS}_v
:=
\frac{1}{K}\sum_{k=1}^{K}
\left(\mathcal{L}_v^{(k)}(\sigma)-\mathcal{L}_0\right)
},
\qquad
\boxed{
\operatorname{PIS}_t
:=
\frac{1}{K}\sum_{k=1}^{K}
\left(\mathcal{L}_t^{(k)}(\sigma)-\mathcal{L}_0\right)
}.
$$

PIS 直接统计扰动前后的 NLL 差异。$\operatorname{PIS}>0$ 表示扰动后模型对原回答更不稳定；$\operatorname{PIS}<0$ 表示扰动后模型反而对原回答更确信；绝对值越大，说明扰动影响越明显。

### 2.6 回答 token 分布 KL

$$
\boxed{
\operatorname{KL}_v
:=
\frac{1}{K}\sum_{k=1}^{K}
\frac{1}{L}\sum_{l=1}^{L}
D_{\mathrm{KL}}\left(
p_0(y_l\mid y_{<l})
\;\|\;
p_v^{(k)}(y_l\mid y_{<l})
\right)
},
$$

文本侧 $\operatorname{KL}_t$ 同理，将 $p_v^{(k)}$ 替换为 $p_t^{(k)}$。这里的 $p_0$ 是原始回答 token 位置的 softmax 分布，$p_v^{(k)}$ 是第 $k$ 次视觉扰动后的 softmax 分布。KL 直接度量 logits 分布被扰动改变了多少，始终非负，适合作为“扰动效果强度”的辅助指标。

## 3. 当前校准设置

当前主流程保留 `norm_isotropic`、`directional` 和 `adversarial` 三种扰动方法，均不需要数据集级协方差或标准差校准。

文本侧平衡因子固定为：

$$
\boxed{\gamma=1}.
$$

因此当前运行一个数据集前不再抽样估计扰动参数；三种方法都需要设定扰动强度 $\sigma$、Monte Carlo 次数 $K$ 和随机种子。

## 4. 梯度引导局部扰动

梯度引导版本只对梯度范数最大的 token 子集施加扰动，减少噪声 token 的稀释效应。

### 4.1 Token 梯度

对 $\mathcal{L}_0$ 反向传播，计算各 embedding token 的梯度范数：

$$
g_i = \left\| \nabla_{\mathbf{v}_i}\mathcal{L}_0 \right\|_2 \quad (i=1,\ldots,N_v),
\qquad
h_j = \left\| \nabla_{\mathbf{t}_j}\mathcal{L}_0 \right\|_2 \quad (j=1,\ldots,N_t).
$$

$g_i$ 度量 token $\mathbf{v}_i$ 对回答置信度的局部影响：若 $\mathbf{v}_i$ 是支撑回答的关键视觉证据，扰动该 token 会在 NLL 上产生大梯度。

### 4.2 Top-K 选择

分别选择视觉和文本侧梯度范数最大的 $K_v, K_t$ 个 token：

$$
\mathcal{S}_v = \operatorname{TopK}_v(\{g_i\}_{i=1}^{N_v}), \qquad
m_i = \mathbb{1}[i \in \mathcal{S}_v],
$$

$$
\mathcal{S}_t = \operatorname{TopK}_t(\{h_j\}_{j=1}^{N_t}), \qquad
m'_j = \mathbb{1}[j \in \mathcal{S}_t].
$$

$K_v, K_t$ 可取绝对值（如 $K_v=100$）或比例（如 top 20%）。

### 4.3 局部扰动

仅对被选中的 token 施加扰动，其余保持原值：

$$
\tilde{\mathbf{v}}_i =
\begin{cases}
\mathbf{v}_i + \sigma \cdot \boldsymbol\delta_i^{(v)}, & m_i = 1, \\[4pt]
\mathbf{v}_i, & m_i = 0,
\end{cases}
$$

其中 $\boldsymbol\delta_i^{(v)}$ 可选 `norm_isotropic` 或 `directional` 三种当前实现。文本侧同理。扰动完成后，按第 2 节相同流程计算 $\operatorname{PIS}_v^{\text{grad}}$、$\operatorname{PIS}_t^{\text{grad}}$、$\operatorname{KL}_v^{\text{grad}}$、$\operatorname{KL}_t^{\text{grad}}$。注意：梯度版本的分数仅反映高梯度 token 子集的扰动效果，与基础版本的全量扰动分数在数值和含义上均不同。

扰动协方差可复用全局 $\boldsymbol\Sigma_v$（默认）或在 $\mathcal{S}_v$ 子集上逐样本重新估计（开销更大但更精确，当 $K_v \ll N_v$ 时可选）。

## 5. 阶段分层扰动

### 5.1 扰动注入位置

设 LLM 共有 $M$ 层 Transformer。阶段分层版本在四个位置注入扰动：

| 阶段 | 注入位置 |
|:---:|---|
| $s=1$ | Visual Encoder 输出、Projector 之前 |
| $s=2$ | Projector 输出、LLM 第 1 层之前 |
| $s=3$ | LLM 第 $k$ 层输出之后，$k=\lfloor M/3\rfloor$ |
| $s=4$ | LLM 第 $k'$ 层输出之后，$k'=\lfloor 2M/3\rfloor$ |

> **模型适配**：Stage 1/2 的划分假设 LLaVA 式的两级视觉通路（CLIP → Projector → LLM）。对于 Qwen2-VL 等使用 ViT + MLP projector 一体化的架构，Stage 1 对应 ViT 输出、Stage 2 对应 MLP projector 输出，注入位置按实际模块边界调整。

将模型前向计算分解为 $S=4$ 个子阶段：

$$
f_\theta = f^{(S)} \circ f^{(S-1)} \circ \cdots \circ f^{(1)},
$$

其中 $f^{(1)}$ 为 Visual Encoder + Projector 之前的部分，$f^{(2)}$ 为 Projector 到 LLM 第 1 层之前，$f^{(3)}$ 为 LLM 第 1 至第 $k$ 层，$f^{(4)}$ 为 LLM 第 $k+1$ 至第 $M$ 层。第 $s$ 阶段的中间表示为：

$$
\mathbf{E}_v^{(s)},\ \mathbf{E}_t^{(s)} = f^{(s)}\big(\mathbf{E}_v^{(s-1)},\ \mathbf{E}_t^{(s-1)}\big),
$$

其中 $\mathbf{E}_v^{(0)} = \mathbf{V}$，$\mathbf{E}_t^{(0)} = \mathbf{T}$。

### 5.2 阶段扰动

在阶段 $s$ 的视觉表示处注入扰动后，继续通过剩余层计算 NLL：

$$
\tilde{\mathbf{E}}_v^{(s)} = \mathbf{E}_v^{(s)} + \sigma \mathbf{L}_s \boldsymbol\epsilon,
\qquad
\boldsymbol\epsilon \sim \mathcal{N}(0, \mathbf{I}),
$$

扰动后的 $\tilde{\mathbf{E}}_v^{(s)}$ 与未扰动的 $\mathbf{E}_t^{(s)}$ 一同输入 $f^{(s+1)} \circ \cdots \circ f^{(S)}$，计算固定回答 $\mathbf{y}$ 的 NLL。文本侧扰动同理。阶段版本可复用当前三种扰动方向：范数保持随机扰动、沿向量方向高斯扰动和对抗梯度扰动。

### 5.3 阶段 PIS

阶段 $s$ 的视觉扰动 NLL 定义为：

$$
\mathcal{L}_v^{(s)}(\sigma)
:=
\mathbb{E}_{\boldsymbol\epsilon}
\Big[
\mathcal{L}\big(
\underbrace{f^{(S)} \circ \cdots \circ f^{(s+1)}}_{\text{剩余层}}
(\tilde{\mathbf{E}}_v^{(s)},\ \mathbf{E}_t^{(s)})
\big)
\Big].
$$

阶段视觉 PIS：

$$
\boxed{
\operatorname{PIS}_v^{(s)}
:=
\frac{1}{K}\sum_{k=1}^{K}
\left(\mathcal{L}_v^{(s,k)}(\sigma)-\mathcal{L}_0\right)
},
\qquad s = 1,2,3,4.
$$

文本侧 $\operatorname{PIS}_t^{(s)}$ 以相同方式定义（扰动 $\mathbf{E}_t^{(s)}$，保持 $\mathbf{E}_v^{(s)}$ 不变）。阶段版本也可计算回答 token 位置的 $\operatorname{KL}_v^{(s)}$ 与 $\operatorname{KL}_t^{(s)}$。

### 5.4 阶段权重

阶段权重定义为各阶段视觉 PIS 的归一化占比：

$$
\boxed{w^{(s)} := \frac{\operatorname{PIS}_v^{(s)}}{\sum_{s'=1}^{4}\operatorname{PIS}_v^{(s')} + \varepsilon}},
\qquad \varepsilon = 10^{-8}.
$$

$w^{(s)}$ 反映视觉敏感性在四个阶段之间的分布——值最大的阶段指示模型对视觉信息的依赖主要发生在哪个处理层次。

### 5.5 阶段 KL

逐阶段也计算回答 token 分布变化：

$$
\boxed{
\operatorname{KL}_v^{(s)}
:=
\frac{1}{K}\sum_{k=1}^{K}
\frac{1}{L}\sum_{l=1}^{L}
D_{\mathrm{KL}}\left(p_0(y_l\mid y_{<l})\|p_v^{(s,k)}(y_l\mid y_{<l})\right)
}.
$$

文本侧 $\operatorname{KL}_t^{(s)}$ 同理。

## 6. 超参数

| 参数 | 定义 | 设置方式 |
|---|---|---|
| $\gamma$ | 文本模态平衡因子 | 当前固定为 $1$，暂不校准 |
| $\sigma$ | 扰动强度 | 实现默认 $0.01$；理论扫描网格 $\{0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0\}$ |
| $K$ | 蒙特卡洛采样次数 | 默认 $5$ 或 $10$ |
| $K_v,K_t$ | 梯度版本选择的 token 数或比例 | 默认 top 20%，可取绝对值或比例 |
| $M$ | LLM Transformer 层数 | 由模型决定 |
| $k,k'$ | LLM 阶段切分位置 | $k=\lfloor M/3\rfloor$，$k'=\lfloor 2M/3\rfloor$ |

### 6.1 校准集

当前三种扰动方法不需要校准集。若后续重新加入分布感知方法，再单独定义校准/搜索流程。

## 7. 计算流程

### 7.1 基础版本

1. 使用原始图像和问题生成回答 $\mathbf y$。
2. 使用 teacher-forcing 计算 $\mathcal L_0$。
3. 对视觉 embedding 采样 $K$ 次扰动，计算每轮的 $\mathcal L_v^{(k)}(\sigma)$。
4. 对文本 embedding 采样 $K$ 次扰动，计算每轮的 $\mathcal L_t^{(k)}(\sigma)$。

对 `norm_isotropic` 和 `directional`，基础版本需要 $1+2K$ 次 teacher-forcing 前向。$K=5$ 时共 11 次前向。对 `adversarial`，每个样本需要分别对视觉和文本 Stage-2 embedding 各做一次带反向传播的梯度计算；第一次梯度前向同时复用为原始 NLL/KL 的基准 logits，因此不再额外执行原始 teacher-forcing forward。当前主导出量为 $\operatorname{PIS}_v$、$\operatorname{PIS}_t$、$\operatorname{KL}_v$、$\operatorname{KL}_t$。

### 7.2 梯度版本

1. 执行基础版本的原始前向。
2. 反向计算各 embedding token 的梯度范数。
3. 分别选择视觉与文本 Top-K token。
4. 仅对选中的 token 执行 $K$ 次扰动前向，保存每轮的 NLL。

梯度 Top-K 局部版本需要 $1+2K$ 次前向和 1 次反向；当前 `adversarial` 方法本身也需要梯度，但它用于确定扰动方向，而不是只用于选择 token 子集。

### 7.3 阶段分层版本

1. 执行原始前向并记录四个注入位置的隐藏表示。
2. 分别在每个阶段执行视觉扰动和文本扰动。
3. 每种”阶段—模态”组合采样 $K$ 次，保存每轮的 NLL。

阶段分层版本需要 $1+2K\times4$ 次前向。$K=5$ 时共 41 次前向。实现时可缓存扰动注入点之前的中间表示，并从注入点继续计算后续层。

## 8. 原始数据输出

扰动实验的每条样本保存以下原始数据，后续指标模块从这些数据计算导出量。

### 8.1 每样本字段

| 字段 | 说明 |
|---|---|
| $\mathbf y$ | 原始输入的 greedy 回答文本与 token ids |
| $\mathcal L_0$ | 原始 NLL（平均，1.2 节） |
| $\{\mathcal L_v^{(k)}(\sigma)\}_{k=1}^{K}$ | $K$ 次视觉扰动前向的 NLL |
| $\{\mathcal L_t^{(k)}(\sigma)\}_{k=1}^{K}$ | $K$ 次文本扰动前向的 NLL |
| 原始与扰动回答 logits | 用于计算回答 token 位置的 KL |
| 扰动版本 | `norm_isotropic`、`directional` 或 `adversarial` |
| $\sigma, \gamma$ | 使用的扰动强度；当前 $\gamma=1$ |

梯度版本额外保存：

| 字段 | 说明 |
|---|---|
| $\mathcal S_v, \mathcal S_t$ | 视觉与文本 Top-K token 索引 |
| $\{g_i\}, \{h_j\}$ | 各 token 的梯度范数（可选） |

阶段版本额外保存：

| 字段 | 说明 |
|---|---|
| $\{\mathcal L_v^{(s,k)}(\sigma)\}_{s,k}$ | 阶段 $s$ 视觉扰动第 $k$ 轮的 NLL |
| $\{\mathcal L_t^{(s,k)}(\sigma)\}_{s,k}$ | 阶段 $s$ 文本扰动第 $k$ 轮的 NLL |

### 8.2 导出量（从原始数据计算）

上述原始数据可计算以下导出量：

- $\operatorname{PIS}_v = \frac{1}{K}\sum_k (\mathcal L_v^{(k)}(\sigma) - \mathcal L_0)$
- $\operatorname{PIS}_t = \frac{1}{K}\sum_k (\mathcal L_t^{(k)}(\sigma) - \mathcal L_0)$
- $\operatorname{KL}_v = \frac{1}{K}\sum_k \operatorname{KL}(p_0\|p_v^{(k)})$
- $\operatorname{KL}_t = \frac{1}{K}\sum_k \operatorname{KL}(p_0\|p_t^{(k)})$
- $\operatorname{PIS}_v^{(s)}, \operatorname{PIS}_t^{(s)}, \operatorname{KL}_v^{(s)}, \operatorname{KL}_t^{(s)}, w^{(s)}$（阶段版本）
