# UMPIRE

本目录依据 [论文 arXiv:2602.24195v1](https://arxiv.org/abs/2602.24195) 和
[作者官方仓库](https://github.com/daohieu17ctt/UMPIRE)复现 UMPIRE。

每条采样回答生成时取得最后一层、最终回答 token 的 hidden vector；同题采样完成后
立即 L2 归一化并计算带 `1e-8` jitter 的 Gram log determinant。回答联合概率只取
统一三部分输出中的最终答案区间。内部向量与概率均不落盘。

按照官方代码，数据集级自适应 alpha 为未归一化 `logdet` 中位数与
`sum(1-p_i)` 中位数之比，正式分数为 `logdet + alpha * sum(1-p_i)`。程序同时
保存论文中的归一化 `semantic_volume` 和 `incoherence_mean` 便于解释。一次完整
生成结束后，程序仅使用这些标量确定 alpha，再原子回写最终分数，不保存内部张量。

自适应 alpha 必须在完整的无标签“模型 × 数据集”运行上估计。`--limit` 只用于验证
代码路径；特别是单样本时，alpha 会按定义把该样本的两个分量平衡，使最终分数约为
零，不能用于正式比较。
