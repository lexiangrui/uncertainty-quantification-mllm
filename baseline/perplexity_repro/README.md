# Perplexity

实现口径参考 [Hugging Face Perplexity 文档](https://huggingface.co/docs/transformers/perplexity)
以及 Transformers 自回归生成接口。

该目录实现 greedy 最终答案的 token 级 Perplexity baseline。生成器从被测 VLM
瞬时原始 logits 取得最终答案 token 的 log probability，并立即计算

\[
\mathrm{PPL}=\exp\left(-\frac{1}{N}\sum_j \log p(w_j\mid q,w_{<j})\right).
\]

输出只保存 PPL、mean NLL 和答案 token 数；不保存 token IDs、logits 或逐 token
概率。该计算与对同一生成 token 序列执行 teacher forcing 数学等价，同时避免额外前向。
