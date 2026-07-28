# Semantic Entropy

该目录只实现项目统一生成流水线中的 Semantic Entropy baseline，不再独立读取数据、
加载被测多模态模型、生成回答、判断正确性或维护独立运行入口。

## 执行位置

公共入口 `scripts/generation/generate_responses.py` 每题完成 greedy 和 10 条随机回答后，
在 VLM 尚未释放时调用本方法。生成器从瞬时原始 logits 得到最终答案的 mean log
probability，只通过内存传递给 Semantic Entropy。

JSONL 不保存 token ID、token probability、logits、hidden states 或其他内部状态。

## 方法

1. 接收每题固定 10 条采样回答的最终答案与瞬时 mean log probability；
2. 使用本地 DeBERTa NLI 对回答进行严格双向蕴含聚类；
3. 对每个语义簇聚合采样回答的概率质量并在已观察回答上归一化；
4. 计算语义簇分布的 Shannon entropy。

一条采样回答无法分段或缺少答案概率时，该题记录为无效，不补写回答或退化为
频率熵。

## 运行

计算节点离线运行：

```bash
python scripts/generation/generate_responses.py \
  --dataset vilp \
  --dataset-source /server/datasets/ViLP.parquet \
  --model-family llava_1_5 \
  --model-path /server/models/llava-1.5-7b-hf \
  --output /server/results/llava_vilp.jsonl \
  --entailment-model-path /server/models/deberta-v2-xlarge-mnli
```

输出在每条生成记录的 `uq.semantic_entropy` 中保存最终 `score`、每条回答的
semantic ID、每个簇的成员和归一化概率。Judge 后续读取相同回答 JSONL 独立运行。
