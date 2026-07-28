#!/usr/bin/env python3
"""Validate MALP perturbations on text embeddings with nearest-neighbor decoding.

An embedding model is not a generative decoder.  Therefore, "natural language
output" here means the nearest entries in an explicitly recorded candidate
lexicon, using cosine similarity in the same embedding space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import PerturbSpec, perturb_tensor  # noqa: E402


DEFAULT_WORDS = ["猫", "狗", "苹果", "汽车", "快乐", "悲伤", "cat", "airplane"]

# A deliberately transparent, small bilingual retrieval vocabulary.  Users can
# supply a larger JSON/JSONL/text vocabulary with --vocab-file.
DEFAULT_VOCAB = [
    "猫", "小猫", "狗", "小狗", "宠物", "动物", "老虎", "狮子", "熊", "鸟", "鱼",
    "苹果", "香蕉", "橙子", "葡萄", "梨", "水果", "蔬菜", "食物", "饮料",
    "汽车", "轿车", "卡车", "公交车", "火车", "自行车", "摩托车", "飞机", "轮船", "交通工具",
    "快乐", "开心", "高兴", "兴奋", "平静", "悲伤", "难过", "愤怒", "害怕", "情绪",
    "红色", "蓝色", "绿色", "黄色", "黑色", "白色", "颜色",
    "北京", "上海", "中国", "城市", "国家", "学校", "大学", "研究", "科学", "技术",
    "cat", "kitten", "dog", "puppy", "pet", "animal", "tiger", "lion", "bear", "bird", "fish",
    "apple", "banana", "orange", "grape", "pear", "fruit", "vegetable", "food", "drink",
    "car", "sedan", "truck", "bus", "train", "bicycle", "motorcycle", "airplane", "ship", "vehicle",
    "happy", "joyful", "excited", "calm", "sad", "angry", "afraid", "emotion",
    "red", "blue", "green", "yellow", "black", "white", "color",
    "Beijing", "Shanghai", "China", "city", "country", "school", "university", "research", "science", "technology",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/lexiangrui/models/bge-m3")
    parser.add_argument("--words", nargs="+", default=DEFAULT_WORDS)
    parser.add_argument("--vocab-file", type=Path)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_vocab(path: Path | None, words: list[str]) -> list[str]:
    if path is None:
        values = DEFAULT_VOCAB
    elif path.suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        values = [item if isinstance(item, str) else item["text"] for item in values]
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(dict.fromkeys([*words, *map(str, values)]))


class BgeEncoder:
    def __init__(self, model_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size], padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            # BGE-M3's dense retrieval representation is the first-token vector.
            chunks.append(F.normalize(hidden[:, 0].float(), p=2, dim=-1))
        return torch.cat(chunks, dim=0)


def nearest(query: torch.Tensor, corpus: torch.Tensor, vocab: list[str], top_k: int) -> list[dict]:
    scores = F.normalize(query.float(), dim=-1) @ corpus.T
    values, indices = scores.topk(min(top_k, len(vocab)))
    return [
        {"text": vocab[index], "cosine_similarity": round(float(score), 6)}
        for score, index in zip(values.cpu(), indices.cpu(), strict=True)
    ]


def main() -> None:
    args = parse_args()
    vocab = load_vocab(args.vocab_file, args.words)
    encoder = BgeEncoder(args.model)
    word_vectors = encoder.encode(args.words, args.batch_size)
    vocab_vectors = encoder.encode(vocab, args.batch_size)

    modes = ("norm_isotropic", "directional")
    results = []
    for word_index, (word, original) in enumerate(zip(args.words, word_vectors, strict=True)):
        row = {
            "input": word,
            "original_norm": float(original.norm()),
            "original_neighbors": nearest(original, vocab_vectors, vocab, args.top_k),
            "original_alternatives": nearest(
                original, vocab_vectors[[item != word for item in vocab]],
                [item for item in vocab if item != word], args.top_k,
            ),
            "perturbations": {},
        }
        target = original.reshape(1, 1, -1)
        for mode_index, mode in enumerate(modes):
            seed = args.seed + word_index * len(modes) + mode_index
            spec = PerturbSpec(
                modality="text", stage="fusion", mode=mode,
                sigma=args.sigma, gamma=1.0, seed=seed,
            )
            changed = perturb_tensor(target, spec).reshape(-1)
            cosine_to_original = F.cosine_similarity(original[None], changed[None]).item()
            row["perturbations"][mode] = {
                "seed": seed,
                "sigma": args.sigma,
                "perturbed_norm": float(changed.float().norm()),
                "delta_l2": float((changed.float() - original).norm()),
                "cosine_to_original": float(cosine_to_original),
                "decoded_neighbors": nearest(changed, vocab_vectors, vocab, args.top_k),
                "decoded_alternatives": nearest(
                    changed, vocab_vectors[[item != word for item in vocab]],
                    [item for item in vocab if item != word], args.top_k,
                ),
            }
        results.append(row)

    payload = {
        "model": args.model,
        "device": str(encoder.device),
        "decoder": "cosine nearest neighbor over recorded candidate vocabulary",
        "candidate_vocabulary_size": len(vocab),
        "configuration": {"sigma": args.sigma, "base_seed": args.seed, "top_k": args.top_k},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
