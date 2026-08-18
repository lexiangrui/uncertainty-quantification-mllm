from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .prompts import LORA_XML_INSTRUCTION, LORA_XML_PROMPT_SHA256


def append_eos(text: str, eos_token: str | None) -> str:
    if not eos_token:
        raise ValueError("the tokenizer must define an EOS token for SFT")
    return text if text.endswith(eos_token) else text + eos_token


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows if limit is None else rows[:limit]
    if not rows:
        raise ValueError(f"empty split: {path}")
    return rows


class LlavaXmlDataset(Dataset):
    def __init__(self, rows: list[dict], image_dir: Path, processor, max_length: int):
        self.rows = rows
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image_path = self.image_dir / row["image_file"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        user_message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{LORA_XML_INSTRUCTION}\n\nQuestion: {row['question']}"},
            ],
        }
        prompt_text = self.processor.apply_chat_template(
            [user_message], add_generation_prompt=True, tokenize=False
        )
        full_text = self.processor.apply_chat_template(
            [
                user_message,
                {"role": "assistant", "content": [{"type": "text", "text": row["response"]}]},
            ],
            add_generation_prompt=False,
            tokenize=False,
        )
        full_text = append_eos(full_text, self.processor.tokenizer.eos_token)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            prompt = self.processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            full = self.processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        input_ids = full["input_ids"][0]
        prompt_ids = prompt["input_ids"][0]
        if len(prompt_ids) >= len(input_ids):
            raise ValueError(f"assistant target was truncated for {row['id']}")
        if not torch.equal(input_ids[: len(prompt_ids)], prompt_ids):
            raise ValueError(f"prompt is not a token prefix for {row['id']}")
        eos_token_id = self.processor.tokenizer.eos_token_id
        if eos_token_id is None or int(input_ids[-1]) != int(eos_token_id):
            raise ValueError(f"assistant target does not end with EOS for {row['id']}")
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100
        if int(labels[-1]) != int(eos_token_id):
            raise ValueError(f"EOS is not supervised for {row['id']}")
        result = {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"][0],
            "pixel_values": full["pixel_values"][0],
            "labels": labels,
        }
        if "image_sizes" in full:
            result["image_sizes"] = full["image_sizes"][0]
        return result


def collate_single(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(rows) != 1:
        raise ValueError("micro batch size is fixed at one")
    return {key: value.unsqueeze(0) for key, value in rows[0].items()}


def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss))
    model.train()
    return sum(losses) / len(losses)


def train(config: dict, max_train_samples: int | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("LLaVA LoRA training requires CUDA")
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_path = Path(config["base_model"])
    dataset_root = Path(config["dataset_root"])
    output_dir = Path(config["output_dir"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r".*language_model.*\.(q_proj|v_proj)$",
        ),
    )
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names[:20]}")
    model.print_trainable_parameters()
    device = torch.device("cuda")
    model.to(device)

    train_rows = read_jsonl(dataset_root / "train.jsonl", max_train_samples)
    validation_rows = read_jsonl(
        dataset_root / "validation.jsonl", int(config["max_validation_samples"])
    )
    image_dir = dataset_root.parent / "images"
    max_length = int(config["max_length"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        LlavaXmlDataset(train_rows, image_dir, processor, max_length),
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=collate_single,
    )
    validation_loader = DataLoader(
        LlavaXmlDataset(validation_rows, image_dir, processor, max_length),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    accumulation = int(config["gradient_accumulation_steps"])
    epochs = int(config["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * epochs
    warmup = max(1, int(total_updates * float(config["warmup_ratio"])))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = output_dir / "train_metrics.jsonl"
    validation_metrics_path = output_dir / "validation_metrics.jsonl"
    train_metrics_path.write_text("", encoding="utf-8")
    validation_metrics_path.write_text("", encoding="utf-8")

    def append_metric(path: Path, metric: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
            handle.flush()

    optimizer.zero_grad(set_to_none=True)
    update = 0
    samples_seen = 0
    started_at = time.monotonic()
    for epoch in range(epochs):
        model.train()
        accumulated_loss = 0.0
        accumulated_micro_batches = 0
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            accumulated_loss += float(loss.detach())
            accumulated_micro_batches += 1
            samples_seen += int(batch["input_ids"].shape[0])
            (loss / accumulation).backward()
            if micro_step % accumulation == 0 or micro_step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                metric = {
                    "epoch": epoch + 1,
                    "update": update,
                    "samples_seen": samples_seen,
                    "micro_batches": accumulated_micro_batches,
                    "train_loss": accumulated_loss / accumulated_micro_batches,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - started_at,
                }
                append_metric(train_metrics_path, metric)
                print(json.dumps(metric), flush=True)
                accumulated_loss = 0.0
                accumulated_micro_batches = 0
        validation_metric = {
            "epoch": epoch + 1,
            "update": update,
            "validation_samples": len(validation_rows),
            "validation_loss": evaluate(model, validation_loader, device),
            "elapsed_seconds": time.monotonic() - started_at,
        }
        append_metric(validation_metrics_path, validation_metric)
        print(json.dumps(validation_metric), flush=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    run = {
        **config,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "total_updates": total_updates,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "assistant_target_ends_with_eos": True,
        "eos_token": processor.tokenizer.eos_token,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "prompt_sha256": LORA_XML_PROMPT_SHA256,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
