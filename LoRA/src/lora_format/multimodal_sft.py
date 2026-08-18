from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .llava_sft import append_eos, read_jsonl
from .prompts import LORA_XML_INSTRUCTION, LORA_XML_PROMPT_SHA256


FAMILIES = ("llava_1_5", "qwen2_5_vl", "internvl3_5")


def _chat_template(processor, messages: list[dict], *, add_generation_prompt: bool) -> str:
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def render_training_text(processor, family: str, question: str, response: str) -> tuple[str, str]:
    user = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{LORA_XML_INSTRUCTION}\n\nQuestion: {question}"},
        ],
    }
    prompt = _chat_template(processor, [user], add_generation_prompt=True)
    full = _chat_template(
        processor,
        [user, {"role": "assistant", "content": [{"type": "text", "text": response}]}],
        add_generation_prompt=False,
    )
    if family == "llava_1_5":
        full = append_eos(full, processor.tokenizer.eos_token)
    return prompt, full


def build_assistant_labels(
    input_ids: torch.Tensor, prompt_length: int, end_token_id: int, sample_id: str
) -> torch.Tensor:
    """Supervise the assistant through its official end-of-turn token only."""
    labels = input_ids.clone()
    labels[:prompt_length] = -100
    end_positions = (input_ids[prompt_length:] == end_token_id).nonzero(as_tuple=False)
    if end_positions.numel() == 0:
        raise ValueError(f"official end-of-turn token is not supervised for {sample_id}")
    end_position = prompt_length + int(end_positions[-1, 0])
    labels[end_position + 1 :] = -100
    if int(labels[end_position]) != end_token_id or torch.any(labels[end_position + 1 :] != -100):
        raise RuntimeError(f"invalid end-of-turn supervision boundary for {sample_id}")
    return labels


class MultimodalXmlDataset(Dataset):
    def __init__(self, rows: list[dict], image_dir: Path, processor, family: str, max_length: int, end_token_id: int):
        self.rows = rows
        self.image_dir = image_dir
        self.processor = processor
        self.family = family
        self.max_length = max_length
        self.end_token_id = end_token_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt_text, full_text = render_training_text(
            self.processor, self.family, row["question"], row["response"]
        )
        with Image.open(self.image_dir / row["image_file"]) as source:
            image = source.convert("RGB")
            prompt = self.processor(
                text=prompt_text, images=image, return_tensors="pt", truncation=True, max_length=self.max_length
            )
            full = self.processor(
                text=full_text, images=image, return_tensors="pt", truncation=True, max_length=self.max_length
            )
        input_ids = full["input_ids"][0]
        prompt_ids = prompt["input_ids"][0]
        if len(prompt_ids) >= len(input_ids) or not torch.equal(input_ids[: len(prompt_ids)], prompt_ids):
            raise ValueError(f"invalid assistant supervision boundary for {row['id']}")
        labels = build_assistant_labels(input_ids, len(prompt_ids), self.end_token_id, row["id"])
        result = {key: value for key, value in full.items() if isinstance(value, torch.Tensor)}
        result["labels"] = labels.unsqueeze(0)
        return result


def collate_single(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(rows) != 1:
        raise ValueError("micro batch size is fixed at one")
    return rows[0]


def checkpoint_path(output: Path) -> Path:
    return output / "checkpoint-latest.pt"


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    *,
    epoch: int,
    completed_micro_steps: int,
    update: int,
    samples_seen: int,
) -> None:
    state = {
        "version": 1,
        "epoch": epoch,
        "completed_micro_steps": completed_micro_steps,
        "update": update,
        "samples_seen": samples_seen,
        "trainable_state": {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer, scheduler) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    missing = set(state["trainable_state"]) - set(parameters)
    if missing:
        raise RuntimeError(f"checkpoint parameters missing from model: {sorted(missing)[:5]}")
    with torch.no_grad():
        for name, value in state["trainable_state"].items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def load_model(config: dict):
    from transformers import (
        AutoProcessor,
        InternVLForConditionalGeneration,
        LlavaForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    )

    family = config["family"]
    model_path = Path(config["base_model"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
    if family == "llava_1_5":
        model_class = LlavaForConditionalGeneration
        kwargs["dtype"] = torch.float16
    elif family == "qwen2_5_vl":
        model_class = Qwen2_5_VLForConditionalGeneration
        kwargs["dtype"] = torch.bfloat16
        kwargs["attn_implementation"] = "sdpa"
    elif family == "internvl3_5":
        model_class = InternVLForConditionalGeneration
        kwargs["dtype"] = torch.bfloat16
    else:
        raise ValueError(f"unknown family: {family}")
    return processor, model_class.from_pretrained(model_path, **kwargs)


def train(
    config: dict,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    resume_from_checkpoint: bool = False,
) -> None:
    if config.get("family") not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if not torch.cuda.is_available():
        raise RuntimeError("multimodal LoRA training requires CUDA")
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from peft import LoraConfig, get_peft_model

    processor, model = load_model(config)
    end_token_id = processor.tokenizer.convert_tokens_to_ids(config["end_token"])
    if end_token_id is None or end_token_id == processor.tokenizer.unk_token_id:
        raise ValueError(f"invalid end token for {config['family']}: {config['end_token']}")
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
            target_modules=config["target_modules"],
        ),
    )
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError(f"unexpected trainable parameters: {[name for name, _ in trainable[:20]]}")
    model.print_trainable_parameters()
    model.to("cuda")
    device = next(parameter for _, parameter in trainable).device

    root = Path(config["dataset_root"])
    rows = read_jsonl(root / "train.jsonl", max_train_samples)
    validation_limit = (
        max_validation_samples
        if max_validation_samples is not None
        else int(config["max_validation_samples"])
    )
    validation_rows = read_jsonl(root / "validation.jsonl", validation_limit)
    train_dataset = MultimodalXmlDataset(
        rows, root.parent / "images", processor, config["family"], int(config["max_length"]), end_token_id
    )
    validation_loader = DataLoader(
        MultimodalXmlDataset(validation_rows, root.parent / "images", processor, config["family"], int(config["max_length"]), end_token_id),
        batch_size=1, shuffle=False, collate_fn=collate_single,
    )
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    accumulation = int(config["gradient_accumulation_steps"])
    epochs = int(config["epochs"])
    total_updates = math.ceil(len(train_dataset) / accumulation) * epochs
    warmup = max(1, int(total_updates * float(config["warmup_ratio"])))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint_path(output)
    checkpoint_interval = int(config.get("checkpoint_interval_updates", 25))
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval_updates must be positive")
    train_path = output / "train_metrics.jsonl"
    validation_path = output / "validation_metrics.jsonl"
    if not resume_from_checkpoint:
        train_path.write_text("")
        validation_path.write_text("")

    def write(path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    optimizer.zero_grad(set_to_none=True)
    start_epoch = completed_micro_steps = update = samples_seen = 0
    if resume_from_checkpoint:
        if not latest_checkpoint.is_file():
            raise FileNotFoundError(latest_checkpoint)
        state = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
        start_epoch = int(state["epoch"])
        completed_micro_steps = int(state["completed_micro_steps"])
        update = int(state["update"])
        samples_seen = int(state["samples_seen"])
        print(json.dumps({"resumed": str(latest_checkpoint), "epoch": start_epoch + 1, "completed_micro_steps": completed_micro_steps, "update": update, "samples_seen": samples_seen}), flush=True)

    stop_requested = False

    def request_checkpoint(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"signal": signum, "action": "checkpoint_at_next_update"}), flush=True)

    signal.signal(signal.SIGUSR1, request_checkpoint)
    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        order_generator = torch.Generator().manual_seed(seed + epoch)
        epoch_order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
        skip_micro_steps = completed_micro_steps if epoch == start_epoch else 0
        if skip_micro_steps > len(epoch_order):
            raise RuntimeError(
                f"checkpoint completed_micro_steps={skip_micro_steps} exceeds epoch size={len(epoch_order)}"
            )
        train_loader = DataLoader(
            Subset(train_dataset, epoch_order[skip_micro_steps:]),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_single,
        )
        model.train()
        loss_sum = 0.0
        micro_count = 0
        for micro_step, batch in enumerate(train_loader, skip_micro_steps + 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss_sum += float(loss.detach())
            micro_count += 1
            samples_seen += 1
            (loss / accumulation).backward()
            if micro_step % accumulation == 0 or micro_step == len(epoch_order):
                torch.nn.utils.clip_grad_norm_((p for _, p in trainable), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                update += 1
                metric = {"epoch": epoch + 1, "update": update, "samples_seen": samples_seen, "micro_batches": micro_count, "train_loss": loss_sum / micro_count, "learning_rate": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started}
                write(train_path, metric); print(json.dumps(metric), flush=True)
                if update % checkpoint_interval == 0 or stop_requested:
                    save_checkpoint(latest_checkpoint, model, optimizer, scheduler, epoch=epoch, completed_micro_steps=micro_step, update=update, samples_seen=samples_seen)
                    print(json.dumps({"checkpoint": str(latest_checkpoint), "epoch": epoch + 1, "completed_micro_steps": micro_step, "update": update}), flush=True)
                if stop_requested:
                    return
                loss_sum = 0.0; micro_count = 0
        completed_micro_steps = 0
        model.eval(); losses = []
        with torch.inference_mode():
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                losses.append(float(model(**batch).loss))
        metric = {"epoch": epoch + 1, "update": update, "validation_samples": len(validation_rows), "validation_loss": sum(losses) / len(losses), "elapsed_seconds": time.monotonic() - started}
        write(validation_path, metric); print(json.dumps(metric), flush=True)
        save_checkpoint(latest_checkpoint, model, optimizer, scheduler, epoch=epoch + 1, completed_micro_steps=0, update=update, samples_seen=samples_seen)

    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    run = {**config, "train_samples": len(rows), "validation_samples": len(validation_rows), "total_updates": total_updates, "trainable_parameters": sum(p.numel() for _, p in trainable), "target_format": "inline_xml", "target_contains_newline": False, "end_token_id": end_token_id, "checkpoint": str(latest_checkpoint), "prompt_sha256": LORA_XML_PROMPT_SHA256}
    (output / "training_config.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
