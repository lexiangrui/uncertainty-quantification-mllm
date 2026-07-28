import random

import pytest
import torch

from lora_format.multimodal_sft import (
    build_assistant_labels,
    collate_single,
    load_checkpoint,
    save_checkpoint,
)


def test_end_token_is_last_supervised_token_when_template_adds_newline() -> None:
    labels = build_assistant_labels(
        torch.tensor([10, 11, 20, 21, 99, 13]),
        prompt_length=2,
        end_token_id=99,
        sample_id="sample-1",
    )
    assert labels.tolist() == [-100, -100, 20, 21, 99, -100]


def test_missing_end_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="official end-of-turn token"):
        build_assistant_labels(
            torch.tensor([10, 11, 20, 21]),
            prompt_length=2,
            end_token_id=99,
            sample_id="sample-2",
        )




def test_single_item_collator_preserves_processor_tensor_shapes() -> None:
    row = {
        "input_ids": torch.zeros((1, 8), dtype=torch.long),
        "pixel_values": torch.zeros((256, 1176)),
        "image_grid_thw": torch.tensor([[1, 16, 16]]),
        "labels": torch.zeros((1, 8), dtype=torch.long),
    }
    batch = collate_single([row])
    assert batch["input_ids"].shape == (1, 8)
    assert batch["pixel_values"].shape == (256, 1176)
    assert batch["image_grid_thw"].shape == (1, 3)


def test_checkpoint_round_trip_restores_training_state(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {name: value.detach().clone() for name, value in model.named_parameters()}
    random.seed(123)
    torch.manual_seed(456)
    path = tmp_path / "checkpoint-latest.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=0,
        completed_micro_steps=32,
        update=2,
        samples_seen=32,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    state = load_checkpoint(path, model, optimizer, scheduler)
    assert state["epoch"] == 0
    assert state["completed_micro_steps"] == 32
    assert state["update"] == 2
    assert state["samples_seen"] == 32
    assert scheduler.last_epoch == 1
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, expected[name])
