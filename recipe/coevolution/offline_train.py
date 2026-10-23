"""Distributed off-policy GRPO updates from saved trajectory records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from recipe.coevolution.buffer import load_records


class TrajectoryDataset(Dataset):
    def __init__(self, records):
        self.records = []
        for record in records:
            prompt_ids = record["prompt_ids"]
            response_ids = record["response_ids"]
            response_mask = record["response_mask"]
            old_logprobs = record["old_logprobs"]
            if not prompt_ids or not response_ids or not (len(response_ids) == len(response_mask) == len(old_logprobs)):
                raise ValueError("Offline trajectory token fields must be nonempty and aligned")
            if not any(response_mask):
                continue
            if any(mask not in (0, 1) for mask in response_mask):
                raise ValueError("Response masks must contain only 0 and 1")
            if any(
                not torch.isfinite(torch.tensor(value)) or value > 0
                for value, mask in zip(old_logprobs, response_mask, strict=True)
                if mask
            ):
                raise ValueError("Trainable response tokens require finite behavior log probabilities")
            self.records.append(
                {
                    "input_ids": prompt_ids + response_ids,
                    "prompt_length": len(prompt_ids),
                    "response_mask": response_mask,
                    "old_logprobs": old_logprobs,
                    "advantage": float(record["advantage"]),
                }
            )
        if not self.records:
            raise ValueError("No trainable trajectory tokens remain in the buffer")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class OfflineGRPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        response_mask = inputs["response_mask"]
        old_logprobs = inputs["old_logprobs"]
        advantages = inputs["advantages"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        token_logits = outputs.logits[:, :-1, :].float()
        target_ids = input_ids[:, 1:]
        current_logprobs = token_logits.log_softmax(dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        response_mask = response_mask[:, 1:].to(current_logprobs.dtype)
        old_logprobs = old_logprobs[:, 1:]
        ratio = (current_logprobs - old_logprobs).clamp(-20, 20).exp()
        advantage = advantages.unsqueeze(1)
        unclipped = ratio * advantage
        clipped = ratio.clamp(0.8, 1.2) * advantage
        loss = -torch.minimum(unclipped, clipped)
        loss = (loss * response_mask).sum() / response_mask.sum().clamp_min(1)
        return (loss, outputs) if return_outputs else loss


def collate_records(rows, pad_token_id):
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention_masks = []
    response_masks = []
    old_logprobs = []
    advantages = []
    for row in rows:
        length = len(row["input_ids"])
        padding = max_length - length
        input_ids.append(row["input_ids"] + [pad_token_id] * padding)
        attention_masks.append([1] * length + [0] * padding)
        token_mask = [0] * row["prompt_length"] + row["response_mask"]
        token_logprobs = [0.0] * row["prompt_length"] + row["old_logprobs"]
        response_masks.append(token_mask + [0] * padding)
        old_logprobs.append(token_logprobs + [0.0] * padding)
        advantages.append(row["advantage"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "response_mask": torch.tensor(response_masks, dtype=torch.float32),
        "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--buffer", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--buffer-capacity", type=int, default=2048)
    arguments = parser.parse_args()

    records = load_records(arguments.buffer, capacity=arguments.buffer_capacity)
    dataset = TrajectoryDataset(records)
    model = AutoModelForCausalLM.from_pretrained(arguments.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    model.config.use_cache = False
    pad_token_id = model.config.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.config.eos_token_id
    if pad_token_id is None:
        raise ValueError("The model must define a pad_token_id or eos_token_id")
    fsdp = None
    fsdp_config = None
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) > 1:
        layer_types = getattr(model, "_no_split_modules", None)
        if not layer_types:
            raise ValueError("The model architecture must expose _no_split_modules for distributed FSDP wrapping")
        fsdp = "full_shard auto_wrap"
        fsdp_config = {"transformer_layer_cls_to_wrap": layer_types}
    training_args = TrainingArguments(
        output_dir=arguments.output,
        max_steps=arguments.steps,
        per_device_train_batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        fsdp=fsdp or "",
        fsdp_config=fsdp_config,
    )
    trainer = OfflineGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda rows: collate_records(rows, pad_token_id),
    )
    trainer.train()
    trainer.save_model(arguments.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(arguments.output)
    if trainer.is_world_process_zero():
        Path(arguments.output, "coevolution_buffer_manifest.json").write_text(
            json.dumps({"trajectory_count": len(dataset), "task_grouped": True}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
