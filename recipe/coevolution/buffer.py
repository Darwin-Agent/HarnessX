"""Persistent trajectory records and cross-Harness GRPO grouping."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def append_records(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        target.flush()


def load_records(
    paths: list[str | Path],
    *,
    capacity: int = 2048,
    min_group_size: int = 2,
    allow_empty: bool = False,
) -> list[dict]:
    if min_group_size < 2:
        raise ValueError("Cross-Harness GRPO requires min_group_size >= 2")
    if capacity < min_group_size:
        raise ValueError("Buffer capacity must be at least min_group_size")
    buffered = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                task_uid = record.get("task_uid")
                harness_id = record.get("harness_id")
                reward = record.get("reward")
                source_model = record.get("source_model")
                round_number = record.get("round")
                if not isinstance(task_uid, str) or not task_uid:
                    raise ValueError(f"Invalid task_uid in {path}:{line_number}")
                if not isinstance(harness_id, str) or not harness_id:
                    raise ValueError(f"Invalid harness_id in {path}:{line_number}")
                if not isinstance(source_model, str) or not source_model:
                    raise ValueError(f"Invalid source_model in {path}:{line_number}")
                if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
                    raise ValueError(f"Invalid round in {path}:{line_number}")
                if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                    raise ValueError(f"Invalid reward in {path}:{line_number}")
                prompt_ids = record.get("prompt_ids")
                response_ids = record.get("response_ids")
                response_mask = record.get("response_mask")
                old_logprobs = record.get("old_logprobs")
                if (
                    not isinstance(prompt_ids, list)
                    or not prompt_ids
                    or not isinstance(response_ids, list)
                    or not response_ids
                    or not isinstance(response_mask, list)
                    or not isinstance(old_logprobs, list)
                    or len(response_ids) != len(response_mask)
                    or len(response_ids) != len(old_logprobs)
                    or any(
                        isinstance(token, bool) or not isinstance(token, int) or token < 0
                        for token in prompt_ids + response_ids
                    )
                    or any(isinstance(mask, bool) or mask not in (0, 1) for mask in response_mask)
                    or not any(response_mask)
                    or any(
                        isinstance(logprob, bool)
                        or not isinstance(logprob, (int, float))
                        or not math.isfinite(logprob)
                        or logprob > 0
                        for logprob, mask in zip(old_logprobs, response_mask, strict=True)
                        if mask
                    )
                ):
                    raise ValueError(f"Invalid or untrainable token annotations in {path}:{line_number}")
                buffered.append(record)
    latest_by_task_harness = {}
    for record in buffered:
        latest_by_task_harness[(record["task_uid"], record["harness_id"])] = record
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in latest_by_task_harness.values():
        groups[record["task_uid"]].append(record)

    selected_groups = []
    selected_count = 0
    for records in reversed(list(groups.values())):
        if selected_count + len(records) > capacity:
            if not selected_groups:
                raise ValueError("Buffer capacity is smaller than the newest complete cross-Harness task group")
            break
        harness_ids = [record["harness_id"] for record in records]
        if len(harness_ids) != len(set(harness_ids)):
            raise ValueError("A task group contains multiple rollouts for the same Harness configuration")
        selected_groups.append(records)
        selected_count += len(records)

    train_records = []
    for records in reversed(selected_groups):
        if len(records) < min_group_size or len({record["harness_id"] for record in records}) < 2:
            continue
        rewards = [float(record["reward"]) for record in records]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        std = math.sqrt(variance)
        if not std:
            continue
        for record in records:
            record["advantage"] = (record["reward"] - mean) / (std + 1e-8)
            train_records.append(record)
    if not train_records and not allow_empty:
        raise ValueError("The trajectory buffer has no task with samples from multiple Harness configurations")
    if not train_records and allow_empty:
        return []
    if train_records and not any(abs(record["advantage"]) > 1e-12 for record in train_records):
        raise ValueError("Cross-Harness groups have no reward variation for a GRPO update")
    return train_records
