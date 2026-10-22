"""Persist evolvable behavior separately from task labels and reward machinery."""

from __future__ import annotations

import json
import os
from pathlib import Path

from harnessx.core.harness import HarnessConfig

SYSTEM = "harnessx.processors.context.system_prompt.SystemPromptProcessor"
FIXED = "harnessx.rl.builder._FixedSystemPromptBuilder"
COMMIT = "recipe.verl.processor.early_commit.EarlyCommitProcessor"


def baseline_behavior(text: str | None = None) -> HarnessConfig:
    from recipe.verl.research.prompt import SYSTEM_PROMPT

    return HarnessConfig(
        init_workspace=False,
        step_snapshots=False,
        processors=[
            {"_target_": SYSTEM, "system_builder": {"_target_": FIXED, "text": text or SYSTEM_PROMPT}},
            {"_target_": COMMIT, "max_steps": 12, "token_budget_ratio": 0.95},
        ],
    )


def validate_behavior(config: HarnessConfig) -> None:
    """Bound the initial search space; never allow evolution to replace the reward."""
    if config.init_workspace or config.step_snapshots or config.workspace_template != "default":
        raise ValueError("Behavior must keep init_workspace=false, step_snapshots=false and workspace_template=default")
    if (
        config._rt_procs
        or config.plugins
        or config._rt_sandbox is not None
        or any(
            value is not None
            for value in (
                config.tool_registry,
                config.workspace,
                config.sandbox_provider,
                config.tracer,
                config.sandbox_hint_id,
            )
        )
    ):
        raise ValueError("Behavior may only configure the system prompt and early-commit processor")
    targets = [processor.get("_target_") for processor in config.processors]
    if targets.count(SYSTEM) != 1 or targets.count(COMMIT) != 1 or len(targets) != 2:
        raise ValueError("Behavior requires exactly one system prompt and one early-commit processor")
    for processor in config.processors:
        allowed = {"_target_", "_code_hash"}
        if processor["_target_"] == SYSTEM:
            allowed.add("system_builder")
            builder = processor.get("system_builder", {})
            if (
                set(builder) - {"_target_", "_code_hash", "text"}
                or builder.get("_target_") != FIXED
                or not isinstance(builder.get("text"), str)
                or not builder["text"].strip()
            ):
                raise ValueError("Behavior requires a nonempty fixed system prompt")
        else:
            allowed.update({"max_steps", "token_budget_ratio"})
            steps = processor.get("max_steps", 12)
            ratio = processor.get("token_budget_ratio", 0.95)
            if (
                type(steps) is not int
                or not 1 <= steps <= 64
                or not isinstance(ratio, (int, float))
                or not 0 < ratio <= 1
            ):
                raise ValueError("Invalid early-commit limits")
        if set(processor) - allowed:
            raise ValueError("Unsupported behavior processor arguments")


def behavior_pool(environment=None) -> list[Path]:
    environment = os.environ if environment is None else environment
    raw = environment.get("HARNESSX_HARNESS_POOL")
    paths = json.loads(raw) if raw else []
    if not isinstance(paths, list) or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("HARNESSX_HARNESS_POOL must be a JSON list of config paths")
    return [Path(path).resolve() for path in paths]


def resolve_behavior(metadata=None, *, explicit=None) -> HarnessConfig | None:
    if explicit is None:
        pool = behavior_pool()
        index = (metadata or {}).get("harness_index", 0)
        if pool:
            if not isinstance(index, int) or not 0 <= index < len(pool):
                raise ValueError("Invalid harness_index for the configured behavior pool")
            explicit = pool[index]
        else:
            explicit = os.environ.get("HARNESSX_HARNESS_CONFIG")
    if explicit is None:
        return None
    config = explicit if isinstance(explicit, HarnessConfig) else HarnessConfig.from_yaml_file(Path(explicit))
    validate_behavior(config)
    return config


def route_harnesses(batch, repeats: int, *, validation: bool = False) -> None:
    """Assign variants after interleaved repetition, preserving the original GRPO uid."""
    pool = behavior_pool()
    if not pool:
        return
    import numpy as np

    if validation:
        batch.non_tensor_batch["harness_index"] = np.zeros(len(batch), dtype=object)
        return
    if repeats < len(pool) or repeats % len(pool):
        raise ValueError("rollout.n must be a positive multiple of the harness pool size")
    if len(batch) % repeats:
        raise ValueError("Repeated batch size is not divisible by rollout.n")
    indices = np.tile(np.arange(repeats) % len(pool), len(batch) // repeats)
    batch.non_tensor_batch["harness_index"] = indices.astype(object)
