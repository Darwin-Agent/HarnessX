# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""RL-aware subagent spawn — pre-registered configs, synchronous, captures child token data.

Child HarnessConfigs are built once at import time. Each call only creates
a fresh SGLangProvider (for independent step_captures) and runs the pre-built
config with a new task description.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from harnessx.tools.spawn_subagent import _spawn_ctx
from harnessx.tools.base import Tool
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.core.builder import HarnessBuilder
from harnessx.providers.sglang import SGLangProvider, build_flat_sequence

logger = logging.getLogger(__name__)

_SUB_TIMEOUT = float(os.environ.get("HARNESSX_MATPO_SUB_TIMEOUT", "120"))
_MATPO_SLOT = "_matpo_sub_episodes"


# ─── Pre-registered child configs (built once) ───────────────────────────────

_CHILD_CONFIGS: dict[str, Any] = {}


def register_child_config(name: str, tools: list[Tool], system_prompt: str, question_prompt: str = "", max_steps: int = 6):
    """Pre-register a child agent config. Call at import time."""
    registry = InMemoryToolRegistry()
    for t in tools:
        registry.register(t)

    config = HarnessBuilder().slot(tool_registry=registry).build()

    _CHILD_CONFIGS[name] = {
        "config": config,
        "system_prompt": system_prompt,
        "question_prompt": question_prompt,
        "max_steps": max_steps,
    }


# ─── Core spawn function ─────────────────────────────────────────────────────


async def spawn_subagent_rl(task: str, agent_name: str) -> str:
    """Spawn a pre-registered subagent synchronously, capture its token data.

    Args:
        task: the subtask description
        agent_name: key into _CHILD_CONFIGS (e.g. "searcher", "coder")

    Returns:
        Child's final_output text.
    """
    if agent_name not in _CHILD_CONFIGS:
        return f"Error: unknown subagent '{agent_name}'. Registered: {list(_CHILD_CONFIGS.keys())}"

    child_cfg = _CHILD_CONFIGS[agent_name]
    ctx = _spawn_ctx.get()
    parent_state = ctx.get("state")
    provider_info = ctx.get("provider_info")

    if not provider_info:
        return "Error: MATPO spawn context not available (missing provider_info)."

    from harnessx.core.harness import BaseTask
    from harnessx.core.model_config import ModelConfig

    # Fresh provider per call (independent step_captures)
    child_provider = SGLangProvider(
        url=provider_info["url"],
        tokenizer=provider_info["tokenizer"],
        sampling_params=provider_info["sampling_params"],
        rollout_max_context_len=provider_info.get("rollout_max_context_len", 32768),
        post_fn=provider_info.get("post_fn"),
        tool_call_format=provider_info.get("tool_call_format", "qwen3_coder"),
        tool_result_role=provider_info.get("tool_result_role", "user"),
        strip_think_prefix=provider_info.get("strip_think_prefix", True),
    )

    # Task = question_prompt + user task (system_prompt handled by model's system message)
    question_prompt = child_cfg.get("question_prompt", "")
    description = f"{question_prompt}{task}" if question_prompt else task

    child_task = BaseTask(
        description=description,
        max_steps=child_cfg["max_steps"],
    )

    child_harness = ModelConfig(main=child_provider).agentic(child_cfg["config"])

    try:
        result = await asyncio.wait_for(
            child_harness.run(child_task),
            timeout=_SUB_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("[MATPO] subagent '%s' timed out after %.0fs", agent_name, _SUB_TIMEOUT)
        return "(subagent timed out)"
    except Exception as e:
        logger.error("[MATPO] subagent '%s' failed: %s", agent_name, e)
        return f"(subagent error: {e})"

    # Extract child's token-level data + reward signals
    if child_provider.step_captures:
        try:
            flat = build_flat_sequence(child_provider)

            # Compute sub format/tool_call rewards from child trajectory
            sub_format_reward = 0.0
            sub_tool_call_reward = 0.0
            traj = result.trajectory
            if traj is not None:
                try:
                    from recipe.verl.research.rewards import _check_format_from_traj, _check_tool_call_from_traj
                    sub_format_reward = _check_format_from_traj(traj) * 0.1
                    sub_tool_call_reward = _check_tool_call_from_traj(traj) * 0.2
                except Exception:
                    pass

            sub_episode = {
                "prompt_ids": flat.prompt_ids,
                "response_ids": flat.response_ids,
                "loss_mask": flat.loss_mask,
                "rollout_log_probs": flat.rollout_logprobs,
                "response_length": len(flat.response_ids),
                "format_reward": sub_format_reward,
                "tool_call_reward": sub_tool_call_reward,
            }

            if parent_state is not None:
                if not hasattr(parent_state, _MATPO_SLOT):
                    setattr(parent_state, _MATPO_SLOT, [])
                getattr(parent_state, _MATPO_SLOT).append(sub_episode)

        except Exception as e:
            logger.error("[MATPO] failed to extract subagent tokens: %s", e)

    return result.final_output or "(no output)"
