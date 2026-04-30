# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..core.builder import HarnessBuilder
from ..processors.context.system_prompt import SystemPromptProcessor
from ..processors.context.user_wrapper import UserWrapperProcessor
from ..processors.control.compaction import CompactionProcessor
from ..processors.control.tool_failure_guard import ToolFailureGuard


# ---------------------------------------------------------------------------
# make_context — context assembly
# ---------------------------------------------------------------------------


def make_context(
    system_builder=None,
    user_wrapper=None,
    memory=None,
    tool_filter=None,
    memory_policy=None,
) -> HarnessBuilder:
    """Return a context assembly bundle.

    Args:
        system_builder:  Strategy that builds the system prompt.
        user_wrapper:    Strategy wrapping each user message before the model sees it.
        memory:          Memory backend.  When provided, a ``MemoryRetrievalProcessor``
                         is added to inject retrieved memories.
        tool_filter:     Strategy filtering which tools are visible per turn.
        memory_policy:   Gate controlling when memory retrieval fires (default: always).
    """
    b = HarnessBuilder()
    b = b.add(SystemPromptProcessor(system_builder))
    if memory is not None:
        from ..processors.memory.memory_retrieval import MemoryRetrievalProcessor

        b = b.add(MemoryRetrievalProcessor(memory, memory_policy=memory_policy))
    b = b.add(UserWrapperProcessor(user_wrapper))
    if tool_filter is not None:
        from ..processors.tools.tool_filter import ToolFilterProcessor

        b = b.add(ToolFilterProcessor(tool_filter))
    return b


context: HarnessBuilder = make_context()
"""Context assembly bundle with default strategies.

Plug into any ``HarnessBuilder`` via ``| context``.
"""


# ---------------------------------------------------------------------------
# make_window_mgmt — context window management
# ---------------------------------------------------------------------------


def make_window_mgmt(
    token_threshold: int = 140_000,
    message_threshold: int = 100,
    retention_window: int = 10,
    eviction_fraction: float = 0.5,
    summarize_key: str = "summarize",
    max_tool_failures: int = 3,
    skill_tool_names: list[str] | None = None,
) -> HarnessBuilder:
    """Return a context window management bundle.

    Args:
        token_threshold:          Compaction triggers above this token count.
        message_threshold:        Compaction triggers above this message count.
        retention_window:         Messages kept intact during compaction.
        eviction_fraction:        Fraction of compactable messages summarised per call.
        summarize_key:            Sub-harnesses registry key for the summarisation model.
        max_tool_failures:        ToolFailureGuard threshold per turn.
        skill_tool_names:         Tool names counted as skill invocations for budget reduction.
    """
    return (
        HarnessBuilder()
        .add(
            CompactionProcessor(
                token_threshold=token_threshold,
                message_threshold=message_threshold,
                retention_window=retention_window,
                eviction_fraction=eviction_fraction,
                summarize_key=summarize_key,
            )
        )
        .add(ToolFailureGuard(max_failures=max_tool_failures))
    )


window_mgmt: HarnessBuilder = make_window_mgmt()
"""Context window management bundle with default parameters.

Compaction uses the ``"summarize"`` sub-harness key for LLM summarisation.
Without it, compaction still triggers but uses a no-op concatenation stub.

Plug into any ``HarnessBuilder`` via ``| window_mgmt``.
"""
