# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Coroutine, Any

from ...core.events import (
    StepStartEvent,
    BeforeModelEvent,
    BoundaryHint,
    TaskStartEvent,
    TaskEndEvent,
    Message,
    rough_token_count,
    _extract_text,
)
from ...core.processor import MultiHookProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default (no-op) summariser
# ---------------------------------------------------------------------------


_DEFAULT_SUMMARY_MAX_MSGS = 15
_DEFAULT_SUMMARY_MSG_CHARS = 150


async def _default_summarize(messages: list[Message]) -> str:
    """Fallback summariser used when no ``summarize`` sub-harness is registered.

    Takes the most recent ``_DEFAULT_SUMMARY_MAX_MSGS`` of the evicted set
    (closest to the retention window, therefore most relevant) and truncates
    each to ``_DEFAULT_SUMMARY_MSG_CHARS`` chars.  This bounds summary size to
    ~15 × 150 chars regardless of how many messages were evicted, preventing
    unbounded accumulation across repeated compaction cycles.
    """
    tail = messages[-_DEFAULT_SUMMARY_MAX_MSGS:]
    parts = [f"[{m.role}] {_extract_text(m.content)[:_DEFAULT_SUMMARY_MSG_CHARS]}" for m in tail]
    return "[Summary of earlier context]\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# CompactionProcessor
# ---------------------------------------------------------------------------


class CompactionProcessor(MultiHookProcessor):
    """Dual-trigger, dual-window context compaction.

    Triggers compaction when *either*:

    - ``token_count > token_threshold``, or
    - ``len(messages) > message_threshold``

    When triggered:

    1. The most recent ``retention_window`` messages are kept intact.
    2. Of the remaining (older) messages, at most ``eviction_fraction`` are
       handed to the summariser.
    3. The summary is prepended as a single synthetic ``user`` message.

    Summarisation runs through a sub-harness (keyed by ``summarize_key``) so
    the LLM call appears in the trace stream linked to the parent run.
    If the key is absent from the registry, ``summarize_fn`` is used as a
    fallback (defaults to a no-op concatenation stub).

    Args:
        token_threshold:    Token count that triggers compaction (default 140 000).
        message_threshold:  Message count that triggers compaction (default 100).
        retention_window:   Number of recent messages never compacted (default 10).
        eviction_fraction:  Maximum fraction of compactable messages to summarise
                            per call (default 0.5 — half at a time).
        summarize_key:      Sub-harnesses registry key for the summarisation model
                            (default ``"summarize"``).
        summarize_fn:       Fallback async callable ``(list[Message]) → str``.
                            Used when ``summarize_key`` is not in the registry.
                            Defaults to a no-op concatenation stub.
        summarize_prompt_template:
                            Optional prompt template for LLM summarisation.
                            When provided, ``{conversation}`` is replaced with
                            the compactable message transcript.
    """

    required_providers: frozenset = frozenset()

    _singleton_group = "compaction"
    _order = 8  # after context.env (5), before token_budget (10)

    def _bind_model_config(self, model_config: Any) -> None:
        super()._bind_model_config(model_config)
        if self._summarize_key not in self._sub_harnesses and model_config is not None:
            from ...core.harness import Harness as _Harness, HarnessConfig
            from ...tracing.null_tracer import NullTracer as _NullTracer
            from ...core.model_config import ModelConfig as _MC

            provider = model_config.get(self._summarize_key)  # falls back to "main" if key absent
            self._sub_harnesses = {
                **self._sub_harnesses,
                self._summarize_key: _Harness(_MC(main=provider), HarnessConfig(tracer=_NullTracer())),
            }

    def _maybe_rebuild_sub_harness(self) -> None:
        """Rebuild the summarize sub-harness with a nested HarnessJournal when the parent uses one.

        Called from on_task_start once session_id is known. The child journal is nested at
        {parent.base_dir}/{session_id}/subharnesses/ mirroring how spawn_subagent nests
        child agents under {parent.base_dir}/{session_id}/subagents/.
        """
        if not self._session_id:
            return
        rt = getattr(self, "_harness_runtime", None)
        if rt is None:
            return
        from ...tracing.journal import HarnessJournal as _HJ

        tracer = rt.tracer
        if not isinstance(tracer, _HJ):
            return
        import os
        from ...core.harness import Harness as _Harness, HarnessConfig
        from ...core.config_schema import TracerConfig as _TC
        from ...core.model_config import ModelConfig as _MC

        model_config = getattr(self, "_model_config", None)
        if model_config is None:
            return
        provider = model_config.get(self._summarize_key)
        child_base_dir = os.path.join(tracer.base_dir, self._session_id, "subharnesses")
        child_tracer = _TC(
            base_dir=child_base_dir,
            export_jsonl=tracer.export_jsonl,
            silent=tracer.silent,
        )
        self._sub_harnesses = {
            **self._sub_harnesses,
            self._summarize_key: _Harness(_MC(main=provider), HarnessConfig(tracer=child_tracer)),
        }

    def __init__(
        self,
        token_threshold: int = 140_000,
        message_threshold: int = 100,
        retention_window: int = 6,
        eviction_fraction: float = 0.5,
        summarize_key: str = "summarize",
        summarize_fn: Callable[[list[Message]], Coroutine[Any, Any, str]] | None = None,
        summarize_prompt_template: str | None = None,
        preserve_first_message: bool = True,
    ) -> None:
        self.token_threshold = token_threshold
        self.message_threshold = message_threshold
        self.retention_window = retention_window
        self.eviction_fraction = max(0.0, min(1.0, eviction_fraction))
        self._summarize_key = summarize_key
        self._summarize_fn = summarize_fn or _default_summarize
        self._summarize_prompt_template = summarize_prompt_template
        self._preserve_first_message = preserve_first_message
        self._task_anchor: Message | None = None
        self._force_compact_mode: bool = False
        self._compact_stats: tuple[int, int, int, int] | None = None  # (before_msgs, after_msgs, before_tok, after_tok)
        self._session_id: str = ""

    async def on_task_start(self, event: TaskStartEvent):
        if self._preserve_first_message and event.task_description:
            self._task_anchor = Message(role="user", content=event.task_description)
        self._force_compact_mode = False
        self._compact_stats = None
        _prev_session = self._session_id
        self._session_id = event.session_id
        if self._session_id != _prev_session:
            self._maybe_rebuild_sub_harness()
        yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._task_anchor = None
        self._force_compact_mode = False
        self._compact_stats = None
        yield event

    async def _compact(self, messages: tuple[Message, ...], run_id: str = "") -> tuple[Message, ...]:
        msgs = list(messages)

        # Strip the anchor from position 0 if already injected by a previous compaction cycle.
        anchor = self._task_anchor if self._preserve_first_message else None
        if anchor is not None and msgs and msgs[0].content == anchor.content:
            msgs = msgs[1:]

        if len(msgs) <= self.retention_window:
            result = tuple(msgs)
            return (anchor,) + result if anchor is not None else result

        recent = msgs[-self.retention_window :]
        compactable = msgs[: -self.retention_window]

        evict_count = max(1, int(len(compactable) * self.eviction_fraction))
        to_evict = list(compactable[:evict_count])
        to_keep = list(compactable[evict_count:])
        recent = list(recent)

        # Ensure the split boundary respects tool_use/tool_result pairs.
        # If to_keep starts with tool results, their parent assistant was evicted → move them too.
        while to_keep and to_keep[0].role == "tool":
            to_evict.append(to_keep.pop(0))
        # If to_keep is now empty, recent may start with tool results that reference
        # the last evicted assistant message → drop them as well.
        if not to_keep:
            while recent and recent[0].role == "tool":
                to_evict.append(recent.pop(0))

        sub = self._sub_harnesses.get(self._summarize_key)
        if sub is not None:
            summary_text = await self._summarize_via_harness(sub, to_evict, run_id)
        else:
            summary_text = await self._summarize_fn(to_evict)
        summary_msg = Message(role="user", content=summary_text)

        result = tuple([summary_msg] + to_keep + list(recent))
        return (anchor,) + result if anchor is not None else result

    async def _summarize_via_harness(
        self,
        sub_harness: object,
        messages: list[Message],
        run_id: str,
    ) -> str:
        from ...core.harness import BaseTask

        conv_text = "\n".join(f"[{m.role}] {_extract_text(m.content)[:300]}" for m in messages)
        if self._summarize_prompt_template and "{conversation}" in self._summarize_prompt_template:
            prompt = self._summarize_prompt_template.replace("{conversation}", conv_text)
        elif self._summarize_prompt_template:
            prompt = self._summarize_prompt_template + "\n\n" + conv_text
        else:
            prompt = (
                "Summarize the following conversation history in 2-4 sentences, "
                "preserving key decisions, facts, and context:\n\n" + conv_text
            )
        try:
            result = await sub_harness.run(
                BaseTask(description=prompt, max_steps=1),
                parent_run_id=run_id,
            )
            return f"[Earlier conversation summary: {result.final_output.strip()}]"
        except Exception:
            return await self._summarize_fn(messages)

    async def on_step_start(self, event: StepStartEvent):
        # Detect force_compact from task (StepStartEvent carries the task object)
        if getattr(event.task, "force_compact", False):
            self._force_compact_mode = True
            self._task_anchor = None  # management op — don't prepend description to compacted history

        messages_to_compact = event.messages if event.messages else event.raw_messages
        before_msgs = len(messages_to_compact)
        before_tokens = rough_token_count(list(messages_to_compact))

        if (
            not self._force_compact_mode
            and before_tokens <= self.token_threshold
            and before_msgs <= self.message_threshold
        ):
            yield event
            return

        new_messages = await self._compact(messages_to_compact, run_id=event.run_id)
        new_count = rough_token_count(list(new_messages))

        self._compact_stats = (before_msgs, len(new_messages), before_tokens, new_count)

        logger.info(
            "Context compacted: %d → %d messages (~%d → ~%d tokens)",
            before_msgs,
            len(new_messages),
            before_tokens,
            new_count,
        )

        # Always set the hint so RunLoop labels the auto-boundary as "compaction".
        # If the content didn't actually change, hash comparison won't fire a boundary
        # and this hint is never read.
        hint = BoundaryHint(
            reason="compaction",
            before_msgs=before_msgs,
            after_msgs=len(new_messages),
            before_tokens=before_tokens,
            after_tokens=new_count,
        )

        yield dataclasses.replace(
            event,
            messages=new_messages,
            raw_messages=new_messages,
            token_count=new_count,
            boundary_hint=hint,
        )

    async def on_before_model(self, event: BeforeModelEvent):
        if not self._force_compact_mode:
            yield event
            return

        stats = self._compact_stats
        if stats:
            before_msgs, after_msgs, before_tok, after_tok = stats
            if before_msgs != after_msgs:
                msg = f"✅ 已压缩：{before_msgs} → {after_msgs} 条消息（~{before_tok:,} → ~{after_tok:,} tokens）"
            else:
                msg = f"无需压缩（当前 {before_msgs} 条消息，~{before_tok:,} tokens）"
        else:
            msg = "✅ 压缩完成"

        yield dataclasses.replace(event, skip_model=True, synthetic_output=msg)
