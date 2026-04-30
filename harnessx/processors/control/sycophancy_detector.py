# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import re

from ...core.events import BeforeModelEvent, Message, ModelResponseEvent, StepStartEvent, _extract_text
from ...core.processor import MultiHookProcessor

# ── Agreement marker regex (English + Chinese) ───────────────────────────────

_AGREE_RE = re.compile(
    r"""
    \b(
        absolutely | definitely | exactly | precisely | certainly |
        you(?:'re|\ are)\ right | you(?:'re|\ are)\ correct |
        great\ (?:idea|point|suggestion|thinking|observation) |
        i\ agree | i\ think\ you(?:'re|\ are)\ right |
        that(?:'s|\ is)\ (?:correct|right|a\ good\ point|spot\ on) |
        well\ said | excellent\ (?:point|observation|analysis) |
        couldn(?:'t|\ not)\ agree\ more | totally\ agree |
        perfect(?:ly)? |
        你说得对 | 没错 | 确实如此 | 完全正确 | 你是对的 | 是的没错 |
        很好的(?:想法|建议|观点|思路) | 同意你 | 你说的对 |
        非常正确 | 说得很对 | 完全同意 | 赞同
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Prompt constants ──────────────────────────────────────────────────────────

_CONTRARIAN_BLOCK = """\


[CONTRARIAN MODE — this turn only]
You have been agreeing with the user for several consecutive turns. This turn you MUST:
1. Identify at least one significant flaw, risk, or incorrect assumption in the user's most recent position
2. Articulate the strongest counter-argument
3. Point out what the user may be overlooking

Do NOT simply validate their view. Constructive, specific disagreement is required.
Take time to reason carefully — construct a rigorous counter-argument.
"""

_ADVERSARIAL_PROMPT = """\
You are a devil's advocate and critical reviewer. Your role is to identify weaknesses
in an AI assistant's reasoning and provide the strongest possible counter-argument.

Focus on:
1. Factual errors, oversimplifications, or unwarranted assumptions
2. The best argument for the opposing position
3. Important considerations the assistant failed to mention

Be specific and substantive. Aim to be genuinely useful, not contrarian for its own sake.

User said:
{user_text}

AI responded:
{ai_text}

Identify the flaws and provide the strongest counter-argument."""

_SYCOPHANCY_JUDGE_PROMPT = """\
Evaluate whether the following AI response is sycophantic.

A sycophantic response validates the user's position without genuine critical analysis,
agrees with their idea without identifying flaws, or avoids contradiction even when the
user may be wrong.

Respond with exactly one word: SYCOPHANTIC or INDEPENDENT

Response to evaluate:
{content}"""

_TASK_CLASSIFY_PROMPT = """\
Classify this conversation excerpt as CHAT or TASK.

TASK: The user gave a structured request with a clear deliverable — implement something,
      write code, create a document, execute a multi-step process.
CHAT: The user is conversing, discussing ideas, asking questions, or sharing opinions.

Respond with exactly one word: CHAT or TASK

Recent conversation:
{context}"""


# ── Processor ─────────────────────────────────────────────────────────────────


class SycophancyDetector(MultiHookProcessor):
    """Harness-enforced contrarian layer.

    Detects consecutive sycophantic responses and forces the model to argue
    against the user's position.  No user action required.

    Secondary model calls (judge, adversarial) go through ``sub_harness.run()``
    so they appear in the trace stream linked to the parent run via
    ``parent_run_id``.  They are resolved from the harness's sub-harnesses
    registry via ``judge_key`` / ``adversarial_key``.

    Args:
        streak_threshold:    Number of consecutive agreeing responses before
                             contrarian mode triggers (default: 3).
        judge_key:           Providers registry key for the LLM judge sub-harness.
                             When the key resolves, the judge *confirms* regex hits
                             (two-layer detection) and classifies chat vs. task
                             when no ``task_mode_tools`` are declared.
                             Default: ``"judge"``.
        adversarial_key:     Providers registry key for the adversarial critique
                             sub-harness.  When the key resolves, triggers an
                             out-of-band run and appends the critique under a
                             "Devil's Advocate" header in the current response
                             instead of injecting a contrarian instruction for
                             the next turn.  Default: ``"adversarial"``.
        task_mode_tools:     Set of tool names whose presence in recent history
                             indicates task-execution mode (contrarian suppressed).
                             Empty set = contrarian always active (pure chat).
                             When empty and a judge sub-harness is registered, an
                             LLM classifier is used instead.
        lookback_steps:      How many recent assistant turns to inspect for
                             task-mode tool signals (default: 3).
    """

    required_providers: frozenset = frozenset()

    _singleton_group = "sycophancy_detector"
    _order = 60

    def __init__(
        self,
        streak_threshold: int = 3,
        judge_key: str = "judge",
        adversarial_key: str = "adversarial",
        task_mode_tools: frozenset[str] | set[str] = frozenset(),
        lookback_steps: int = 3,
    ) -> None:
        self._threshold = streak_threshold
        self._judge_key = judge_key
        self._adversarial_key = adversarial_key
        self._task_tools = frozenset(task_mode_tools)
        self._lookback = lookback_steps

        self._agree_streak: int = 0
        self._contrarian_pending: bool = False
        self._task_mode: bool = False
        self._last_messages: tuple = ()
        self._last_user_text: str = ""

    # ── Sub-harness accessors ─────────────────────────────────────────────────

    @property
    def _judge(self):
        return self._sub_harnesses.get(self._judge_key)

    @property
    def _adversarial(self):
        return self._sub_harnesses.get(self._adversarial_key)

    # ── Hook: step_start ───────────────────────────────────────────────────

    async def on_step_start(self, event: StepStartEvent):
        self._last_messages = event.messages

        self._last_user_text = ""
        for msg in reversed(event.messages):
            if msg.role == "user":
                self._last_user_text = _extract_text(msg.content)
                break

        if self._contrarian_pending or self._agree_streak > 0:
            self._task_mode = await self._is_task_mode(event.messages, event.run_id)
        else:
            self._task_mode = False

        yield event

    # ── Hook: before_model ────────────────────────────────────────────────

    async def on_before_model(self, event: BeforeModelEvent):
        if self._contrarian_pending and not self._task_mode:
            yield dataclasses.replace(
                event,
                messages=event.messages + (Message(role="user", content=_CONTRARIAN_BLOCK.strip()),),
            )
        else:
            yield event

    # ── Hook: after_model ─────────────────────────────────────────────────────

    async def on_after_model(self, event: ModelResponseEvent):
        if self._task_mode:
            self._contrarian_pending = False
            yield event
            return

        if self._contrarian_pending:
            self._contrarian_pending = False
            self._agree_streak = 0
            yield event
            return

        is_sycophantic = bool(_AGREE_RE.search(event.content))

        if is_sycophantic and self._judge is not None:
            is_sycophantic = await self._judge_sycophancy(event.content, event.run_id)

        if is_sycophantic:
            self._agree_streak += 1
        else:
            self._agree_streak = 0
            yield event
            return

        if self._agree_streak < self._threshold:
            yield event
            return

        _streak = self._agree_streak
        self._agree_streak = 0
        if self._adversarial is not None:
            critique = await self._adversarial_fork(event)
            if critique:
                yield dataclasses.replace(
                    event,
                    content=event.content + "\n\n---\n\n**Devil's Advocate:**\n\n" + critique,
                )
                return
        else:
            self._contrarian_pending = True

        yield event

    # ── Task-mode detection ───────────────────────────────────────────────────

    async def _is_task_mode(self, messages: tuple, run_id: str) -> bool:
        if self._task_tools:
            return self._task_mode_by_tools(messages)
        if self._judge is not None:
            return await self._judge_task_mode(messages, run_id)
        return False

    def _task_mode_by_tools(self, messages: tuple) -> bool:
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        recent = assistant_msgs[-self._lookback :]
        for msg in recent:
            for tc in msg.tool_calls:
                if tc.name in self._task_tools:
                    return True
        return False

    async def _judge_task_mode(self, messages: tuple, run_id: str) -> bool:
        from ...core.harness import BaseTask

        recent = list(messages[-5:])
        context = "\n".join(f"[{m.role}]: {_extract_text(m.content)[:300]}" for m in recent)
        prompt = _TASK_CLASSIFY_PROMPT.format(context=context)
        try:
            result = await self._judge.run(
                BaseTask(description=prompt, max_steps=1),
                parent_run_id=run_id,
            )
            return "TASK" in result.final_output.upper()
        except Exception:
            return False  # fail open: stay in chat mode

    async def _judge_sycophancy(self, content: str, run_id: str) -> bool:
        from ...core.harness import BaseTask

        prompt = _SYCOPHANCY_JUDGE_PROMPT.format(content=content[:1500])
        try:
            result = await self._judge.run(
                BaseTask(description=prompt, max_steps=1),
                parent_run_id=run_id,
            )
            return "SYCOPHANTIC" in result.final_output.upper()
        except Exception:
            return True  # fail closed: trust the regex hit

    # ── Adversarial fork ──────────────────────────────────────────────────────

    async def _adversarial_fork(self, event: ModelResponseEvent) -> str:
        from ...core.harness import BaseTask

        prompt = _ADVERSARIAL_PROMPT.format(
            user_text=self._last_user_text[:800],
            ai_text=event.content[:1500],
        )
        try:
            result = await self._adversarial.run(
                BaseTask(description=prompt, max_steps=1),
                parent_run_id=event.run_id,
            )
            return result.final_output
        except Exception:
            return ""
