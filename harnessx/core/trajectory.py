# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Protocol, runtime_checkable

from ..core.events import (
    Event,
    ModelResponseEvent,
    StepStartEvent,
    ToolResultEvent,
    message_to_dict,
)

if TYPE_CHECKING:
    from ..core.state import State

logger = logging.getLogger(__name__)


# ─── State Snapshot Primitives ────────────────────────────────────────────────


@dataclass
class StateSlotSnapshot:
    """Snapshot of a single state slot at a step."""

    key: str
    slot_type: str
    content: Any
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SlotOperation:
    """A single slot change operation within a step."""

    operation: str  # "create" | "update" | "delete"
    key: str
    before: StateSlotSnapshot | None  # None for "create"
    after: StateSlotSnapshot | None  # None for "delete"


@dataclass(frozen=True)
class StateDelta:
    """
    Δz_t: explicit set of slot changes from z_{t-1} to z_t.
    Design principle: state updates are structural transforms on the slot set.
    """

    step_id: int
    operations: tuple[SlotOperation, ...]

    def created(self) -> list[SlotOperation]:
        return [op for op in self.operations if op.operation == "create"]

    def updated(self) -> list[SlotOperation]:
        return [op for op in self.operations if op.operation == "update"]

    def deleted(self) -> list[SlotOperation]:
        return [op for op in self.operations if op.operation == "delete"]


@dataclass(frozen=True)
class FullStateSnapshot:
    """
    z_t: complete state snapshot at step t.
    Immutable (frozen) so historical snapshots cannot be polluted by later state changes.

    messages here is the raw ``state.raw_messages`` (user/assistant/tool history),
    WITHOUT the CE-assembled system prompt. For full context including
    system prompt, use TrajectoryStep.ce_output.messages.
    """

    step_id: int
    messages: tuple  # tuple[Message, ...] — raw conversation history at this step
    slots: dict[str, StateSlotSnapshot]
    cumulative_tokens: int
    cumulative_cost_usd: float

    @classmethod
    def from_state(cls, state: "State", step_id: int) -> "FullStateSnapshot":
        return cls(
            step_id=step_id,
            messages=tuple(state.raw_messages),
            slots={
                k: StateSlotSnapshot(
                    key=k,
                    slot_type=v.slot_type,
                    content=v.content,
                    metadata=dict(v.metadata),
                )
                for k, v in state.slots.items()
            },
            cumulative_tokens=state.cumulative_tokens,
            cumulative_cost_usd=state.cumulative_cost_usd,
        )

    def diff(self, new_state: "State") -> StateDelta:
        """Compute the slot-level delta from this snapshot to new_state."""
        operations: list[SlotOperation] = []

        old_keys = set(self.slots.keys())
        new_keys = set(new_state.slots.keys())

        # Created slots
        for key in new_keys - old_keys:
            v = new_state.slots[key]
            operations.append(
                SlotOperation(
                    operation="create",
                    key=key,
                    before=None,
                    after=StateSlotSnapshot(
                        key=key,
                        slot_type=v.slot_type,
                        content=v.content,
                        metadata=dict(v.metadata),
                    ),
                )
            )

        # Deleted slots
        for key in old_keys - new_keys:
            operations.append(
                SlotOperation(
                    operation="delete",
                    key=key,
                    before=self.slots[key],
                    after=None,
                )
            )

        # Updated slots
        for key in old_keys & new_keys:
            old = self.slots[key]
            v = new_state.slots[key]
            if old.content != v.content or old.slot_type != v.slot_type:
                operations.append(
                    SlotOperation(
                        operation="update",
                        key=key,
                        before=old,
                        after=StateSlotSnapshot(
                            key=key,
                            slot_type=v.slot_type,
                            content=v.content,
                            metadata=dict(v.metadata),
                        ),
                    )
                )

        return StateDelta(step_id=self.step_id, operations=tuple(operations))


# ─── TokenAnnotation ───────────────────────────────────────────────────


@dataclass
class TokenAnnotation:
    """
    Token-level annotation for a single TrajectoryStep, for RL/GRPO training.

    prompt_ids: CE-assembled full context token sequence for this step (complete
    snapshot, not incremental). Populated by backfill_token_annotations().

    response_ids: model-generated tokens + tool result tokens for this step.
    Does NOT include prompt tokens. Concatenating all steps' response_ids
    gives the full rollout response sequence.

    response_mask: 1 = model-generated (assistant content + tool_call JSON),
    0 = tool result / injected content. Same length as response_ids.

    response_logprobs: rollout log-probabilities for each response token.
    Real logprobs for model-generated tokens (loss_mask=1), 0.0 for tool
    result tokens (loss_mask=0). Same length as response_ids.
    Used by GRPO importance sampling ratio computation.

    token_reward: step-level reward from ProcessRewardModel (default 0.0).
    Filled by reward_func() after PRM scoring.

    Invariant: len(response_ids) == len(response_mask) == len(response_logprobs)
    """

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] = field(default_factory=list)
    token_reward: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.response_ids)
        if len(self.response_mask) != n:
            raise ValueError(f"response_mask length {len(self.response_mask)} != response_ids length {n}")
        if self.response_logprobs and len(self.response_logprobs) != n:
            raise ValueError(f"response_logprobs length {len(self.response_logprobs)} != response_ids length {n}")


# ─── RLFormat Protocol ────────────────────────────────────────────────────────


@runtime_checkable
class RLFormat(Protocol):
    """
    Protocol for converting a StatefulTrajectory into RL framework training records.

    Implementations are framework-specific and live outside harnessx core:
        recipe/slime/formats/slime_format.py  → SlimeRLFormat   (Slime GRPO)

    Usage::

        from recipe.slime.formats.slime_format import SlimeRLFormat

        fmt = SlimeRLFormat(tokenizer)
        episode = traj.to_rl_records(fmt)
        # episode["tokens"], episode["loss_mask"], episode["rollout_log_probs"], ...

    Contract:
    - Implementations MUST call to_rl_records() only after backfill_token_annotations()
      has been called (all traj.steps[t].token_annotation must be non-None).
    - to_episode() returns a single flat dict representing the full episode.
    - Key names are framework-specific (for example Slime uses "tokens").
    """

    def to_episode(self, traj: "StatefulTrajectory") -> dict:
        """Convert a full trajectory into a single RL episode training record.

        Args:
            traj: trajectory with all steps' token_annotation already populated.

        Returns:
            dict with framework-specific fields for one training episode.
        """
        ...


# ─── TrajectoryStep ───────────────────────────────────────────────────────────


@dataclass
class TrajectoryStep:
    """
    s_t = (z_t, Δz_t, a_t, o_t, e_t, r_t) + assembled context snapshot.

    Fields:
    - state_snapshot: z_t — raw state before step (state.raw_messages, slots, tokens, cost)
    - state_delta: Δz_t — explicit slot changes this step
    - action: a_t — model response (content + tool_calls)
    - observation: o_t — tool execution results
    - event: e_t — StepEndEvent
    - reward: r_t — filled by EvaluationProcessor.on_task_end()
    - step_start_event: assembled context for this step (system prompt + managed history)
      captured from StepStartEvent; makes the step self-contained for training.
    - subagent_trajectories: child agent trajectories spawned during this step.
    """

    step_id: int
    state_snapshot: FullStateSnapshot  # z_t: raw state before step
    state_delta: StateDelta  # Δz_t: explicit slot changes this step
    action: ModelResponseEvent | None  # a_t: model response
    observation: list[ToolResultEvent]  # o_t: tool results
    event: Event | None  # e_t: StepEndEvent
    reward: float = 0.0  # r_t: filled by EvaluationProcessor
    step_start_event: "StepStartEvent | None" = None  # assembled context (system prompt + history)
    subagent_trajectories: "list[StatefulTrajectory]" = field(default_factory=list)
    token_annotation: "TokenAnnotation | None" = None  # filled by token-aware rollout/provider


# ─── StatefulTrajectory ───────────────────────────────────────────────────────


@dataclass
class StatefulTrajectory:
    """
    τ = {s_1, ..., s_T} — RunLoop's first-class output.

    Not a log side-product; built inline during run_loop() execution and
    returned as HarnessResult.trajectory alongside TaskEndEvent.

    parent_run_id links child trajectories back to the spawning parent step,
    enabling full multi-agent causal chain reconstruction.
    """

    run_id: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    parent_run_id: str | None = None  # set by spawner for child agent linkage

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def backfill_rewards(self, eval_result: object) -> None:
        """
        Backfill reward from EvalResult into all steps.
        Evaluator calls this after task completion.
        """
        reward = getattr(eval_result, "reward", 0.0)
        for step in self.steps:
            step.reward = reward

    def to_training_records(self) -> list[dict]:
        """
        SFT/GRPO training records with full OpenAI-format messages per step.

        Each record contains:
        1. System message (from ce_output.system_prompt if available)
        2. Full conversation history up to this step (from ce_output.messages or state_snapshot)
        3. This step's assistant response with proper tool_calls structure (OpenAI format)
        4. This step's tool result messages (observations)

        Multi-agent support — if a step has subagent_trajectories, their records are
        recursively included with parent_run_id and parent_step_id in metadata, producing
        a complete causal chain training record for multi-agent tasks.

        Format is compatible with Claude Code SFT / OpenAI chat fine-tuning.
        """
        records = []
        for step in self.steps:
            messages = _build_step_messages(step)

            record: dict = {
                "run_id": self.run_id,
                "step": step.step_id,
                "messages": messages,
                "reward": step.reward,
                "metadata": {
                    "cumulative_tokens": step.state_snapshot.cumulative_tokens,
                    "cumulative_cost_usd": step.state_snapshot.cumulative_cost_usd,
                    "slots": {k: v.content for k, v in step.state_snapshot.slots.items()},
                    "delta": {
                        "created": [op.key for op in step.state_delta.created()],
                        "updated": [op.key for op in step.state_delta.updated()],
                        "deleted": [op.key for op in step.state_delta.deleted()],
                    },
                    "context_system_prompt": (step.step_start_event.system_prompt if step.step_start_event else None),
                    "context_token_count": (step.step_start_event.token_count if step.step_start_event else None),
                    "parent_run_id": self.parent_run_id,
                },
            }

            # Recursively embed sub-agent records for multi-agent causal chain
            if step.subagent_trajectories:
                subagent_records = []
                for sub_traj in step.subagent_trajectories:
                    for sub_record in sub_traj.to_training_records():
                        # Tag each subagent record with parent context
                        sub_record["metadata"]["parent_run_id"] = self.run_id
                        sub_record["metadata"]["parent_step_id"] = step.step_id
                        subagent_records.append(sub_record)
                record["subagent_records"] = subagent_records

            records.append(record)
        return records

    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    # ── Human-readable markdown rendering ──────────────────────────────

    def to_markdown(
        self,
        level: str = "full",
        *,
        task: Any = None,
        result: Any = None,
        config: Any = None,
    ) -> str:
        """Render this trajectory as human/LLM-readable markdown.

        Consolidates the previous two duplicated builders (plugin-side and
        recipe-side) into a single canonical renderer used by meta-harness
        reflect / runner trajectory dumps.

        Args:
            level:  ``"summary"`` — exit reason + final output + eval only.
                    ``"full"`` — task, result, harness config, diagnostics,
                    per-step thinking/response/tool-calls/observations.
            task:   Optional :class:`~harnessx.core.base_task.BaseTask` —
                    adds description and (if present) task_id.
            result: Optional :class:`~harnessx.core.harness.HarnessResult` —
                    source of exit_reason, totals, and eval_result; falls
                    back to self.steps when omitted.
            config: Optional :class:`~harnessx.core.harness.HarnessConfig` —
                    adds processor + tool inventory section. ``full`` only.
        """
        if level not in ("summary", "full"):
            raise ValueError(f"level must be 'summary' or 'full', got {level!r}")

        task_end = getattr(result, "task_end", None) if result is not None else None
        exit_reason = getattr(result, "exit_reason", None) or getattr(task_end, "exit_reason", None) or "?"
        total_steps = getattr(result, "total_steps", None) or getattr(task_end, "total_steps", None) or len(self.steps)
        final_output = getattr(result, "final_output", None) or getattr(task_end, "final_output", None) or ""
        eval_result = getattr(result, "eval_result", None) or getattr(task_end, "eval_result", None)

        task_id = getattr(task, "task_id", "") if task is not None else ""
        header = f"# Trajectory: {task_id}" if task_id else "# Trajectory"
        lines: list[str] = [header]

        if level == "full" and task is not None:
            desc = getattr(task, "description", "") or ""
            if not isinstance(desc, str):
                desc = str(desc)
            if desc:
                lines.append(f"\n## Task\n\n{desc}")

        lines.append("\n## Result\n")
        lines.append(f"- exit_reason: {exit_reason}")
        lines.append(f"- total_steps: {total_steps}")
        if final_output:
            truncated = final_output if level == "full" else final_output[:500]
            lines.append(f"- final_output: {truncated}")
        if eval_result is not None:
            lines.append(f"- eval_passed: {getattr(eval_result, 'passed', '?')}")
            lines.append(f"- eval_score: {getattr(eval_result, 'score', '?')}")
            reason = getattr(eval_result, "reason", None)
            if reason:
                lines.append(f"- eval_reason: {reason}")

        if level == "summary":
            return "\n".join(lines)

        if config is not None:
            proc_parts: list[str] = []
            for _hook, procs in (getattr(config, "processors", None) or {}).items():
                for p in procs:
                    group = getattr(p, "_singleton_group", "") or ""
                    order = getattr(p, "_order", "?")
                    label = group or type(p).__name__
                    proc_parts.append(f"{label}({order})")
            registry = getattr(config, "tool_registry", None)
            tool_names: list[str] = []
            if registry is not None and hasattr(registry, "list_names"):
                try:
                    tool_names = list(registry.list_names())
                except Exception:
                    tool_names = []
            if proc_parts or tool_names:
                lines.append("\n## Harness Config\n")
                if proc_parts:
                    lines.append(f"Processors: {', '.join(proc_parts)}")
                if tool_names:
                    lines.append(f"Tools: [{', '.join(sorted(tool_names))}]")

        total_tool_calls = 0
        tool_errors = 0
        tool_call_counts: dict[str, int] = {}
        for step in self.steps:
            for tr in step.observation or []:
                total_tool_calls += 1
                tname = getattr(tr, "tool_name", "?")
                tool_call_counts[tname] = tool_call_counts.get(tname, 0) + 1
                if getattr(tr, "error", ""):
                    tool_errors += 1

        max_steps = getattr(task, "max_steps", 0) if task is not None else 0
        total_tokens = getattr(result, "total_tokens", 0) or 0 if result is not None else 0
        total_cost = getattr(result, "total_cost_usd", 0) or 0 if result is not None else 0
        max_cost = getattr(task, "max_cost_usd", 0) if task is not None else 0

        lines.append("\n## Diagnostics\n")
        if max_steps:
            pct = f"{100 * total_steps // max_steps}%"
            lines.append(f"- steps: {total_steps}/{max_steps} ({pct} budget)")
        else:
            lines.append(f"- steps: {total_steps}")
        lines.append(f"- tokens: {total_tokens}")
        if max_cost:
            cost_pct = f"{100 * total_cost / max_cost:.0f}%"
            lines.append(f"- cost: ${total_cost:.3f}/${max_cost:.2f} ({cost_pct} budget)")
        else:
            lines.append(f"- cost: ${total_cost:.3f}")
        err_rate = f"{100 * tool_errors / total_tool_calls:.0f}%" if total_tool_calls else "0%"
        lines.append(f"- tool_calls: {total_tool_calls}, errors: {tool_errors} (error_rate={err_rate})")
        if tool_call_counts:
            top = sorted(tool_call_counts.items(), key=lambda x: -x[1])[:5]
            lines.append(f"- top_tools: {', '.join(f'{n}({c})' for n, c in top)}")

        if self.steps:
            lines.append("\n---\n")
            lines.append("## Execution Steps\n")
            for step in self.steps:
                lines.append(f"\n### Step {step.step_id}")
                action = step.action
                if action is not None:
                    thinking = getattr(action, "thinking", "") or ""
                    raw = getattr(action, "content", None)
                    content = raw if isinstance(raw, str) else (str(raw) if raw else "")
                    if thinking:
                        lines.append(f"\n#### Thinking\n\n{thinking}")
                    if content:
                        lines.append(f"\n#### Response\n\n{content}")
                    tool_calls = getattr(action, "tool_calls", None) or ()
                    if tool_calls:
                        lines.append("\n#### Tool Calls\n")
                        for tc in tool_calls:
                            try:
                                input_str = json.dumps(tc.input, ensure_ascii=False)
                            except (TypeError, ValueError):
                                input_str = str(getattr(tc, "input", ""))
                            lines.append(f"- **{tc.name}**(`{input_str}`)")
                for tr in step.observation or []:
                    tname = getattr(tr, "tool_name", "?")
                    error_str = getattr(tr, "error", "") or ""
                    result_str = getattr(tr, "result", "") or ""
                    if error_str:
                        lines.append(f"  -> {tname}: ERROR: {error_str}")
                    else:
                        lines.append(f"  -> {tname}: {result_str}")

        return "\n".join(lines)

    # ── Token-level RL training records ────────────────────────────────

    def has_token_annotations(self) -> bool:
        """True iff every step has a filled TokenAnnotation."""
        return bool(self.steps) and all(s.token_annotation is not None for s in self.steps)

    def to_rl_records(self, fmt: "RLFormat") -> dict:
        """
        Convert this trajectory into a single RL episode training record.

        Dispatches to ``fmt.to_episode(self)``.  ``fmt`` is a framework-specific
        format object (e.g. ``SlimeRLFormat``) that knows the exact field names
        and layout expected by the downstream training framework.

        Requires :meth:`has_token_annotations` to be True — call
        ``backfill_token_annotations(traj, provider)`` first.

        Args:
            fmt: an object implementing the :class:`RLFormat` protocol.

        Returns:
            dict — framework-specific episode record (keys vary by format).

        Example::

            from recipe.slime.formats.slime_format import SlimeRLFormat

            fmt = SlimeRLFormat(tokenizer)
            episode = traj.to_rl_records(fmt)
            # episode["tokens"], episode["loss_mask"], episode["rollout_log_probs"], ...
        """
        if not self.has_token_annotations():
            raise ValueError(
                "to_rl_records() requires all steps to have token_annotation. "
                "Call backfill_token_annotations(traj, provider) first."
            )
        return fmt.to_episode(self)


# ─── Message Building Helpers ─────────────────────────────────────────────────


def _build_step_messages(step: TrajectoryStep) -> list[dict]:
    """
    Build the complete OpenAI-format message sequence for a single TrajectoryStep.

    Priority:
    1. Use ce_output.messages as context base (includes CE-assembled system prompt + managed history)
    2. Fall back to state_snapshot.messages (raw state, no system prompt)

    Then append:
    - The action (assistant message with tool_calls in OpenAI format)
    - The observations (tool result messages in OpenAI format)
    """
    messages: list[dict] = []

    # ── Context base (system + history) ──────────────────────────────────────
    if step.step_start_event is not None and step.step_start_event.messages:
        # Assembled context: includes system prompt (if any) + managed history
        for m in step.step_start_event.messages:
            msg = message_to_dict(m)
            messages.append(msg)
    else:
        # Fallback: raw state messages
        for m in step.state_snapshot.messages:
            msg = message_to_dict(m)
            messages.append(msg)

    # ── Action (assistant message, proper OpenAI tool_calls format) ────
    if step.action is not None:
        assistant_msg = _action_to_dict(step.action)
        messages.append(assistant_msg)

    # ── Observations (tool result messages in OpenAI format) ─────────────────
    for obs in step.observation:
        messages.append(_tool_result_to_dict(obs))

    return messages


def _action_to_dict(action: ModelResponseEvent) -> dict:
    """
    Convert ModelResponseEvent to OpenAI-format assistant message.

    If tool_calls present: use OpenAI tool_calls format with function objects.
    Content can be None/empty when tool_calls is present (per OpenAI spec).
    thinking is included when present so the trajectory is a complete record.
    """
    if action.tool_calls:
        tool_calls_list = []
        for tc in action.tool_calls:
            tool_calls_list.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input, ensure_ascii=False),
                    },
                }
            )
        msg: dict = {
            "role": "assistant",
            "content": action.content or None,  # null when only tool calls
            "tool_calls": tool_calls_list,
        }
    else:
        msg = {
            "role": "assistant",
            "content": action.content or "",
        }
    if action.thinking:
        msg["thinking"] = action.thinking
    return msg


def _tool_result_to_dict(obs: ToolResultEvent) -> dict:
    """
    Convert ToolResultEvent to OpenAI-format tool result message.
    """
    content = obs.result if not obs.error else f"Error: {obs.error}"
    return {
        "role": "tool",
        "tool_call_id": obs.tool_call_id,
        "name": obs.tool_name,
        "content": content,
    }
