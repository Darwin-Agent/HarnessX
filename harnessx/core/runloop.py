# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import time
from typing import TYPE_CHECKING

from ..tools.mcp import _mcp_results_ctx, enforce_turn_budget

from .events import (
    BeforeModelEvent,
    Message,
    ModelResponseEvent,
    SegmentBoundaryEvent,
    SpawnSubAgentEvent,
    StepEndEvent,
    StepStartEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    compute_history_hash,
    compute_windows,
    make_run_id,
    rough_token_count,
)
from .processor import ContractViolationError, Processor, pipe_all

_contract_logger = logging.getLogger("harnessx.contract")


def _enforce_boundary_invariant(
    boundary_events: list,
    hash_before: str,
    hash_after: str,
    step_id: int,
) -> None:
    """Invariant 7: processor must not emit SegmentBoundaryEvent when history_hash is unchanged.

    Raises ContractViolationError in strict mode; logs a warning otherwise.
    """
    if not (boundary_events and hash_before == hash_after):
        return
    msg = (
        f"step_start step={step_id}: processor emitted SegmentBoundaryEvent "
        "but history_hash is unchanged — spurious boundary"
    )
    if os.environ.get("HARNESSX_CONTRACT_MODE", "warn").lower() == "strict":
        raise ContractViolationError(
            msg,
            hook="step_start",
            violation_type="spurious_boundary",
            step_id=step_id,
        )
    _contract_logger.warning("CONTRACT [spurious_boundary] hook=step_start step=%d: %s", step_id, msg)


from .state import State
from .trajectory import (
    FullStateSnapshot,
    StatefulTrajectory,
    StateSlotSnapshot,
    TrajectoryStep,
)

if TYPE_CHECKING:
    from typing import Callable
    from ..providers.base import BaseModelProvider
    from ..tools.base import BaseToolRegistry
    from ..tracing.base import BaseTracer
    from ..workspace.workspace import Workspace

_logger = logging.getLogger(__name__)


class HarnessError(Exception):
    pass


class BudgetExceededError(HarnessError):
    pass


class LoopDetectedError(HarnessError):
    pass


class ModelParseError(HarnessError):
    pass


_USER_INTERRUPTED_MESSAGE = "user actively interrupted execution"


def _append_user_interrupted_message(state: State) -> None:
    """Append one terminal assistant message describing user interruption.

    The message is written into the factual track (`raw_messages`) so the
    effective track (`messages`) stays in sync and resumed turns keep a complete
    user/assistant transcript.
    """
    if state.raw_messages:
        last = state.raw_messages[-1]
        if last.role == "assistant" and str(last.content).strip() == _USER_INTERRUPTED_MESSAGE:
            return
    state.add_raw_message(Message(role="assistant", content=_USER_INTERRUPTED_MESSAGE))


async def run_loop(
    task: object,
    state: State,
    model_provider: "BaseModelProvider",
    tool_registry: "BaseToolRegistry",
    tracer: "BaseTracer",
    processors: dict[str, list[Processor]],
    workspace: "Workspace | None" = None,
    parent_run_id: "str | None" = None,
    step_snapshots: bool = True,
    stream_callback: "object | None" = None,
    model_selector: "Callable[[State], BaseModelProvider] | None" = None,
    model_config: "object | None" = None,
    harness_config: "object | None" = None,
    child_harness_config: "object | None" = None,
) -> "tuple[TaskEndEvent, StatefulTrajectory, ToolCall | None]":
    """
    Core RunLoop. Pure loop — context assembly, memory, and evaluation
    are all handled by processors registered on the step_start / step_end /
    task_end hooks.

    Returns (TaskEndEvent, StatefulTrajectory, interrupted_at) triple.

    Interrupt/resume: if model calls a tool whose name is in task.interrupt_on,
    run_loop exits immediately with exit_reason='interrupted' and returns the
    ToolCall as the third element. Caller can resume by passing the state back.

    step_snapshots=False skips storing large message data inside trajectory
    steps to reduce memory usage (O(n²) → O(n) across n steps).  Disable in
    RL training where reward_func never reads step snapshots or step_start_event.
    Nudge behaviour is controlled by registered processors, not this flag.
    """
    # Initialize trajectory
    trajectory = StatefulTrajectory(run_id=state.run_id)
    interrupted_at: ToolCall | None = None
    if parent_run_id is not None:
        state.parent_run_id = parent_run_id

    _star_procs: list[Processor] = processors.get("*", [])

    def get_procs(key: str) -> list[Processor]:
        specific = processors.get(key)
        return _star_procs + specific if specific else _star_procs

    # Pull pending command side-channels out of state before processors run.
    # Both are plain attributes set by Harness.run() (not StateSlot objects).
    _cmd_prompt_prefix = getattr(state, "_pending_command_prompt", None) or ""
    if _cmd_prompt_prefix:
        try:
            del state._pending_command_prompt  # type: ignore[attr-defined]
        except AttributeError:
            pass

    # allowed_tools: when a plugin command restricts the tool set, only those
    # tool names are visible to the model for the entire task.  None means no
    # restriction (all tools from the registry are available).
    _raw_allowed = getattr(state, "_pending_command_allowed_tools", None)
    _task_allowed_tools: frozenset[str] | None = frozenset(_raw_allowed) if _raw_allowed else None
    if _raw_allowed is not None:
        try:
            del state._pending_command_allowed_tools  # type: ignore[attr-defined]
        except AttributeError:
            pass

    def _get_tools() -> tuple:
        """Return tool schemas, optionally filtered by the active command's allowed_tools."""
        schemas = tool_registry.get_schemas()
        if _task_allowed_tools is not None:
            schemas = [s for s in schemas if s.name in _task_allowed_tools]
        return tuple(schemas)

    # Emit TaskStart — on_task_start processors assemble the static system prompt
    start_event = TaskStartEvent(
        run_id=state.run_id,
        step_id=0,
        task_description=(
            task.description
            if isinstance(task.description, str)
            else " ".join(
                b.get("text", "") for b in task.description if isinstance(b, dict) and b.get("type") == "text"
            )
        ),
        model=getattr(model_provider, "model", ""),
        parent_run_id=parent_run_id,
        session_id=getattr(tracer, "session_id", ""),
        state=state,
        workspace=workspace,
        tools=_get_tools(),
        system_prompt=_cmd_prompt_prefix,
    )
    _ts_events = await pipe_all(start_event, get_procs("task_start"), tracer=tracer, hook="task_start")
    start_event = next(
        (e for e in reversed(_ts_events) if isinstance(e, (TaskStartEvent, TaskEndEvent))),
        start_event,
    )
    # SlashCommandProcessor (and any other PRE-phase processor) may short-circuit
    # the entire run by yielding a TaskEndEvent instead of a TaskStartEvent.
    if isinstance(start_event, TaskEndEvent):
        await tracer.on_event(start_event)
        await tracer.flush()
        return start_event, trajectory, None
    # Send TaskStartEvent first so the journal initialises _session_dir before any
    # SegmentBoundaryEvent arrives.  The boundary check below may rotate the segment
    # immediately after, but _session_dir is guaranteed non-None at that point.
    await tracer.on_event(start_event)
    # Detect system prompt changes across tasks within the same session.
    # state.last_sys_prompt_hash persists in snapshots so this works after restart.
    _new_sys_hash = hashlib.sha256((start_event.system_prompt or "").encode()).hexdigest()
    if state.last_sys_prompt_hash is not None and state.last_sys_prompt_hash != _new_sys_hash:
        # Update hash before snapshot so the checkpoint carries the new value,
        # closing the crash window where step_state.json would have the old hash.
        state.last_sys_prompt_hash = _new_sys_hash
        _sp_boundary = SegmentBoundaryEvent(
            run_id=state.run_id,
            step_id=state.step,
            reason="system_prompt_change",
            new_run_id=make_run_id(),
            state_snapshot=state.snapshot(),
            compacted_messages=tuple(state.messages),
            compacted_raw_messages=tuple(state.raw_messages),
        )
        await tracer.on_event(_sp_boundary)
        state.run_id = _sp_boundary.new_run_id
        start_event = dataclasses.replace(start_event, run_id=state.run_id)
    else:
        state.last_sys_prompt_hash = _new_sys_hash
    # Freeze the system prompt for the entire task — StepStartEvent always
    # inherits this value; on_step_start processors must NOT modify it.
    task_system_prompt: str = start_event.system_prompt

    exit_reason = "done"
    final_output = ""
    error_message = ""
    last_model_input_tokens: int = 0  # input_tokens of the most recent model call
    last_model_output_tokens: int = 0  # output_tokens of the most recent model call
    _cost_warned = False
    _model_empty_end_turn_seen = False
    _empty_end_turn_retried = False
    try:
        while True:
            # Check budget before each step
            if state.budget_exceeded():
                exit_reason = "budget_exceeded"
                break

            step_id = state.step
            step_start_ts = time.monotonic()
            active_model_provider = model_provider
            if model_selector is not None:
                try:
                    selected = model_selector(state)
                    if selected is not None:
                        active_model_provider = selected
                except Exception:
                    _logger.warning(
                        "run_loop: model_selector failed, falling back to main provider",
                        exc_info=True,
                    )
            context_window: int = getattr(active_model_provider, "context_window", 64_000)

            # ── 0. Capture pre-step state snapshot z_t ─────────────────────
            # step_snapshots=False: skip copying the full message history — it grows
            # O(t) at step t, giving O(n²) total allocations across n steps.
            # reward_func() never reads TrajectoryStep.state_snapshot.messages;
            # only .diff() uses slots, which are still copied for StateDelta.
            if not step_snapshots:
                snapshot_before = FullStateSnapshot(
                    step_id=step_id,
                    messages=(),
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
            else:
                snapshot_before = FullStateSnapshot.from_state(state, step_id=step_id)

            # ── 1. step_start: context assembly (delegated to processors) ───
            step_start_event = StepStartEvent(
                run_id=state.run_id,
                step_id=step_id,
                raw_messages=tuple(state.raw_messages),
                messages=tuple(state.messages),
                task=task,
                tools=_get_tools(),  # respects command-level allowed_tools restriction
                context_window=context_window,
                workspace=workspace,
                system_prompt=task_system_prompt,  # frozen by on_task_start; never modified by on_step_start
                token_count=rough_token_count(list(state.messages)),
            )
            # Snapshot history_hash before processors run so we can detect structural changes.
            _pre_step_msgs = step_start_event.messages or tuple(state.messages)
            _, _hist_before, _ = compute_windows(_pre_step_msgs)
            _history_hash_before = compute_history_hash(_hist_before)

            # pipe_all: collect SegmentBoundaryEvents
            step_start_events = await pipe_all(
                step_start_event,
                get_procs("step_start"),
                tracer=tracer,
                hook="step_start",
            )
            boundary_events: list[SegmentBoundaryEvent] = [
                e for e in step_start_events if isinstance(e, SegmentBoundaryEvent)
            ]
            step_start_event = next(
                (e for e in reversed(step_start_events) if isinstance(e, StepStartEvent)),
                step_start_event,
            )

            # Compute history_hash after step_start processors ran (needed for both
            # invariant-7 check and auto-boundary generation).
            _post_step_msgs = step_start_event.messages or _pre_step_msgs
            _, _hist_after, _ = compute_windows(_post_step_msgs)
            _history_hash_after = compute_history_hash(_hist_after)

            # Invariant 7: spurious boundary detection (hash equal → no boundary allowed).
            _enforce_boundary_invariant(boundary_events, _history_hash_before, _history_hash_after, step_id)

            # Auto-detect structural history changes not already covered by a
            # processor-emitted SegmentBoundaryEvent.  If history_window changed
            # but no processor signalled a boundary, RunLoop generates one now.
            if not boundary_events and _history_hash_before != _history_hash_after:
                _hint = step_start_event.boundary_hint
                _auto_boundary = SegmentBoundaryEvent(
                    run_id=state.run_id,
                    step_id=step_id,
                    reason=_hint.reason if _hint else "auto_history_mutation",
                    new_run_id=make_run_id(),
                    state_snapshot=None,
                    compacted_messages=tuple(_post_step_msgs),
                    compacted_raw_messages=tuple(step_start_event.raw_messages or _post_step_msgs),
                    before_msgs=_hint.before_msgs if _hint else 0,
                    after_msgs=_hint.after_msgs if _hint else 0,
                    before_tokens=_hint.before_tokens if _hint else 0,
                    after_tokens=_hint.after_tokens if _hint else 0,
                )
                boundary_events = [_auto_boundary]

            # Handle segment boundary: update state.run_id and notify tracer
            for boundary in boundary_events:
                await tracer.on_event(boundary)
                state.run_id = boundary.new_run_id
                if boundary.compacted_messages:
                    state.raw_messages = list(boundary.compacted_messages)
                    state.messages = list(boundary.compacted_messages)
            await tracer.on_event(step_start_event)

            # ── 2. Before model ─────────────────────────────────────────────
            # RunLoop owns a safe default assembly path so step_start processors
            # are optional: if no processor produced messages, start from the
            # current effective context. Always ensure system prompt is present.
            _assembled = step_start_event.messages or tuple(state.messages)
            _assembled_list = list(_assembled)
            if task_system_prompt:
                # Strip any stale system message and inject the current task prompt.
                # On resume, state.messages may retain a system from a prior session;
                # always replace it so the model sees a fresh prompt rather than
                # silently perpetuating stale content.
                _assembled_list = [m for m in _assembled_list if m.role != "system"]
                _assembled_list.insert(0, Message(role="system", content=task_system_prompt))
            before_event = BeforeModelEvent(
                run_id=state.run_id,
                step_id=step_id,
                messages=tuple(_assembled_list),
                tools=step_start_event.tools,
                cumulative_cost_usd=state.cumulative_cost_usd,
            )
            _bm_events = await pipe_all(
                before_event,
                get_procs("before_model"),
                tracer=tracer,
                hook="before_model",
            )
            before_event = next(
                (e for e in reversed(_bm_events) if isinstance(e, BeforeModelEvent)),
                before_event,
            )
            await tracer.on_event(before_event)

            # ── 3. Call model ────────────────────────────────────────────────
            if before_event.skip_model:
                model_event = ModelResponseEvent(
                    run_id=state.run_id,
                    step_id=step_id,
                    content=before_event.synthetic_output,
                    finish_reason="stop",
                    usage=Usage(),
                )
                spawn_events: list[SpawnSubAgentEvent] = []
            else:
                # Validate message sequence before sending to API:
                # - Ensure tool_use/tool_result pairs are complete
                # - Ensure messages don't end with assistant role (would be interpreted as prefill)
                _before_msg_list = list(before_event.messages)
                final_messages = _validate_messages(_before_msg_list)
                model_response = await active_model_provider.complete(
                    messages=final_messages,
                    tools=list(before_event.tools),
                    stream_callback=stream_callback,
                )
                model_event = ModelResponseEvent(
                    run_id=state.run_id,
                    step_id=step_id,
                    content=model_response.content,
                    thinking=model_response.thinking,
                    thinking_blocks=model_response.thinking_blocks,
                    tool_calls=model_response.tool_calls,
                    finish_reason=model_response.finish_reason,
                    usage=model_response.usage,
                    model=model_response.model,
                )
                # after_model uses pipe_all: processors may yield SpawnSubAgentEvents
                after_events = await pipe_all(model_event, get_procs("after_model"), tracer=tracer, hook="after_model")
                spawn_events = [e for e in after_events if isinstance(e, SpawnSubAgentEvent)]
                model_event = next(
                    (e for e in reversed(after_events) if isinstance(e, ModelResponseEvent)),
                    model_event,
                )
            await tracer.on_event(model_event)
            for se in spawn_events:
                await tracer.on_event(se)

            # Update token/cost tracking
            state.cumulative_tokens += model_event.usage.total_tokens
            state.cumulative_input_tokens += model_event.usage.input_tokens
            state.cumulative_output_tokens += model_event.usage.output_tokens
            last_model_input_tokens = model_event.usage.input_tokens
            last_model_output_tokens = model_event.usage.output_tokens
            cost = _estimate_cost(model_event.usage)
            state.cumulative_cost_usd += cost

            if (
                not _cost_warned
                and state.max_cost_usd is not None
                and state.cumulative_cost_usd >= state.max_cost_usd * 0.7
            ):
                _cost_warned = True
                remaining_pct = 100 * (1 - state.cumulative_cost_usd / state.max_cost_usd)
                state.add_raw_message(
                    Message(
                        role="user",
                        content=(
                            f"[COST WARNING: You have used {state.cumulative_cost_usd:.2f} of "
                            f"${state.max_cost_usd:.2f} budget ({remaining_pct:.0f}% remaining). "
                            "Start wrapping up — provide your FINAL ANSWER soon. "
                            "A best-effort answer is better than running out of budget.]"
                        ),
                    )
                )

            if not before_event.skip_model:
                state.add_raw_message(
                    Message(
                        role="assistant",
                        content=model_event.content,
                        tool_calls=model_event.tool_calls,
                        thinking=model_event.thinking,
                        thinking_blocks=model_event.thinking_blocks,
                    )
                )

            if model_event.content:
                final_output = model_event.content

            # ── 4. Execute tools ─────────────────────────────────────────────
            step_tool_results: list[ToolResultEvent] = []
            interrupt_triggered = False
            # Compute MCP results dir once per step (stable across all tool calls).
            _tracer_session = getattr(tracer, "session_id", "")
            _step_mcp_dir: str | None = None
            if hasattr(tracer, "base_dir") and _tracer_session:
                _step_mcp_dir = os.path.join(tracer.base_dir, _tracer_session, "mcp_results")
            _pre_tool_raw_idx = len(state.raw_messages)
            for tc in model_event.tool_calls:
                interrupt_on = getattr(task, "interrupt_on", []) or []
                if tc.name in interrupt_on:
                    interrupted_at = tc
                    exit_reason = "interrupted"
                    interrupt_triggered = True
                    break

                tc_event = ToolCallEvent(
                    run_id=state.run_id,
                    step_id=step_id,
                    tool_name=tc.name,
                    tool_input=tc.input,
                    tool_call_id=tc.id,
                    approved=True,
                )
                _bt_events = await pipe_all(
                    tc_event,
                    get_procs("before_tool"),
                    tracer=tracer,
                    hook="before_tool",
                )
                tc_event = next(
                    (e for e in reversed(_bt_events) if isinstance(e, ToolCallEvent)),
                    tc_event,
                )
                await tracer.on_event(tc_event)

                # Propagate run context for spawn_subagent and similar tools
                try:
                    from ..tools.spawn_subagent import _spawn_ctx

                    _spawn_ctx_token = _spawn_ctx.set(
                        {
                            "run_id": state.run_id,
                            "step_id": step_id,
                            "spawn_depth": state.spawn_depth,
                            "state": state,
                            "tracer": tracer,
                            "model_config": model_config,
                            "harness_config": harness_config,
                            "child_harness_config": child_harness_config,
                        }
                    )
                except ImportError:
                    _spawn_ctx_token = None

                _mcp_ctx_token = _mcp_results_ctx.set(_step_mcp_dir)
                try:
                    if tc_event.approved:
                        t0 = time.monotonic()
                        try:
                            result = await tool_registry.execute(tc_event.tool_name, tc_event.tool_input)
                            tr_event = ToolResultEvent(
                                run_id=state.run_id,
                                step_id=step_id,
                                tool_name=tc_event.tool_name,
                                tool_call_id=tc_event.tool_call_id,
                                result=result.output,
                                error=result.error,
                                duration_ms=(time.monotonic() - t0) * 1000,
                                content_blocks=tuple(result.content_blocks) if result.content_blocks else (),
                            )
                        except Exception as e:
                            tr_event = ToolResultEvent(
                                run_id=state.run_id,
                                step_id=step_id,
                                tool_name=tc_event.tool_name,
                                tool_call_id=tc_event.tool_call_id,
                                result="",
                                error=str(e),
                                duration_ms=(time.monotonic() - t0) * 1000,
                            )
                        _at_events = await pipe_all(
                            tr_event,
                            get_procs("after_tool"),
                            tracer=tracer,
                            hook="after_tool",
                        )
                        tr_event = next(
                            (e for e in reversed(_at_events) if isinstance(e, ToolResultEvent)),
                            tr_event,
                        )
                        await tracer.on_event(tr_event)
                        state.add_tool_result(tr_event)
                        step_tool_results.append(tr_event)

                        result_text = tr_event.result if not tr_event.error else f"Error: {tr_event.error}"
                        if tr_event.content_blocks:
                            result_content: str | list = [
                                {"type": "text", "text": result_text},
                                *tr_event.content_blocks,
                            ]
                        else:
                            result_content = result_text
                        state.add_raw_message(
                            Message(
                                role="tool",
                                content=result_content,
                                tool_call_id=tc_event.tool_call_id,
                                name=tc_event.tool_name,
                            )
                        )
                    elif tc_event.synthetic_result is not None:
                        tr_event = ToolResultEvent(
                            run_id=state.run_id,
                            step_id=step_id,
                            tool_name=tc_event.tool_name,
                            tool_call_id=tc_event.tool_call_id,
                            result=tc_event.synthetic_result,
                            duration_ms=0.0,
                        )
                        _at_syn_events = await pipe_all(
                            tr_event,
                            get_procs("after_tool"),
                            tracer=tracer,
                            hook="after_tool",
                        )
                        tr_event = next(
                            (e for e in reversed(_at_syn_events) if isinstance(e, ToolResultEvent)),
                            tr_event,
                        )
                        await tracer.on_event(tr_event)
                        state.add_tool_result(tr_event)
                        step_tool_results.append(tr_event)
                        state.add_raw_message(
                            Message(
                                role="tool",
                                content=tr_event.result if not tr_event.error else f"Error: {tr_event.error}",
                                tool_call_id=tc_event.tool_call_id,
                                name=tc_event.tool_name,
                            )
                        )
                    else:
                        state.add_raw_message(
                            Message(
                                role="tool",
                                content="Tool call not approved.",
                                tool_call_id=tc_event.tool_call_id,
                                name=tc_event.tool_name,
                            )
                        )
                finally:
                    # Always reset per-tool-call context vars, even if a processor raises.
                    _mcp_results_ctx.reset(_mcp_ctx_token)
                if _spawn_ctx_token is not None:
                    try:
                        from ..tools.spawn_subagent import _spawn_ctx

                        _spawn_ctx.reset(_spawn_ctx_token)
                    except Exception:
                        pass
                    _spawn_ctx_token = None

            # ── 4b. Aggregate turn budget ────────────────────────────────────
            # Spill the largest tool results to disk if the step total exceeds
            # _TURN_BUDGET_CHARS, replacing the already-added raw messages.
            if step_tool_results and not interrupt_triggered:
                _new_tool_msgs = await enforce_turn_budget(
                    state.raw_messages[_pre_tool_raw_idx:],
                    results_dir=_step_mcp_dir,
                )
                state.raw_messages[_pre_tool_raw_idx:] = _new_tool_msgs
                state.messages[_pre_tool_raw_idx:] = _new_tool_msgs

            if interrupt_triggered:
                break

            # ── 5. Step end ──────────────────────────────────────────────────
            state.step += 1
            tool_call_summary = "|".join(f"{tc.name}:{str(tc.input)}" for tc in model_event.tool_calls)
            step_event = StepEndEvent(
                run_id=state.run_id,
                step_id=step_id,
                step_summary=model_event.content[:100] if model_event.content else "",
                tool_call_summary=tool_call_summary,
                cumulative_tokens=state.cumulative_tokens,
                cumulative_cost_usd=state.cumulative_cost_usd,
                # step_snapshots=False: skip snapshot to avoid O(n²) memory allocation.
                state_snapshot=None if not step_snapshots else state.snapshot(),
                duration_ms=(time.monotonic() - step_start_ts) * 1000,
                input_tokens=model_event.usage.input_tokens,
                output_tokens=model_event.usage.output_tokens,
            )
            _se_events = await pipe_all(step_event, get_procs("step_end"), tracer=tracer, hook="step_end")
            step_event = next(
                (e for e in reversed(_se_events) if isinstance(e, StepEndEvent)),
                step_event,
            )
            await tracer.on_event(step_event)

            # ── 6. Record Δz_t and build TrajectoryStep ──────────────────────
            delta = snapshot_before.diff(state)
            traj_step = TrajectoryStep(
                step_id=step_id,
                state_snapshot=snapshot_before,
                state_delta=delta,
                action=model_event,
                observation=step_tool_results,
                event=step_event,
                reward=0.0,
                # step_snapshots=False: skip step_start_event — it holds assembled messages
                # which reward_func never reads (saves memory in RL training).
                step_start_event=None if not step_snapshots else step_start_event,
            )
            trajectory.add_step(traj_step)

            # ── 7. Loop termination ──────────────────────────────────────────
            # Thinking-only response: model emitted a reasoning block but no content
            # and no tool calls. NOT a done turn — the model is mid-reasoning (e.g.
            # extended-thinking models that emit <think> before deciding which tool
            # to call). Append a user continuation so the next step's assembled
            # context ends with "user" (required by all provider APIs) and give the
            # model another turn to emit content or a tool call.
            _thinking_only = (
                not model_event.content
                and not model_event.tool_calls
                and (bool(model_event.thinking) or bool(model_event.thinking_blocks))
            )
            if _thinking_only:
                state.add_raw_message(
                    Message(
                        role="user",
                        content="Please continue. Use a tool or provide your final answer.",
                    )
                )
            # Token-limit truncation: model hit max_tokens mid-generation with no tool
            # calls.  The assistant message is already in state; inject a user nudge so
            # the next step's context ends with "user" (required by all provider APIs).
            _length_truncated = (
                model_event.finish_reason == "length" and not model_event.tool_calls and not _thinking_only
            )
            if _length_truncated:
                state.add_raw_message(
                    Message(
                        role="user",
                        content=(
                            "Your previous response was cut off by the token limit. "
                            "Please continue from where you left off."
                        ),
                    )
                )
                _logger.warning(
                    "run_loop: finish_reason=length with no tool calls at step=%d — "
                    "injecting user continuation (run_id=%s)",
                    step_id,
                    state.run_id,
                )
            if (
                not _thinking_only
                and not _length_truncated
                and model_event.finish_reason in ("end_turn", "stop")
                and not model_event.tool_calls
            ):
                empty_content = not (model_event.content or "").strip()
                if step_id == 0 and empty_content and not _empty_end_turn_retried:
                    _model_empty_end_turn_seen = True
                    _empty_end_turn_retried = True
                    state.set_slot("__model_empty_end_turn_seen", "diagnostic", True)
                    state.add_raw_message(
                        Message(
                            role="user",
                            content=(
                                "Your previous response was empty. Please provide a concise, non-empty "
                                "answer for this task now."
                            ),
                        )
                    )
                    _logger.warning(
                        "run_loop: recovered empty first-turn end_turn via single retry (run_id=%s)",
                        state.run_id,
                    )
                    continue
                break
            if task.is_done(state):
                break
            if state.budget_exceeded():
                exit_reason = "budget_exceeded"
                break

    except BudgetExceededError:
        exit_reason = "budget_exceeded"
        # Try to recover the best output from assistant messages
        final_output = _recover_best_output(state.raw_messages, final_output)
    except LoopDetectedError:
        exit_reason = "loop_detected"
        final_output = _recover_best_output(state.raw_messages, final_output)
    except asyncio.CancelledError:
        # User actively interrupted the execution (Ctrl+C / API cancel).
        interrupt_step = state.step - 1 if state.step > 0 else state.step
        await tracer.on_event(
            ModelResponseEvent(
                run_id=state.run_id,
                step_id=interrupt_step,
                content=_USER_INTERRUPTED_MESSAGE,
                finish_reason="interrupted",
                usage=Usage(),
            )
        )
        _append_user_interrupted_message(state)
        exit_reason = "interrupted"
        final_output = _USER_INTERRUPTED_MESSAGE
    except Exception as e:
        # RL rollout: ContextLengthExceeded and GenerationAborted are expected
        # flow-control signals (not bugs).  Log at debug to avoid flooding the
        # console with thousands of tracebacks per run.
        _exc_name = type(e).__name__
        if _exc_name in ("ContextLengthExceeded", "GenerationAborted"):
            _logger.debug("run_loop: %s: %s", _exc_name, e)
        else:
            # Warning without traceback so non-verbose CLI stays clean.
            # Full traceback is available at DEBUG level (harnessx -v).
            _logger.warning("run_loop error: %s: %s", _exc_name, e)
            _logger.debug("run_loop error traceback:", exc_info=True)
        exit_reason = "error"
        final_output = f"Error: {e}"
        error_message = f"{_exc_name}: {e}"

    # Recover best output for non-clean exits (budget/loop hit via normal break)
    if exit_reason in ("budget_exceeded", "loop_detected") and not final_output:
        final_output = _recover_best_output(state.raw_messages, final_output)
    if _model_empty_end_turn_seen:
        state.set_slot("__model_empty_end_turn_seen", "diagnostic", True)
        state.set_slot(
            "__empty_end_turn_recovered",
            "diagnostic",
            bool((final_output or "").strip()),
        )

    # ── 8. Task end ──────────────────────────────────────────────────────────
    end_event = TaskEndEvent(
        run_id=state.run_id,
        step_id=state.step,
        final_output=final_output,
        exit_reason=exit_reason,
        error=error_message,
        total_steps=state.step,
        total_tokens=state.cumulative_tokens,
        total_input_tokens=state.cumulative_input_tokens,
        total_output_tokens=state.cumulative_output_tokens,
        total_cost_usd=state.cumulative_cost_usd,
        last_step_input_tokens=last_model_input_tokens,
        last_step_output_tokens=last_model_output_tokens,
        success_criteria=getattr(task, "success_criteria", ""),
        final_messages=tuple(state.messages),
        state_snapshot=state.snapshot(),
    )
    _te_events = await pipe_all(end_event, get_procs("task_end"), tracer=tracer, hook="task_end")
    end_event = next(
        (e for e in reversed(_te_events) if isinstance(e, TaskEndEvent)),
        end_event,
    )
    await tracer.on_event(end_event)
    await tracer.flush()

    return end_event, trajectory, interrupted_at


def _validate_messages(
    messages: "list[Message]",
) -> "list[Message]":
    """Validate and fix message sequence before sending to model API.

    Fixes known issues:
    1. Orphaned tool results (tool_result without preceding tool_use) — removes them
    2. Assistant messages with tool_calls but missing tool results — removes tool_calls
    3. Consecutive user messages — merges them into one (happens when a prior turn
       ended without an assistant response and a new turn appends another user message)

    Raises AssertionError if messages end with an assistant role — this should never
    happen in a correct run: CompactionProcessor always ends context_snapshot with user,
    and the normal loop flow always ends each turn with tool_result or user messages.
    An assistant-ending message indicates an upstream bug.
    """
    if not messages:
        return messages

    # Pass 1: collect all tool_call_ids and tool_result_ids
    all_tc_ids: set[str] = set()  # from assistant tool_calls
    all_tr_ids: set[str] = set()  # from tool result messages
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                all_tc_ids.add(tc.id)
        if m.role == "tool" and m.tool_call_id:
            all_tr_ids.add(m.tool_call_id)

    # Pass 2: fix orphaned messages
    result = []
    for m in messages:
        if m.role == "tool" and m.tool_call_id and m.tool_call_id not in all_tc_ids:
            continue  # orphaned tool result — skip

        if m.role == "assistant" and m.tool_calls:
            # Check if ALL tool_calls have corresponding results
            matched = [tc for tc in m.tool_calls if tc.id in all_tr_ids]
            unmatched = [tc for tc in m.tool_calls if tc.id not in all_tr_ids]
            if unmatched and not matched:
                # No tool results at all — convert to plain text message
                result.append(
                    Message(
                        role="assistant",
                        content=m.content or "(tool calls were truncated)",
                    )
                )
                continue
            elif unmatched:
                # Partial — keep only matched tool_calls
                result.append(
                    Message(
                        role=m.role,
                        content=m.content,
                        tool_calls=tuple(matched),
                    )
                )
                continue

        result.append(m)

    # Pass 3: merge consecutive user messages.
    # This can happen when a session ended without an assistant response (the
    # unresponded user message is restored from the journal) and a new user
    # message is appended for the current turn.  Provider APIs require
    # user/assistant alternation — merging keeps the full context intact.
    merged: list[Message] = []
    for m in result:
        if merged and merged[-1].role == "user" and m.role == "user":
            prev = merged[-1]
            # Merge string contents; if either is a list (multimodal), keep only the newer one.
            if isinstance(prev.content, str) and isinstance(m.content, str):
                new_content = prev.content + "\n" + m.content if prev.content else m.content
            else:
                new_content = m.content  # non-string: keep the newer message's content
            merged[-1] = Message(role="user", content=new_content)
        else:
            merged.append(m)
    result = merged

    # Raise on assistant-ending messages — this must not happen in a correct run.
    # CompactionProcessor always ends context_snapshot with role="user".
    # If this fires, there is an upstream bug that needs to be fixed, not papered over.
    assert not (result and result[-1].role == "assistant"), (
        "_validate_messages: assembled messages end with 'assistant' role — "
        "this indicates an upstream bug (compaction or processor left messages in "
        "an invalid state for model input)"
    )

    return result


def _estimate_cost(usage: object) -> float:
    """Rough cost estimate: $3/M input, $15/M output (Claude Sonnet pricing)."""
    return (usage.input_tokens * 3.0 + usage.output_tokens * 15.0) / 1_000_000


def _recover_best_output(messages: "list[Message]", current_output: str) -> str:
    """Search messages for the best output to return on abnormal exit.

    Priority: 1) Any message containing 'FINAL ANSWER:' 2) current_output
    3) last assistant message with substantial content 4) last tool result.
    """

    # 1. Search for FINAL ANSWER in any assistant message
    for m in reversed(messages):
        if m.role == "assistant" and isinstance(m.content, str):
            if "FINAL ANSWER:" in m.content.upper():
                return m.content
    # 2. Use current output if it exists and is meaningful
    if current_output and len(current_output.strip()) > 5:
        return current_output
    # 3. Find last assistant message with substantial content
    for m in reversed(messages):
        if m.role == "assistant" and isinstance(m.content, str) and len(m.content.strip()) > 20:
            return m.content
    # 4. Fall back to tool result
    for m in reversed(messages):
        if m.role == "tool" and m.content:
            if isinstance(m.content, str):
                return m.content
            if isinstance(m.content, list):
                parts = [b.get("text", "") for b in m.content if isinstance(b, dict) and b.get("type") == "text"]
                if parts:
                    return "\n".join(parts)
            return str(m.content)
    return current_output
