# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable

from .events import (
    BeforeModelEvent,
    Event,
    ModelResponseEvent,
    ProcessorTriggerEvent,
    StepEndEvent,
    StepStartEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)

_logger = logging.getLogger(__name__)
_contract_logger = logging.getLogger("harnessx.contract")

# ─── Hook Contract Enforcement ────────────────────────────────────────────────


class ContractViolationError(Exception):
    """Raised when a processor violates the hook-contract spec.

    Attributes:
        processor:      Class name of the violating processor.
        hook:           Hook where the violation occurred.
        violation_type: One of the _VT_* constants below.
        step_id:        Step at which the violation occurred.
    """

    def __init__(
        self,
        message: str,
        *,
        processor: str = "",
        hook: str = "",
        violation_type: str = "",
        step_id: int = -1,
    ) -> None:
        super().__init__(message)
        self.processor = processor
        self.hook = hook
        self.violation_type = violation_type
        self.step_id = step_id


# Canonical violation type labels
_VT_WINDOW_OUT_OF_SCOPE = "window_out_of_scope"
_VT_FORBIDDEN_ROLE_MUTATION = "forbidden_role_mutation"
_VT_CROSS_TOOL_MUTATION = "cross_tool_mutation"
_VT_LENGTH_EXCEEDED = "length_exceeded"
_VT_EMPTY_MESSAGES = "empty_messages"
_VT_SYSTEM_MUTATED = "system_mutated"
_VT_LAST_MSG_ONLY = "last_msg_only_mutated"


def _contract_mode() -> str:
    """Return ``'strict'`` or ``'warn'`` based on ``HARNESSX_CONTRACT_MODE`` env var.

    ``strict``: violations raise :class:`ContractViolationError` (阶段 B).
    ``warn``  : violations are logged as warnings only (阶段 A, default).
    """
    return os.environ.get("HARNESSX_CONTRACT_MODE", "warn").lower()


def _handle_violation(
    msg: str,
    *,
    processor: str = "",
    hook: str = "",
    violation_type: str = "",
    step_id: int = -1,
) -> None:
    """Dispatch a contract violation: warn or raise depending on CONTRACT_MODE."""
    if _contract_mode() == "strict":
        raise ContractViolationError(
            msg,
            processor=processor,
            hook=hook,
            violation_type=violation_type,
            step_id=step_id,
        )
    _contract_logger.warning(
        "CONTRACT [%s] hook=%s processor=%s step=%d: %s",
        violation_type,
        hook,
        processor,
        step_id,
        msg,
    )


def _get_event_messages(event: "Event | None") -> "tuple | None":
    """Extract the ``messages`` tuple from events that carry one, else ``None``."""
    if event is None:
        return None
    msgs = getattr(event, "messages", None)
    return tuple(msgs) if msgs is not None else None


def _validate_messages_contract(
    hook: str,
    before_msgs: "tuple",
    after_msgs: "tuple",
    processor_name: str = "",
    step_id: int = -1,
    *,
    chain_user_additions: int = 0,
) -> int:
    """Validate per-processor message mutation against hook contract rules.

    Returns the updated ``chain_user_additions`` counter (only meaningful for
    ``before_model``; callers should pass the returned value into the next call
    so cross-processor constraints are cumulative within the chain).
    """
    if hook == "step_start":
        # system_window must not change
        before_sys = (before_msgs[0],) if before_msgs and before_msgs[0].role == "system" else ()
        after_sys = (after_msgs[0],) if after_msgs and after_msgs[0].role == "system" else ()
        if before_sys and after_sys and before_sys != after_sys:
            _handle_violation(
                "step_start processor modified system_window",
                processor=processor_name,
                hook=hook,
                violation_type=_VT_SYSTEM_MUTATED,
                step_id=step_id,
            )
        # Cannot ONLY modify the last message without structural history change
        if (
            before_msgs
            and after_msgs
            and len(before_msgs) == len(after_msgs)
            and before_msgs[:-1] == after_msgs[:-1]
            and before_msgs[-1] != after_msgs[-1]
        ):
            _handle_violation(
                "step_start processor modified only last message without any structural change to history",
                processor=processor_name,
                hook=hook,
                violation_type=_VT_LAST_MSG_ONLY,
                step_id=step_id,
            )

    elif hook == "before_model":
        if not after_msgs:
            _handle_violation(
                "before_model: messages became empty after processor — fail fast",
                processor=processor_name,
                hook=hook,
                violation_type=_VT_EMPTY_MESSAGES,
                step_id=step_id,
            )
            return chain_user_additions

        len_delta = len(after_msgs) - len(before_msgs)

        if len_delta < 0:
            _handle_violation(
                f"before_model: processor removed {-len_delta} message(s)",
                processor=processor_name,
                hook=hook,
                violation_type=_VT_LENGTH_EXCEEDED,
                step_id=step_id,
            )
        elif len_delta > 1:
            _handle_violation(
                f"before_model: processor added {len_delta} messages in a single step (max +1)",
                processor=processor_name,
                hook=hook,
                violation_type=_VT_LENGTH_EXCEEDED,
                step_id=step_id,
            )
        elif len_delta == 1:
            # Adding a message is only allowed when the original last role is NOT user
            if before_msgs and before_msgs[-1].role == "user":
                _handle_violation(
                    "before_model: processor added a message when last role is already 'user' "
                    "(only content modification of the last user is allowed in that case)",
                    processor=processor_name,
                    hook=hook,
                    violation_type=_VT_LENGTH_EXCEEDED,
                    step_id=step_id,
                )
            # Cross-processor constraint: only one insertion per chain
            if chain_user_additions >= 1:
                _handle_violation(
                    "before_model: second processor attempted to add user message "
                    "(chain already has +1 user insertion)",
                    processor=processor_name,
                    hook=hook,
                    violation_type=_VT_LENGTH_EXCEEDED,
                    step_id=step_id,
                )
            new_msg = after_msgs[-1]
            if new_msg.role != "user":
                _handle_violation(
                    f"before_model: new tail message has role={new_msg.role!r} (must be 'user')",
                    processor=processor_name,
                    hook=hook,
                    violation_type=_VT_FORBIDDEN_ROLE_MUTATION,
                    step_id=step_id,
                )
            chain_user_additions += 1
        else:  # len_delta == 0
            if before_msgs and after_msgs:
                if before_msgs[-1].role == "user":
                    # Only the last user content may change; history must be untouched
                    if before_msgs[:-1] != after_msgs[:-1]:
                        _handle_violation(
                            "before_model: processor modified messages other than the last user",
                            processor=processor_name,
                            hook=hook,
                            violation_type=_VT_WINDOW_OUT_OF_SCOPE,
                            step_id=step_id,
                        )
                else:
                    # last is not user and no length change: no content change allowed
                    if before_msgs != after_msgs:
                        _handle_violation(
                            "before_model: processor modified messages when last role is not "
                            "'user' and no user was appended",
                            processor=processor_name,
                            hook=hook,
                            violation_type=_VT_WINDOW_OUT_OF_SCOPE,
                            step_id=step_id,
                        )

    # after_model / before_tool / after_tool / step_end / task_end carry no
    # ``messages`` field — _get_event_messages returns None so this function is
    # never called for those hooks. Structural enforcement suffices.

    return chain_user_additions


def check_post_hook_invariants(
    hook: str,
    initial_msgs: "tuple",
    final_msgs: "tuple",
    *,
    step_id: int = -1,
    raw_msgs: "tuple | None" = None,
) -> None:
    """Check all post-hook-chain invariants (草案 §Post-hook 不变量).

    Called once after the **entire** hook-chain completes.  Enforces structural
    guarantees that must hold regardless of how many processors ran.

    Invariants checked:
    1. system count ≤ 1 and at position 0 when present.
    2. ``len(raw_track) == len(effective_track)`` (when ``raw_msgs`` provided).
    3. Same-index role match between raw and effective tracks.
    4. Same-index ``tool_call_id`` for ``role=tool`` messages.
    5. ``before_model`` net length change only 0 or +1.
    6. Non-authorized hooks must have net length change == 0.
    7. boundary consistency is handled by RunLoop (hash-based); not rechecked here.
    """
    # Invariant 1: system constraint (only meaningful when final_msgs non-empty)
    sys_count = sum(1 for m in final_msgs if m.role == "system") if final_msgs else 0
    if sys_count > 1:
        _handle_violation(
            f"post-hook: {sys_count} system messages in final messages (max 1 allowed)",
            hook=hook,
            violation_type="invariant_system_count",
            step_id=step_id,
        )
    if sys_count == 1 and final_msgs and final_msgs[0].role != "system":
        _handle_violation(
            "post-hook: system message exists but is not at position 0",
            hook=hook,
            violation_type="invariant_system_position",
            step_id=step_id,
        )

    # Invariants 2-4: raw/effective track alignment (when raw provided)
    if raw_msgs is not None:
        if len(raw_msgs) != len(final_msgs):
            _handle_violation(
                f"post-hook: raw_track len={len(raw_msgs)} != effective_track len={len(final_msgs)}",
                hook=hook,
                violation_type="invariant_track_length",
                step_id=step_id,
            )
        else:
            for i, (r, e) in enumerate(zip(raw_msgs, final_msgs)):
                if r.role != e.role:
                    _handle_violation(
                        f"post-hook: role mismatch at index {i}: raw={r.role!r} eff={e.role!r}",
                        hook=hook,
                        violation_type="invariant_role_mismatch",
                        step_id=step_id,
                    )
                    break
                if r.role == "tool" and r.tool_call_id != e.tool_call_id:
                    _handle_violation(
                        f"post-hook: tool_call_id mismatch at index {i}: raw={r.tool_call_id!r} eff={e.tool_call_id!r}",
                        hook=hook,
                        violation_type="invariant_tool_id_mismatch",
                        step_id=step_id,
                    )
                    break

    if initial_msgs is None:
        return
    net = len(final_msgs) - len(initial_msgs)

    # Invariant 5: before_model net length 0 or +1
    if hook == "before_model" and net not in (0, 1):
        _handle_violation(
            f"post-hook: before_model chain net length change is {net:+d} (must be 0 or +1)",
            hook=hook,
            violation_type=_VT_LENGTH_EXCEEDED,
            step_id=step_id,
        )

    # Invariant 6: non-authorized hooks must not change messages length
    _NO_LEN_CHANGE = {"after_model", "before_tool", "after_tool", "step_end", "task_end", "task_start"}
    if hook in _NO_LEN_CHANGE and net != 0:
        _handle_violation(
            f"post-hook: {hook} chain changed messages length by {net:+d} (must be 0)",
            hook=hook,
            violation_type=_VT_LENGTH_EXCEEDED,
            step_id=step_id,
        )


# Named ordering phases for MultiHookProcessor._order.
# Use these instead of magic integers; fine-grained values are still accepted
# (e.g. PRE + 5 runs just after PRE processors).
PRE = 0  # Interceptors, guards, setup — run before main logic
NORMAL = 50  # Main processing logic (default)
POST = 100  # Cleanup, export, finalization


@runtime_checkable
class Processor(Protocol):
    """
    Consumes an Event, can:
    - yield the same event (pass-through)
    - yield a modified event (transform)
    - yield multiple events (split, for multi-agent fork)
    - yield nothing (intercept/block)
    - raise an exception (interrupt)
    """

    async def process(self, event: Event) -> AsyncIterator[Event]: ...


def _diff_primary(
    prev_events: "list[Event]",
    curr_events: "list[Event]",
    event_type: type,
) -> "str | None":
    """Compare the primary (same-type-as-input) event before/after a processor.

    Returns ``"intervention"`` when the primary event changed, ``None`` otherwise.
    """
    prev = next((e for e in reversed(prev_events) if isinstance(e, event_type)), None)
    curr = next((e for e in reversed(curr_events) if isinstance(e, event_type)), None)
    if prev is None or curr is None or prev is curr or prev == curr:
        return None
    return "intervention"


class ProcessorChain:
    """
    Sequentially composes multiple Processors into one.

    ProcessorChain satisfies the Processor protocol, so chains can be nested:

        inner = ProcessorChain(proc_a, proc_b)
        outer = ProcessorChain(inner, proc_c)

    Each event passes through proc_a → proc_b → proc_c.
    If any processor yields nothing, the pipeline is cut short.

    When *tracer* is supplied:
    - ``tracer.on_raw_event(event)`` is called **before** the chain runs, so
      the journal can record the pre-processor (raw) event content.
    - After each individual processor, if the primary event changed, a
      ``ProcessorTriggerEvent`` is emitted to the tracer automatically.
      Processors no longer need to manually ``yield ProcessorTriggerEvent``.
    *hook* is the logical hook name (e.g. ``"step_start"``, ``"after_model"``)
    that will be attached to any auto-generated trigger events.
    """

    def __init__(self, *processors: Processor):
        self.processors = list(processors)

    def add(self, processor: Processor) -> "ProcessorChain":
        self.processors.append(processor)
        return self

    async def process(  # type: ignore[override]
        self,
        event: Event,
        tracer: "object | None" = None,
        hook: str = "",
    ) -> AsyncIterator[Event]:
        if tracer is not None:
            _raw_fn = getattr(tracer, "on_raw_event", None)
            if _raw_fn is not None:
                await _raw_fn(event)

        _do_validate = bool(hook)
        initial_msgs = _get_event_messages(event) if _do_validate else None

        # Fail-fast pre-condition: before_model and step_start must not receive empty messages
        if hook in ("before_model", "step_start") and initial_msgs is not None and len(initial_msgs) == 0:
            _handle_violation(
                f"{hook}: received empty messages — fail fast",
                hook=hook,
                violation_type=_VT_EMPTY_MESSAGES,
                step_id=getattr(event, "step_id", -1),
            )

        events: list[Event] = [event]
        event_type = type(event)
        chain_user_additions = 0  # tracks user msg insertions in before_model chain

        for processor in self.processors:
            prev = events[:]
            prev_primary = next((e for e in reversed(prev) if isinstance(e, event_type)), None)
            prev_msgs = _get_event_messages(prev_primary) if _do_validate else None

            next_events: list[Event] = []
            for ev in events:
                async for out in processor.process(ev):
                    next_events.append(out)
            events = next_events
            if not events:
                return

            # Per-processor contract validation
            if _do_validate and prev_msgs is not None:
                curr_primary = next((e for e in reversed(events) if isinstance(e, event_type)), None)
                curr_msgs = _get_event_messages(curr_primary)
                if curr_msgs is not None:
                    chain_user_additions = _validate_messages_contract(
                        hook,
                        prev_msgs,
                        curr_msgs,
                        processor_name=type(processor).__name__,
                        step_id=getattr(event, "step_id", -1),
                        chain_user_additions=chain_user_additions,
                    )

            if tracer is not None:
                action = _diff_primary(prev, events, event_type)
                if action:
                    trigger = ProcessorTriggerEvent(
                        run_id=event.run_id,
                        step_id=event.step_id,
                        processor=type(processor).__name__,
                        hook=hook,
                        action=action,
                        detail={},
                    )
                    await tracer.on_event(trigger)  # type: ignore[attr-defined]

        # Chain-level post-hook invariant check
        if _do_validate and initial_msgs is not None:
            final_primary = next((e for e in reversed(events) if isinstance(e, event_type)), None)
            final_msgs = _get_event_messages(final_primary)
            if final_msgs is not None:
                check_post_hook_invariants(
                    hook,
                    initial_msgs,
                    final_msgs,
                    step_id=getattr(event, "step_id", -1),
                )

        for ev in events:
            yield ev


def on(event_type: type) -> Callable:
    """Method decorator for ``MultiHookProcessor`` subclasses.

    Marks a method as the handler for *event_type*, auto-registering it in the
    class's ``_DISPATCH`` table via ``__init_subclass__``.  This lets you name
    handler methods freely instead of being forced to override ``on_before_model``
    etc.

    Example::

        class CostGuard(MultiHookProcessor):
            @on(BeforeModelEvent)
            async def check(self, event: BeforeModelEvent):
                if self._spent > self.limit:
                    raise BudgetExceededError()
                yield event

    For processors that handle multiple event types, stack decorators or use one
    method per event type — either way, only one handler fires per event.
    """

    def decorator(fn: Callable) -> Callable:
        fn._on_event_type = event_type
        return fn

    return decorator


class MultiHookProcessor:
    """Recommended base class for implementing harness processors.

    **Why subclass this instead of implementing ``Processor`` directly?**

    - Override only the ``on_*`` methods you need; all others are no-ops.
    - Use ``@on(EventClass)`` to name handler methods freely (see module docs).
    - ``HarnessBuilder.add()`` auto-registers subclasses under ``"*"`` with no
      extra configuration — just set ``_singleton_group`` and ``_order``.

    Internally, ``process()`` dispatches to the right ``on_*`` method via
    ``_DISPATCH`` (a ``{EventType: method_name}`` table built at class creation).
    Subclasses that use ``@on()`` have their entries merged into ``_DISPATCH``
    automatically by ``__init_subclass__``.

    Example — two hooks, one class::

        class AuditProcessor(MultiHookProcessor):
            _singleton_group = "audit"
            _order = 50

            def __init__(self):
                self._log: list[dict] = []

            async def on_step_end(self, event: StepEndEvent):
                self._log.append({"step": event.step_id, "cost": event.cumulative_cost_usd})
                yield event

            async def on_task_end(self, event: TaskEndEvent):
                print(f"Audit: {len(self._log)} steps, "
                      f"${event.total_cost_usd:.4f} total")
                yield event

    Example — free method name via ``@on``::

        class CostGuard(MultiHookProcessor):
            _singleton_group = "cost_guard"
            _order = 10

            @on(BeforeModelEvent)
            async def check(self, event: BeforeModelEvent):
                if event.cumulative_cost_usd >= self.limit:
                    raise BudgetExceededError()
                yield event
    """

    # Dispatch table: event type → method name.
    # Subclasses that use @on() have their entries merged in by __init_subclass__.
    _DISPATCH: dict[type, str] = {
        TaskStartEvent: "on_task_start",
        StepStartEvent: "on_step_start",
        BeforeModelEvent: "on_before_model",
        ModelResponseEvent: "on_after_model",
        ToolCallEvent: "on_before_tool",
        ToolResultEvent: "on_after_tool",
        StepEndEvent: "on_step_end",
        TaskEndEvent: "on_task_end",
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Merge @on() decorated methods into this subclass's _DISPATCH table.

        Also validates that any method named ``on_*`` is either a recognised
        hook name or decorated with ``@on(EventClass)``.  This catches typos
        like ``on_tool_call`` (should be ``on_before_tool``) at import time
        rather than silently never being called.
        """
        super().__init_subclass__(**kwargs)
        dispatch: dict[type, str] = dict(cls._DISPATCH)  # inherit parent entries

        # First pass: register @on()-decorated methods (they may use any name).
        for name in vars(cls):
            method = vars(cls)[name]  # use vars() to avoid MRO lookup
            if callable(method) and hasattr(method, "_on_event_type"):
                dispatch[method._on_event_type] = name

        # Second pass: flag on_* methods that aren't valid hooks and aren't @on()-decorated.
        valid_hook_names = set(dispatch.values())
        for name in vars(cls):
            method = vars(cls)[name]
            if (
                callable(method)
                and name.startswith("on_")
                and name not in valid_hook_names
                and not hasattr(method, "_on_event_type")
            ):
                raise TypeError(
                    f"{cls.__qualname__}.{name}: unrecognised hook method name. "
                    f"Valid on_* names: {sorted(valid_hook_names)}. "
                    f"Use @on(EventClass) to handle a custom event type under a free name."
                )

        cls._DISPATCH = dispatch

    # Ordering hints — set on subclasses, read by HarnessBuilder.
    # _order:  use PRE / NORMAL / POST constants (or fine-grained int offsets).
    # _after:  list of _singleton_group names that must run before this processor
    #          within the same hook.  Soft: unregistered groups are silently ignored.
    _after: list[str] = []

    # Sub-harness registry: populated at Harness.__init__() time via _bind_sub_harnesses().
    # Each entry is a minimal Harness configured for that provider key.
    _sub_harnesses: dict[str, Any] = {}

    # ModelConfig reference: populated at Harness.__init__() time via _bind_model_config().
    _model_config: Any = None

    # HarnessConfig reference: populated at Harness.__init__() time via _bind_harness_config().
    _harness_config: Any = None

    def _bind_model_config(self, model_config: Any) -> None:
        """Called by Harness.__init__() to inject the parent ModelConfig.

        Processors that need parent model context use this to access the
        current ``ModelConfig``. The default implementation stores it on
        the instance.
        """
        self._model_config = model_config

    def _bind_harness_config(self, harness_config: Any) -> None:
        """Called by Harness.__init__() to inject the parent HarnessConfig.

        Processors that need harness context use this to access the current
        harness configuration (workspace, tool_registry, tracer). The default
        implementation stores it on the instance.
        """
        self._harness_config = harness_config

    def _bind_runtime(self, rt: Any) -> None:
        """Called by Harness.__init__() to inject the live _HarnessRuntime.

        Processors that need to manipulate the active processor set (e.g.
        LightMeta's plan processor) use this to access _rt.processors — the
        hook-keyed dict of instantiated Processor objects used by the runloop.
        """
        self._harness_runtime = rt

    def _bind_sub_harnesses(self, sub_harnesses: dict[str, Any]) -> None:
        """Called by Harness.__init__() to inject minimal sub-harnesses keyed by provider role.

        Processors that need a secondary model call ``sub_harness.run()`` instead
        of ``provider.complete()`` — all model calls go through the RunLoop, so
        cost, trace, and trajectory are captured automatically.

        ``parent_run_id`` is threaded via the event's ``run_id`` at call time::

            sub = self._sub_harnesses.get(self._judge_key)
            if sub is not None:
                result = await sub.run(BaseTask(description=prompt, max_steps=1),
                                       parent_run_id=event.run_id)
        """
        self._sub_harnesses = dict(sub_harnesses)

    def _bind_tool_registry(self, tool_registry: Any) -> None:
        """Called by Harness.__init__() to give processors access to the registered tool names.

        Override in subclasses that need to introspect available tools — for example to
        build a case-insensitive name lookup table.  The default implementation is a no-op.
        """

    async def process(self, event: Event) -> AsyncIterator[Event]:
        method_name = self._DISPATCH.get(type(event))
        if not method_name:
            yield event
            return
        handler = getattr(self, method_name, None)
        if not handler:
            yield event
            return
        try:
            async for out in handler(event):
                yield out
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Control-flow exceptions (budget/loop/context/parse signals) must
            # propagate so run_loop can translate them into exit_reason.  Any
            # other exception is a bug inside the processor — we log identity
            # (processor class + hook) and pass the event through unchanged so
            # one buggy processor cannot crash the run.  This is the backstop
            # for generated meta-skill processors whose runtime bugs escape
            # smoke_load + admission probe.
            from .runloop import HarnessError

            if isinstance(exc, (HarnessError, ContractViolationError)):
                raise
            exc_name = type(exc).__name__
            if exc_name in (
                "ToolFailureLimitError",
                "ContextLengthExceeded",
                "GenerationAborted",
            ):
                raise
            _logger.warning(
                "processor crashed, skipping: processor=%s hook=%s event=%s error=%s: %s",
                type(self).__name__,
                method_name,
                type(event).__name__,
                exc_name,
                exc,
            )
            _logger.debug("processor crash traceback:", exc_info=True)
            yield event

    # Default no-op handlers — override as needed

    async def on_task_start(self, event: TaskStartEvent) -> AsyncIterator[Event]:
        """Once per task. Allowed: modify ``system_prompt``. No ``messages`` field on this event."""
        yield event

    async def on_step_start(self, event: StepStartEvent) -> AsyncIterator[Event]:
        """Start of each step. Structural history edits OK (system stays at [0]). Forbidden: last-user-only substitution without a structural change."""
        yield event

    async def on_before_model(self, event: BeforeModelEvent) -> AsyncIterator[Event]:
        """Just before the LLM call. last=user → may edit last user content only; last≠user → may append exactly +1 user. History window is frozen."""
        yield event

    async def on_after_model(self, event: ModelResponseEvent) -> AsyncIterator[Event]:
        """After model responds. Allowed: modify ``content`` / ``tool_calls``. ``ModelResponseEvent`` carries no ``messages`` field — history is structurally inaccessible."""
        yield event

    async def on_before_tool(self, event: ToolCallEvent) -> AsyncIterator[Event]:
        """Before a tool executes. Allowed: modify ``tool_input``, set ``approved=False``. ``ToolCallEvent`` carries no ``messages`` field — history is structurally inaccessible."""
        yield event

    async def on_after_tool(self, event: ToolResultEvent) -> AsyncIterator[Event]:
        """After a tool returns. Allowed: modify ``result``. ``ToolResultEvent`` carries no ``messages`` field — history is structurally inaccessible."""
        yield event

    async def on_step_end(self, event: StepEndEvent) -> AsyncIterator[Event]:
        """Step complete. Read-only pass-through; no ``messages`` field. Aggregate cost/token data available."""
        yield event

    async def on_task_end(self, event: TaskEndEvent) -> AsyncIterator[Event]:
        """Task complete. Read-only pass-through; no ``messages`` field. Final cost/exit info available."""
        yield event


async def pipe(
    event: Event,
    processors: "list[Processor]",
    tracer: "object | None" = None,
    hook: str = "",
) -> "Event | None":
    """Pass event through processors sequentially.  Returns the last yielded event, or
    ``None`` if any processor intercepted (yielded nothing).

    Use ``pipe_all`` when the processor chain may yield heterogeneous event types
    (e.g. ``ModelResponseEvent`` + ``SpawnSubAgentEvent`` from ``after_model``).
    """
    chain = ProcessorChain(*processors)
    result = None
    async for ev in chain.process(event, tracer=tracer, hook=hook):
        result = ev
    return result


async def pipe_all(
    event: Event,
    processors: "list[Processor]",
    tracer: "object | None" = None,
    hook: str = "",
) -> "list[Event]":
    """Pass *event* through processors; return **every** yielded event.

    Unlike ``pipe()``, no intermediate events are discarded.  Necessary when a
    hook point may produce heterogeneous output — for example when a custom
    ``after_model`` processor yields a ``ModelResponseEvent`` plus one or more
    ``SpawnSubAgentEvent`` values.

    Returns ``[event]`` when *processors* is empty (pass-through).
    """
    chain = ProcessorChain(*processors)
    return [ev async for ev in chain.process(event, tracer=tracer, hook=hook)]


# ─── Hook Decorators (syntax sugar for single-node Processors) ─────────────────


def _make_hook_processor(fn: Callable, event_type: type) -> Processor:
    """Wrap an async generator function into a Processor object."""

    class HookProcessor:
        async def process(self, event: Event) -> AsyncIterator[Event]:
            if isinstance(event, event_type):
                async for out in fn(event):
                    yield out
            else:
                yield event

    HookProcessor.__name__ = getattr(fn, "__name__", "HookProcessor")
    return HookProcessor()


def task_start(fn: Callable) -> Processor:
    """Decorator: wrap async generator into TaskStartEvent processor."""
    return _make_hook_processor(fn, TaskStartEvent)


def step_start(fn: Callable) -> Processor:
    """Decorator: wrap async generator into StepStartEvent processor."""
    return _make_hook_processor(fn, StepStartEvent)


def before_model(fn: Callable) -> Processor:
    """Decorator: wrap async generator into BeforeModelEvent processor."""
    return _make_hook_processor(fn, BeforeModelEvent)


def after_model(fn: Callable) -> Processor:
    """Decorator: wrap async generator into ModelResponseEvent processor."""
    return _make_hook_processor(fn, ModelResponseEvent)


def before_tool(fn: Callable) -> Processor:
    """Decorator: wrap async generator into ToolCallEvent processor."""
    return _make_hook_processor(fn, ToolCallEvent)


def after_tool(fn: Callable) -> Processor:
    """Decorator: wrap async generator into ToolResultEvent processor."""
    return _make_hook_processor(fn, ToolResultEvent)


def on_step_end(fn: Callable) -> Processor:
    """Decorator: wrap async generator into StepEndEvent processor."""
    return _make_hook_processor(fn, StepEndEvent)


def on_task_end(fn: Callable) -> Processor:
    """Decorator: wrap async generator into TaskEndEvent processor."""
    return _make_hook_processor(fn, TaskEndEvent)
