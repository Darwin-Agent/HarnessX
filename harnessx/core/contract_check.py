# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Dev-time validation that a processor honors the hook-mutation contract.

The runtime enforces these rules in ``ProcessorChain`` via
``_validate_messages_contract``; this module reuses the same validator
against representative fixtures so authors can catch violations BEFORE
a real run surfaces them as ``CONTRACT [...]`` warnings.

Only the two hooks that may touch ``event.messages`` are checked:
``on_before_model`` and ``on_step_start``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .events import (
    BeforeModelEvent,
    Message,
    StepStartEvent,
    TaskStartEvent,
    ToolCall,
)
from .processor import (
    ContractViolationError,
    _validate_messages_contract,
    check_post_hook_invariants,
)


# ─── Representative message fixtures ─────────────────────────────────────────

_SYS = Message(role="system", content="You are a test harness.")
_USER = Message(role="user", content="test query")
_ASST_ONLY = Message(role="assistant", content="thinking text")
_ASST_TC = Message(
    role="assistant",
    content="",
    tool_calls=(ToolCall(id="tc_1", name="Read", input={"path": "/x"}),),
)
_TOOL_RES = Message(role="tool", content="tool output", tool_call_id="tc_1")

BEFORE_MODEL_FIXTURES: dict[str, tuple[Message, ...]] = {
    "tail_user": (_SYS, _USER),
    "tail_assistant": (_SYS, _USER, _ASST_ONLY),
    "tail_tool": (_SYS, _USER, _ASST_TC, _TOOL_RES),
}

STEP_START_FIXTURES: dict[str, tuple[Message, ...]] = {
    "with_tool": (_SYS, _USER, _ASST_TC, _TOOL_RES),
    "just_user": (_SYS, _USER),
}

# Probe both "early" and "near-budget" zones so step-dependent branches fire.
_DEFAULT_STEP_IDS: tuple[int, ...] = (5, 49)


@dataclass(frozen=True)
class ContractViolation:
    processor: str
    hook: str
    fixture: str
    step_id: int
    violation_type: str
    message: str


class _StrictMode:
    """Context manager forcing HARNESSX_CONTRACT_MODE=strict inside the block."""

    def __enter__(self) -> "_StrictMode":
        self._prev = os.environ.get("HARNESSX_CONTRACT_MODE")
        os.environ["HARNESSX_CONTRACT_MODE"] = "strict"
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._prev is None:
            os.environ.pop("HARNESSX_CONTRACT_MODE", None)
        else:
            os.environ["HARNESSX_CONTRACT_MODE"] = self._prev
        return False


def _is_default_noop(method: Any) -> bool:
    """True when a hook method is the inherited no-op from MultiHookProcessor."""
    if method is None or not callable(method):
        return True
    qual = getattr(method, "__qualname__", "") or ""
    return qual.startswith("MultiHookProcessor.")


async def _prime_task_start(processor: Any) -> None:
    """Fire on_task_start once so processors relying on it can populate state."""
    method = getattr(processor, "on_task_start", None)
    if _is_default_noop(method):
        return
    ev = TaskStartEvent(run_id="contract_check", step_id=0)
    try:
        async for _ in method(ev):
            break
    except Exception:
        # Priming is best-effort — some processors need a real state object.
        pass


async def _run_hook_primary(processor: Any, hook_name: str, event: Any) -> Any:
    """Run a hook and return the first yielded event of the same type."""
    method = getattr(processor, hook_name, None)
    if method is None:
        return None
    primary: Any = None
    async for out in method(event):
        if isinstance(out, type(event)) and primary is None:
            primary = out
    return primary


async def _check_one_hook(
    processor: Any,
    *,
    hook: str,
    hook_method_name: str,
    fixtures: dict[str, tuple[Message, ...]],
    step_ids: tuple[int, ...],
    event_factory,
) -> list[ContractViolation]:
    method = getattr(processor, hook_method_name, None)
    if _is_default_noop(method):
        return []

    await _prime_task_start(processor)

    violations: list[ContractViolation] = []
    proc_name = type(processor).__name__

    for fname, msgs in fixtures.items():
        for step_id in step_ids:
            event = event_factory(msgs, step_id)
            try:
                with _StrictMode():
                    after = await _run_hook_primary(processor, hook_method_name, event)
                    after_msgs = tuple(after.messages) if after is not None else msgs
                    _validate_messages_contract(
                        hook,
                        msgs,
                        after_msgs,
                        processor_name=proc_name,
                        step_id=step_id,
                    )
                    check_post_hook_invariants(
                        hook,
                        msgs,
                        after_msgs,
                        step_id=step_id,
                    )
            except ContractViolationError as exc:
                violations.append(
                    ContractViolation(
                        processor=proc_name,
                        hook=hook,
                        fixture=fname,
                        step_id=step_id,
                        violation_type=exc.violation_type or "unknown",
                        message=str(exc),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                violations.append(
                    ContractViolation(
                        processor=proc_name,
                        hook=hook,
                        fixture=fname,
                        step_id=step_id,
                        violation_type="runtime_error",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
    return violations


async def check_before_model_contract(
    processor: Any,
    *,
    fixtures: dict[str, tuple[Message, ...]] | None = None,
    step_ids: tuple[int, ...] = _DEFAULT_STEP_IDS,
) -> list[ContractViolation]:
    """Return contract violations observed when running on_before_model fixtures."""
    return await _check_one_hook(
        processor,
        hook="before_model",
        hook_method_name="on_before_model",
        fixtures=fixtures if fixtures is not None else BEFORE_MODEL_FIXTURES,
        step_ids=step_ids,
        event_factory=lambda msgs, step_id: BeforeModelEvent(
            run_id="contract_check",
            step_id=step_id,
            messages=msgs,
        ),
    )


async def check_step_start_contract(
    processor: Any,
    *,
    fixtures: dict[str, tuple[Message, ...]] | None = None,
    step_ids: tuple[int, ...] = _DEFAULT_STEP_IDS,
) -> list[ContractViolation]:
    """Return contract violations observed when running on_step_start fixtures."""
    return await _check_one_hook(
        processor,
        hook="step_start",
        hook_method_name="on_step_start",
        fixtures=fixtures if fixtures is not None else STEP_START_FIXTURES,
        step_ids=step_ids,
        event_factory=lambda msgs, step_id: StepStartEvent(
            run_id="contract_check",
            step_id=step_id,
            messages=msgs,
        ),
    )


async def check_processor_contract(processor: Any) -> list[ContractViolation]:
    """Run every message-mutation contract check applicable to this processor."""
    violations: list[ContractViolation] = []
    violations.extend(await check_before_model_contract(processor))
    violations.extend(await check_step_start_contract(processor))
    return violations


__all__ = [
    "BEFORE_MODEL_FIXTURES",
    "STEP_START_FIXTURES",
    "ContractViolation",
    "check_before_model_contract",
    "check_processor_contract",
    "check_step_start_contract",
]
