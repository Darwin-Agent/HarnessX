# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from .events import Message, ToolResultEvent, dict_to_message, message_to_dict


_JSON_PRIMITIVES = (bool, int, float, str, bytes, type(None))


def _serialize_slot_content(val: Any) -> Any:
    """Serialize a StateSlot content value to a JSON-safe form.

    - JSON primitives / list / dict → returned as-is
    - set / frozenset → sorted list
    - Python object with __module__ → {"_target_": "module.Class", ...params}
    - Completely unserializable → {"_unserializable_": "ClassName"} (+ one warning)
    """
    if isinstance(val, _JSON_PRIMITIVES):
        return val
    if isinstance(val, (set, frozenset)):
        try:
            return sorted(val)
        except TypeError:
            return list(val)
    if isinstance(val, (list, tuple)):
        return [_serialize_slot_content(item) for item in val]
    if isinstance(val, dict):
        return {k: _serialize_slot_content(v) for k, v in val.items()}
    # Python object: attempt _target_ serialization (Hydra-style)
    cls = type(val)
    module = getattr(cls, "__module__", None)
    if module and not module.startswith("_"):
        import inspect as _inspect

        target = f"{module}.{cls.__qualname__}"
        result: dict = {"_target_": target}
        try:
            sig = _inspect.signature(cls.__init__)
            for pname, _ in sig.parameters.items():
                if pname == "self":
                    continue
                attr = getattr(val, pname, getattr(val, f"_{pname}", None))
                if attr is None:
                    continue
                if isinstance(attr, _JSON_PRIMITIVES):
                    result[pname] = attr
                elif isinstance(attr, (list, tuple, set, frozenset, dict)):
                    result[pname] = _serialize_slot_content(attr)
        except (ValueError, TypeError):
            pass
        return result
    # Last resort: preserve key, mark as unserializable
    warnings.warn(
        f"StateSlot content of type {type(val).__name__!r} cannot be serialized; "
        "it will be restored as None on wake(). "
        "Store only JSON-safe data or _target_-compatible objects in state slots.",
        stacklevel=4,
    )
    return {"_unserializable_": type(val).__name__}


def _deserialize_slot_content(val: Any) -> Any:
    """Inverse of _serialize_slot_content."""
    if not isinstance(val, dict):
        return val
    if "_unserializable_" in val:
        warnings.warn(
            f"StateSlot content was marked _unserializable_ ({val['_unserializable_']!r}); restoring as None.",
            stacklevel=4,
        )
        return None
    if "_target_" in val:
        target = val["_target_"]
        params = {k: v for k, v in val.items() if k != "_target_"}
        try:
            module_path, _, cls_name = target.rpartition(".")
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            return cls(**params)
        except Exception:
            # Construction failed — return the raw dict so data isn't lost
            return val
    return val


@dataclass
class PendingSubagent:
    """Tracks an async sub-agent that has been spawned but not yet completed.

    Persisted in ``State.pending_subagents`` so that interrupt/resume cycles
    can surface outstanding children to the model and allow it to re-spawn or
    acknowledge them.
    """

    label: str
    task: str
    run_id: str = ""  # filled once child harness starts
    model: str = ""
    system_prompt: str = ""
    tools: list = field(default_factory=list)


@dataclass
class StateSlot:
    """A dynamic key-value unit in the State."""

    slot_type: str
    content: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class State:
    """Current run state snapshot.

    Two message tracks are kept:
    - raw_messages: append-only factual stream (user / assistant / tool outputs)
    - messages: effective context stream used by the run loop (may include
      processor-injected guidance messages)
    """

    run_id: str
    parent_run_id: str | None = None
    raw_messages: list[Message] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    step: int = 0
    cumulative_tokens: int = 0
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    cumulative_cost_usd: float = 0.0
    tool_results: list[ToolResultEvent] = field(default_factory=list)
    slots: dict[str, StateSlot] = field(default_factory=dict)

    # Budget limits (set from Task). None = no limit.
    max_steps: int = 50
    token_budget: int | None = None
    max_cost_usd: float | None = None

    # Sub-agent tracking
    spawn_depth: int = 0
    """Nesting depth of this agent (0 = root, 1 = first-level sub-agent, …)."""
    pending_subagents: dict[str, PendingSubagent] = field(default_factory=dict)
    """Async sub-agents that have been spawned but not yet completed.
    Key is the label passed to spawn_subagent(label=…).
    Persisted in snapshots so interrupt/resume can surface outstanding children.
    """

    last_sys_prompt_hash: str | None = None
    """SHA-256 hex of the system_prompt seen at the last TaskStartEvent.
    Used by RunLoop to detect cross-task system prompt changes within a session
    and generate a SegmentBoundaryEvent(reason="system_prompt_change").
    None on first run or when loaded from a pre-v3 snapshot."""

    def budget_exceeded(self) -> bool:
        if self.step >= self.max_steps:
            return True
        if self.token_budget is not None and self.cumulative_tokens >= self.token_budget:
            return True
        if self.max_cost_usd is not None and self.cumulative_cost_usd >= self.max_cost_usd:
            return True
        return False

    def __post_init__(self) -> None:
        # Backward-compatible initialisation:
        # - older callers may populate only messages
        # - newer callers may populate only raw_messages
        if not self.raw_messages and self.messages:
            self.raw_messages = list(self.messages)
        if not self.messages and self.raw_messages:
            self.messages = list(self.raw_messages)

    def add_raw_message(self, message: Message) -> None:
        """Append a factual conversation message.

        Both ``raw_messages`` and ``messages`` are updated, preserving the
        invariant ``len(raw_messages) == len(messages)``.
        """
        self.raw_messages.append(message)
        self.messages.append(message)

    def add_message(self, message: Message) -> None:
        # Backward compatibility: historical call sites treated add_message as
        # "append conversation fact". Keep that behavior.
        self.add_raw_message(message)

    def add_tool_result(self, event: ToolResultEvent) -> None:
        self.tool_results.append(event)

    def set_slot(self, key: str, slot_type: str, content: Any, metadata: dict | None = None) -> None:
        self.slots[key] = StateSlot(slot_type=slot_type, content=content, metadata=metadata or {})

    def get_slot(self, key: str) -> StateSlot | None:
        return self.slots.get(key)

    def delete_slot(self, key: str) -> None:
        self.slots.pop(key, None)

    def snapshot(self) -> dict:
        """Return a serializable snapshot of state for checkpointing and wake() recovery.

        Message tracks are included so crash-recovery checkpoints are
        immediately usable even when JSONL reconstruction is unavailable.
        """
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "raw_messages": [message_to_dict(m) for m in self.raw_messages],
            "messages": [message_to_dict(m) for m in self.messages],
            "step": self.step,
            "cumulative_tokens": self.cumulative_tokens,
            "cumulative_input_tokens": self.cumulative_input_tokens,
            "cumulative_output_tokens": self.cumulative_output_tokens,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "max_steps": self.max_steps,
            "token_budget": self.token_budget,
            "max_cost_usd": self.max_cost_usd,
            "slots": {
                k: {
                    "slot_type": v.slot_type,
                    "content": _serialize_slot_content(v.content),
                    "metadata": v.metadata,
                }
                for k, v in self.slots.items()
            },
            "spawn_depth": self.spawn_depth,
            "pending_subagents": {
                k: {
                    "label": v.label,
                    "task": v.task,
                    "run_id": v.run_id,
                    "model": v.model,
                    "system_prompt": v.system_prompt,
                    "tools": v.tools,
                }
                for k, v in self.pending_subagents.items()
            },
            "last_sys_prompt_hash": self.last_sys_prompt_hash,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "State":
        """Restore numeric state from a snapshot dict.

        For older snapshots that do not contain ``raw_messages``/``messages``,
        message tracks default to empty and can be rebuilt from JSONL by wake().
        """
        state = cls(run_id=data["run_id"])
        raw = data.get("raw_messages")
        msgs = data.get("messages")
        if isinstance(raw, list):
            state.raw_messages = [dict_to_message(m) if isinstance(m, dict) else m for m in raw]
        if isinstance(msgs, list):
            state.messages = [dict_to_message(m) if isinstance(m, dict) else m for m in msgs]
        if not state.raw_messages and state.messages:
            state.raw_messages = list(state.messages)
        if not state.messages and state.raw_messages:
            state.messages = list(state.raw_messages)
        state.step = data["step"]
        state.cumulative_tokens = data["cumulative_tokens"]
        state.cumulative_input_tokens = data.get("cumulative_input_tokens", 0)
        state.cumulative_output_tokens = data.get("cumulative_output_tokens", 0)
        state.cumulative_cost_usd = data["cumulative_cost_usd"]
        state.max_steps = data.get("max_steps", 50)
        state.token_budget = data.get("token_budget", None)
        state.max_cost_usd = data.get("max_cost_usd", None)
        state.slots = {
            k: StateSlot(
                slot_type=v["slot_type"],
                content=_deserialize_slot_content(v["content"]),
                metadata=v.get("metadata", {}),
            )
            for k, v in data.get("slots", {}).items()
        }
        state.spawn_depth = data.get("spawn_depth", 0)
        state.pending_subagents = {k: PendingSubagent(**v) for k, v in data.get("pending_subagents", {}).items()}
        state.last_sys_prompt_hash = data.get("last_sys_prompt_hash")
        return state
