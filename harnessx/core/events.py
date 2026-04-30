# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str | list  # str for text; list of content blocks for multimodal (Anthropic format)
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple = ()  # tuple[ToolCall, ...] — populated for role=assistant with tool calls
    thinking: str = ""  # reasoning/thinking text (for display/trajectory recording)
    thinking_blocks: tuple = ()  # tuple[dict, ...] — raw provider thinking blocks with signatures
    # Required for Anthropic multi-turn: blocks must be replayed verbatim
    # (including "signature") so the API can verify they weren't tampered with.
    msg_id: str | None = None  # stable per-message identity; set on raw-track records by journal


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: dict  # JSON Schema
    metadata: dict = field(default_factory=dict, hash=False, compare=False)
    """Arbitrary metadata attached to the tool.

    Convention — use ``"tags"`` for tag-based filtering::

        ToolSchema(name="rm", ..., metadata={"tags": ["dangerous", "write"]})

    ``TagToolFilter`` reads ``metadata["tags"]`` to allow/block tool schemas
    before they are sent to the model.
    """


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class EvalResult:
    passed: bool
    score: float  # 0.0 ~ 1.0
    reason: str
    reward: float = 0.0


# ─── Base Event ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    type: str
    run_id: str
    step_id: int
    ts: float = field(default_factory=time.time)


# ─── Concrete Events ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskStartEvent(Event):
    type: str = field(default="task_start", init=False)
    task_description: str = ""
    model: str = ""
    parent_run_id: str | None = None
    session_id: str = ""  # session identifier — for /session display and /compact state loading
    state: Any | None = field(
        default=None, hash=False, compare=False
    )  # mutable State ref — processors may read/write slots and messages
    # Fields populated/modified by on_task_start processors:
    system_prompt: str = ""  # assembled system prompt (frozen for task lifetime)
    workspace: Any | None = field(default=None, hash=False, compare=False)  # workspace passed to run_loop
    tools: tuple["ToolSchema", ...] = ()  # initial tool schemas from tool_registry


@dataclass(frozen=True)
class BoundaryHint:
    """Annotation a step_start processor sets on StepStartEvent to pass boundary
    metadata to RunLoop without emitting SegmentBoundaryEvent directly.
    RunLoop reads this when constructing the auto-generated SegmentBoundaryEvent."""

    reason: str = "compaction"
    before_msgs: int = 0
    after_msgs: int = 0
    before_tokens: int = 0
    after_tokens: int = 0


@dataclass(frozen=True)
class StepStartEvent(Event):
    type: str = field(default="step_start", init=False)
    # Input fields — set by runloop, read by context processors
    raw_messages: tuple["Message", ...] = ()  # state.raw_messages at this step
    task: Any = field(default=None, hash=False, compare=False)
    context_window: int = 64_000
    workspace: Any | None = field(default=None, hash=False, compare=False)
    tools: tuple["ToolSchema", ...] = ()  # initial schemas from tool_registry
    # Output fields — populated/modified by step_start processors
    messages: tuple["Message", ...] = ()  # assembled context
    system_prompt: str = ""
    token_count: int = 0
    boundary_hint: "BoundaryHint | None" = field(default=None, hash=False, compare=False)


@dataclass(frozen=True)
class BeforeModelEvent(Event):
    type: str = field(default="before_model", init=False)
    messages: tuple[Message, ...] = ()
    tools: tuple[ToolSchema, ...] = ()
    cumulative_cost_usd: float = 0.0  # needed by CostGuardProcessor
    skip_model: bool = field(default=False, hash=False, compare=False)
    synthetic_output: str = field(default="", hash=False, compare=False)


@dataclass(frozen=True)
class ModelResponseEvent(Event):
    type: str = field(default="model_response", init=False)
    content: str = ""
    thinking: str = ""  # extended thinking / reasoning content from the model
    thinking_blocks: tuple = ()  # raw provider blocks (with signatures) for multi-turn replay
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    attempted_models: tuple[str, ...] = ()  # models tried before this one (ProviderGroup fallback)


@dataclass(frozen=True)
class ToolCallEvent(Event):
    type: str = field(default="tool_call", init=False)
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_call_id: str = ""
    approved: bool = True
    synthetic_result: str | None = None
    """When set alongside approved=False, the RunLoop injects this as the
    ToolResultEvent result instead of sending "Tool call not approved."
    Processors can use this to intercept tool calls and inject custom results.
    """


@dataclass(frozen=True)
class ToolResultEvent(Event):
    type: str = field(default="tool_result", init=False)
    tool_name: str = ""
    tool_call_id: str = ""
    result: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    content_blocks: tuple = field(default=(), hash=False, compare=False)
    # Native multimodal blocks (Anthropic content format) from MCP tools.
    # Empty tuple = text-only result; processor chain passes this through unchanged.


@dataclass(frozen=True)
class StepEndEvent(Event):
    type: str = field(default="step_end", init=False)
    step_summary: str = ""
    tool_call_summary: str = ""  # "tool1(args)|tool2(args)" — used by LoopDetectionProcessor
    cumulative_tokens: int = 0
    cumulative_cost_usd: float = 0.0
    memory_written: bool = False
    state_snapshot: "dict | None" = field(
        default=None, hash=False, compare=False
    )  # needed by CheckpointProcessor; None by default to keep events lightweight
    # Timing + token breakdown (populated by run_loop, forwarded via SSETracer)
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class TaskEndEvent(Event):
    type: str = field(default="task_end", init=False)
    final_output: str = ""
    exit_reason: str = "done"
    total_steps: int = 0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    last_step_input_tokens: int = 0  # input_tokens of the final model call (= context size)
    last_step_output_tokens: int = 0  # output_tokens of the final model call
    eval_result: EvalResult | None = None
    success_criteria: str = ""  # from task.success_criteria — used by EvaluationProcessor
    task_description: str = ""  # from task.description — used by LLMJudgeProcessor
    final_messages: tuple = field(default_factory=tuple, hash=False, compare=False)
    # tuple[Message, ...] — full conversation at task end; used by LLMJudgeEvaluator
    error: str = ""  # exception message when exit_reason == "error"
    state_snapshot: "dict | None" = field(
        default=None, hash=False, compare=False
    )  # State.snapshot() at task end — used by HarnessJournal to write final_state.json


@dataclass(frozen=True)
class SegmentBoundaryEvent(Event):
    """Signals that the conversation history has been restructured and a new
    recording segment should begin.

    Emitted by any processor that modifies ``state.messages`` in a way that
    breaks continuity with the previous event stream — compaction, multi-agent
    handoff, strategy-based context resets, etc.

    HarnessJournal handles this event by:

    1. Writing ``{run_id}_state.json`` for the *current* segment (checkpoint).
    2. Rotating to a new ``{new_run_id}.jsonl`` + ``{new_run_id}_trace.jsonl``.
    3. Updating the session index (appending ``new_run_id``, updating
       ``latest_run_id``).

    The RunLoop does **not** restart; only the journal's file pointers change.

    Args:
        reason:         Human-readable reason for the boundary.  Conventional
                        values: ``"compaction"``, ``"agent_handoff"``.
                        Any string is valid — used for tracing / analytics only.
        new_run_id:     ID for the new recording segment.  Generated by the
                        emitting processor via ``make_run_id()``.
        state_snapshot: ``state.snapshot()`` captured *after* the restructuring.
                        Written by HarnessJournal to the checkpoint file so the
                        segment can be resumed independently.
    """

    type: str = field(default="segment_boundary", init=False)
    reason: str = "compaction"
    new_run_id: str = ""
    state_snapshot: "dict | None" = field(default=None, hash=False, compare=False)
    # Post-compaction message list — written by HarnessJournal as a
    # ``context_snapshot`` record in the NEW segment's JSONL so that
    # wake() can reconstruct State.messages from the latest JSONL alone,
    # without reading any prior segment or the state file's messages field.
    compacted_messages: "tuple[Message, ...]" = field(default=(), hash=False, compare=False)
    # When the two tracks diverge after trimming (e.g. token_budget), the raw
    # track snapshot is stored here separately.  When absent, both snapshots use
    # compacted_messages (compaction case: two tracks are identical after summary).
    compacted_raw_messages: "tuple[Message, ...]" = field(default=(), hash=False, compare=False)
    # Populated by CompactionProcessor so downstream consumers can show before/after stats
    before_msgs: int = 0
    after_msgs: int = 0
    before_tokens: int = 0
    after_tokens: int = 0


@dataclass(frozen=True)
class SpawnSubAgentEvent(Event):
    """Signal that a Processor wants to fork a sub-agent task.

    A Processor can yield this *alongside* a ``ModelResponseEvent`` from the
    ``after_model`` hook.  The RunLoop collects all ``SpawnSubAgentEvent`` s,
    emits them to the tracer, and stores them in the trajectory metadata for
    downstream orchestration.  Actual task-forking is the caller's
    responsibility (e.g. ``asyncio.create_task``).

    Example::

        class ForkingRouter(MultiHookProcessor):
            async def on_after_model(self, event: ModelResponseEvent):
                yield event  # keep original step running
                for sub_task in self._parse_sub_tasks(event):
                    yield SpawnSubAgentEvent(
                        run_id=event.run_id,
                        step_id=event.step_id,
                        sub_task=sub_task,
                    )
    """

    type: str = field(default="spawn_sub_agent", init=False)
    sub_task: Any = field(default=None, hash=False, compare=False)
    """Task-like object describing the sub-agent's work.  Shape is
    application-defined; the RunLoop treats it as opaque."""
    child_run_id: str = ""
    """Pre-generated run_id for the child agent — set by the spawner so the
    frontend can correlate child SSE events before they arrive."""


@dataclass(frozen=True)
class ProcessorTriggerEvent(Event):
    """Emitted by a processor when it performs a real intervention (not a transparent pass-through).

    Recorded in ``{run_id}_trace.jsonl`` as ``event_type: "processor_trigger"``.
    Only fired when the processor *does* something — warnings injected, parameters
    corrected, context truncated, budget halted, etc.  Pure pass-throughs are silent.

    Args:
        processor:  Class name of the triggering processor, e.g. ``"LoopDetectionProcessor"``.
        hook:       Hook method name where the trigger fired, e.g. ``"after_tool"``.
        action:     Short snake_case label describing what the processor did,
                    e.g. ``"loop_warning"``, ``"params_corrected"``, ``"context_truncated"``.
        detail:     Optional key/value metadata (tool name, counts, thresholds, …).
    """

    type: str = field(default="processor_trigger", init=False)
    processor: str = ""
    hook: str = ""
    action: str = ""
    detail: dict = field(default_factory=dict, hash=False, compare=False)


def make_run_id() -> str:
    return str(uuid.uuid4())


def make_msg_id() -> str:
    return "m-" + str(uuid.uuid4())


# ─── Message Window Helpers ───────────────────────────────────────────────────


def compute_windows(
    messages: "tuple[Message, ...] | list[Message]",
) -> "tuple[tuple[Message,...], tuple[Message,...], tuple[Message,...]]":
    """Split a message sequence into the three mutable windows defined by the
    hook-contract spec:

    Returns:
        (system_window, history_window, active_user_window)

    - ``system_window``: the leading system message if present, else empty.
    - ``history_window``:
        * last role is ``"user"``: messages between system and last-user (exclusive both ends)
        * last role is not ``"user"``: messages after system to end (inclusive)
        * no system: window starts at index 0
    - ``active_user_window``: the trailing user message, or empty tuple.
    """
    msgs: tuple[Message, ...] = tuple(messages)
    if not msgs:
        return (), (), ()

    system_window: tuple[Message, ...] = (msgs[0],) if msgs[0].role == "system" else ()
    sys_end = 1 if system_window else 0

    if msgs[-1].role == "user":
        active_user_window: tuple[Message, ...] = (msgs[-1],)
        history_window: tuple[Message, ...] = msgs[sys_end:-1]
    else:
        active_user_window = ()
        history_window = msgs[sys_end:]

    return system_window, history_window, active_user_window


def _stable_serialize_message(m: "Message") -> dict:
    """Serialize a Message to a deterministic dict for hashing (no external I/O)."""
    d: dict = {"role": m.role}

    if isinstance(m.content, list):
        d["content"] = [dict(sorted(b.items())) if isinstance(b, dict) else b for b in m.content]
    else:
        d["content"] = m.content or ""

    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    if m.name is not None:
        d["name"] = m.name
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "input": dict(sorted(tc.input.items()))} for tc in m.tool_calls
        ]
    if m.thinking_blocks:
        d["thinking_blocks"] = list(m.thinking_blocks)

    return dict(sorted(d.items()))


def compute_history_hash(history_window: "tuple[Message, ...] | list[Message]") -> str:
    """Return a hex-digest SHA-256 hash of the history window.

    Fields included: role, content, tool_call_id, name, tool_calls, thinking_blocks.
    ``thinking`` (display text) and ``msg_id`` are intentionally excluded — they
    carry no structural information relevant to continuity.

    No external I/O is performed; the hash is computed purely from in-memory data.
    When a message has no in-memory content (content is empty string / empty list)
    the serialized empty value is used and ``hash_ref_only`` is noted as a no-op
    (we never read content_ref files during hashing).
    """
    serialized = json.dumps(
        [_stable_serialize_message(m) for m in history_window],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def message_to_dict(m: "Message") -> dict:
    """Serialize a Message to a lossless plain dict.

    All fields required for correct multi-turn replay are preserved,
    including ``thinking_blocks`` (which carry Anthropic cryptographic
    signatures that must be replayed verbatim in extended-thinking sessions).
    """
    d: dict = {
        "role": m.role,
        "content": m.content if isinstance(m.content, list) else (m.content or ""),
    }
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.name:
        d["name"] = m.name
    if m.tool_calls:
        d["tool_calls"] = [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in m.tool_calls]
    if m.thinking:
        d["thinking"] = m.thinking
    if m.thinking_blocks:
        d["thinking_blocks"] = list(m.thinking_blocks)
    return d


def dict_to_message(d: dict) -> "Message":
    """Deserialize a plain dict back to a Message."""
    tool_calls = tuple(
        ToolCall(id=tc["id"], name=tc["name"], input=tc.get("input", {})) for tc in d.get("tool_calls", [])
    )
    return Message(
        role=d.get("role", "user"),
        content=d.get("content", ""),
        tool_call_id=d.get("tool_call_id"),
        name=d.get("name"),
        tool_calls=tool_calls,
        thinking=d.get("thinking", ""),
        thinking_blocks=tuple(d.get("thinking_blocks") or ()),
    )


def trim_messages_to_budget(
    messages: "list[Message]",
    target_tokens: int,
    token_counter: "object | None" = None,
) -> "list[Message]":
    """Drop oldest non-first messages until ``count(messages) <= target_tokens``.

    Keeps index 0 (system prompt or first user message) intact.
    Preserves tool_use/tool_result pairs: when an assistant message with tool_calls
    is removed, its corresponding tool result messages are also removed (and vice versa).
    Shared by context-window guard processors.

    Args:
        token_counter: optional callable(messages) -> int.  When provided,
                       replaces ``rough_token_count`` for budget checks.
                       Pass ``provider.count_tokens`` in RL mode to use
                       the training tokenizer instead of tiktoken.
    """
    _count = token_counter if token_counter is not None else rough_token_count
    result = list(messages)
    while len(result) > 1 and _count(result) > target_tokens:
        # Find the first removable message group (starting from index 1)
        idx = 1
        if idx >= len(result):
            break
        msg = result[idx]

        if msg.role == "assistant" and msg.tool_calls:
            # Remove this assistant message AND all its subsequent tool result messages
            tool_call_ids = {tc.id for tc in msg.tool_calls}
            result.pop(idx)
            # Remove consecutive tool results that belong to these tool calls
            while idx < len(result) and result[idx].role == "tool" and result[idx].tool_call_id in tool_call_ids:
                result.pop(idx)
        elif msg.role == "tool":
            # Orphaned tool result — find and remove its parent assistant message too,
            # or just remove this message if parent is already gone
            result.pop(idx)
        else:
            result.pop(idx)
    return result


def _make_token_encoder() -> "object | None":
    """Try to build a tiktoken encoder. Returns None if unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")  # GPT-4 / Claude compatible
    except Exception:
        return None


_TOKEN_ENCODER = _make_token_encoder()


def _extract_text(content: "str | list") -> str:
    """Extract plain text from a content value (str or multimodal block list)."""
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


# ─── Multimodal token estimation ─────────────────────────────────────────────

# Default estimates — users can override via monkeypatch or subclass.
_DEFAULT_IMAGE_TOKENS = 1000  # ~Anthropic pricing: 1 token per 750 pixels
_DEFAULT_AUDIO_TOKENS_PER_SEC = 25  # placeholder for audio blocks


def estimate_block_tokens(block: dict) -> int:
    """Estimate token cost of a single non-text content block.

    Text blocks return 0 (counted by the BPE encoder in ``rough_token_count``).
    Image blocks return ``_DEFAULT_IMAGE_TOKENS``.  Audio blocks estimate from
    ``duration_seconds`` metadata.  Unknown types return a conservative 200.

    Override this function or the module-level constants for more accurate
    estimates tailored to your model provider.
    """
    btype = block.get("type", "")
    if btype == "text":
        return 0
    if btype == "image":
        return _DEFAULT_IMAGE_TOKENS
    if btype == "audio":
        duration = block.get("duration_seconds", 30)
        return int(duration * _DEFAULT_AUDIO_TOKENS_PER_SEC)
    return 200  # unknown modality fallback


def rough_token_count(messages: "list[Message] | tuple[Message, ...]") -> int:
    """Count tokens using tiktoken (cl100k_base) when available, else 4-char fallback.

    Uses the cl100k_base encoding (GPT-4/Claude-compatible BPE). Accurate enough
    for history truncation and budget checks. Not for billing — use provider
    usage fields (ModelResponseEvent.usage) for exact counts.

    Non-text content blocks (images, audio) are estimated via
    ``estimate_block_tokens``.

    Single source of truth: context processors, budget guards, and memory
    backends call this function so they all use the same counting method.
    """
    enc = _TOKEN_ENCODER
    total = 0
    for m in messages:
        text = _extract_text(m.content)
        if enc is not None:
            if text:
                # Token counting is an internal budgeting heuristic. Conversation
                # text may legitimately include literals that look like special
                # tokens (e.g. "<|endoftext|>"). For counting we treat them as
                # normal text instead of failing the whole run.
                try:
                    total += len(enc.encode(text, disallowed_special=())) + 4  # +4 for role/separator overhead
                except TypeError:
                    # Backward compatibility for encoder implementations without
                    # the disallowed_special kwarg.
                    total += len(enc.encode(text)) + 4
        else:
            total += len(text) // 4
        # Count non-text blocks (images, audio, etc.)
        if isinstance(m.content, list):
            for block in m.content:
                if isinstance(block, dict) and block.get("type") != "text":
                    total += estimate_block_tokens(block)
    return total
