# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from harnessx.core.events import (
    Message,
    ModelResponseEvent,
    ToolCall,
    ToolSchema,
    Usage,
    make_run_id,
)
from harnessx.core.trajectory import StatefulTrajectory, TokenAnnotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


async def _post_json(
    url: str,
    payload: dict,
    client: "httpx.AsyncClient | None" = None,
    timeout: float = 300.0,
) -> dict:
    """POST JSON payload to url, return parsed JSON response dict.

    If ``client`` is provided (SGLangProvider's persistent client), it is
    reused for connection pooling.  If None, a one-shot client is created
    (fallback for standalone usage).
    """
    if client is not None:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=timeout) as _client:
        resp = await _client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Exceptions for graceful abort / truncation signalling
# ---------------------------------------------------------------------------


class ContextLengthExceeded(Exception):
    """Raised when assembled input_ids exceed rollout_max_context_len."""


class GenerationAborted(Exception):
    """Raised when SGLang returns finish_reason='abort'."""


# ---------------------------------------------------------------------------
# StepCapture — per-turn side-channel data
# ---------------------------------------------------------------------------


@dataclass
class StepCapture:
    """Token-level data captured per model generation turn.

    Used by build_flat_sequence() after Harness.run() completes to reconstruct
    the flat token sequence (response_ids, loss_mask, rollout_logprobs) for
    RL training.

    input_ids:        complete context token IDs sent to SGLang at this step
    output_token_ids: SGLang-generated token IDs (model turn)
    output_logprobs:  corresponding log probabilities (for GRPO importance sampling)
    finish_reason:    "stop" | "length" | "abort"
    """

    input_ids: list[int]
    output_token_ids: list[int]
    output_logprobs: list[float]
    finish_reason: str
    inference_ms: float = 0.0  # SGLang /generate wall time for this turn


# ---------------------------------------------------------------------------
# FlatSequence — result of build_flat_sequence()
# ---------------------------------------------------------------------------


@dataclass
class FlatSequence:
    """Token-level flat sequence for a complete episode.

    prompt_ids:       step_captures[0].input_ids (initial context)
    response_ids:     interleaved model tokens + tool result tokens
    loss_mask:        1 for model tokens (contribute to loss), 0 for tool tokens
    rollout_logprobs: real logprobs for model tokens, 0.0 for tool tokens

    Invariant: len(response_ids) == len(loss_mask) == len(rollout_logprobs)
    """

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_logprobs: list[float]


# ---------------------------------------------------------------------------
# SGLangProvider
# ---------------------------------------------------------------------------


class SGLangProvider:
    """
    BaseModelProvider implementation for SGLang HTTP API.

    Bridges HarnessX Harness.run() <-> SGLang's token-level /generate endpoint.
    Captures (input_ids, output_token_ids, output_logprobs) per turn in
    self.step_captures for post-run flat sequence reconstruction via
    build_flat_sequence().

    Tokenization strategy — incremental (OpenClaw-RL style):
        Step 0:    full tokenization via message_formatter (or apply_chat_template)
        Step t+1:  known_prefix = caps[t].input_ids + caps[t].output_token_ids
                   new_suffix   = inter_turn_formatter(new_messages)
                   input_ids    = known_prefix + new_suffix

    This guarantees the flat sequence invariant by construction:
        caps[t+1].input_ids[:base_len] == caps[t].input_ids + caps[t].output_token_ids

    No BPE re-tokenization of the full conversation after step 0 — aligned with
    incremental multi-turn rollout tokenization.
    """

    def __init__(
        self,
        url: str,  # SGLang /generate endpoint
        tokenizer: Any,  # HF tokenizer (same as training tokenizer)
        sampling_params: dict,  # passed through to SGLang
        rollout_max_context_len: int,  # context length truncation guard
        message_formatter: Any = None,  # (messages, tools) -> list[int] | None = apply_chat_template
        inter_turn_formatter: Any = None,  # (new_messages) -> list[int] | None = native tail
        http_timeout: float = 300.0,  # per-request timeout for SGLang /generate (ignored when post_fn set)
        post_fn: "Callable | None" = None,  # optional HTTP post override; when set, _http_client is NOT created.
        # Signature: async (url: str, payload: dict) -> dict
        # Use slime.utils.http_utils.post in training to get
        # trust_env=False, Timeout(None), and 60-retry semantics.
        chat_template_kwargs: "dict | None" = None,
        # Optional extra kwargs for tokenizer.apply_chat_template().
        # Only used when message_formatter is None (native path).
        # Example: {"enable_thinking": False} for non-think mode.
        # Default None: no extra kwargs (backward-compatible).
        truncation_strategy: "str | None" = None,
        # How to handle input_ids exceeding rollout_max_context_len.
        # None (default): raise ContextLengthExceeded (existing behavior).
        # "head_tail": keep head (30%) + separator + tail, like terminal-rl.
        # Math recipe uses None; terminal recipe uses "head_tail".
        truncation_head_ratio: float = 0.3,
        # Fraction of budget allocated to the head portion when
        # truncation_strategy="head_tail".  Default 0.3 matches terminal-rl.
    ) -> None:
        self.url = url
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.rollout_max_context_len = rollout_max_context_len
        self.message_formatter = message_formatter
        self.inter_turn_formatter = inter_turn_formatter
        self._chat_template_kwargs: dict = dict(chat_template_kwargs or {})
        self._truncation_strategy = truncation_strategy
        self._truncation_head_ratio = truncation_head_ratio

        self.step_captures: list[StepCapture] = []
        self.truncated: bool = False
        self.aborted: bool = False

        # Expose context_window so run_loop and token-window guard processors
        # use the real rollout budget instead of the 64k default.
        self.context_window: int = rollout_max_context_len
        self.model: str = "sglang"

        # Dummy run_id/step_id for ModelResponseEvent (harness doesn't inspect these)
        self._run_id: str = make_run_id()

        # Tracks how many messages were processed at the last complete() call.
        # Used for incremental tokenization: new_messages = messages[_prev_msg_count:]
        self._prev_msg_count: int = 0

        # Head/tail truncation separator tokens (only computed when needed)
        if self._truncation_strategy == "head_tail":
            self._sep_marker = "\n[OMITTED MIDDLE]\n"
            self._sep_ids: list[int] = self.tokenizer.encode(
                self._sep_marker,
                add_special_tokens=False,
            )
        else:
            self._sep_ids = []

        # HTTP transport — two modes:
        #   post_fn provided (Slime training): use caller's client (trust_env=False,
        #     Timeout(None), 60-retry).  No local httpx.AsyncClient needed.
        #   post_fn=None (standalone/test): create a per-episode httpx.AsyncClient
        #     reused across all complete() calls in this episode.
        self._post_fn: "Callable | None" = post_fn
        if post_fn is None:
            self._http_client: "httpx.AsyncClient | None" = httpx.AsyncClient(timeout=http_timeout)
        else:
            self._http_client = None

    # ── Input truncation ──────────────────────────────────────────────────────

    def _truncate_input_ids(self, input_ids: list[int]) -> list[int]:
        """Truncate input_ids using head/tail strategy when they exceed the budget.

        Keeps the head (system prompt + early context) and tail (recent turns),
        inserting a [OMITTED MIDDLE] separator.  Matches terminal-rl's
        SGLangTurnClient._truncate_input_ids() behavior.

        Only called when truncation_strategy="head_tail".
        When truncation_strategy is None, the caller raises ContextLengthExceeded instead.
        """
        max_toks = self.rollout_max_context_len
        if len(input_ids) <= max_toks:
            return input_ids

        dropped = len(input_ids) - max_toks
        logger.warning(
            "Input too long: input=%d, budget=%d. Truncating %d token(s) (head_tail).",
            len(input_ids),
            max_toks,
            dropped,
        )

        head = max(1, int(max_toks * self._truncation_head_ratio))
        tail = max_toks - head - len(self._sep_ids)
        if tail <= 0:
            logger.warning(
                "Truncation tail is non-positive: tail=%d, head=%d, budget=%d, sep=%d",
                tail,
                head,
                max_toks,
                len(self._sep_ids),
            )
            tail = 1

        self.truncated = True
        return input_ids[:head] + self._sep_ids + input_ids[-tail:]

    # ── Public interface (BaseModelProvider protocol) ────────────────────────

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback=None,
    ) -> ModelResponseEvent:
        """
        Convert messages to token IDs, call SGLang, return ModelResponseEvent.

        Side effect: appends one StepCapture to self.step_captures.
        Raises ContextLengthExceeded or GenerationAborted on corresponding errors.
        """
        # 2. Tokenize — incremental strategy (OpenClaw-RL style).
        #
        # Step 0: full tokenization from scratch via message_formatter or apply_chat_template.
        # Step t+1: build input_ids incrementally:
        #     known_prefix = caps[t].input_ids + caps[t].output_token_ids
        #     new_suffix   = inter_turn_formatter(new_messages)   [retool path]
        #                  OR fresh_tokenization[len(known_prefix):]  [native path]
        #     input_ids    = known_prefix + new_suffix
        #
        # This guarantees the flat sequence invariant by construction — no BPE
        # re-tokenization of the full conversation after step 0.
        if not self.step_captures:
            # ── Step 0: full tokenization ────────────────────────────────────
            if self.message_formatter is not None:
                input_ids: list[int] = self.message_formatter(messages, tools)
            else:
                msg_dicts = _messages_to_openai_dicts(messages)
                tool_dicts = _schemas_to_openai_tools(tools) or None
                input_ids = self.tokenizer.apply_chat_template(
                    msg_dicts,
                    tools=tool_dicts,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self._chat_template_kwargs,
                )
        else:
            # ── Step t+1: incremental tokenization ───────────────────────────
            prev_cap = self.step_captures[-1]
            known_prefix: list[int] = prev_cap.input_ids + prev_cap.output_token_ids
            new_messages = messages[self._prev_msg_count :]

            if self.inter_turn_formatter is not None:
                # Retool path: independently tokenize only the inter-turn suffix
                # (\n\n<interpreter>...\n</interpreter>\n\n<|im_end|>\n<|im_start|>assistant\n).
                # new_messages contains the assistant message at step t (skipped —
                # its tokens are already in output_token_ids) + tool result messages.
                suffix_ids: list[int] = self.inter_turn_formatter(new_messages)
                input_ids = known_prefix + suffix_ids
            else:
                # Native path: full retokenization, then take tail after known_prefix.
                # Native Qwen3 format uses explicit special tokens (<|im_end|>) as turn
                # boundaries, which anchor BPE cleanly and make the tail extraction safe.
                msg_dicts = _messages_to_openai_dicts(messages)
                tool_dicts = _schemas_to_openai_tools(tools) or None
                fresh_ids: list[int] = self.tokenizer.apply_chat_template(
                    msg_dicts,
                    tools=tool_dicts,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self._chat_template_kwargs,
                )
                if len(fresh_ids) >= len(known_prefix):
                    input_ids = known_prefix + fresh_ids[len(known_prefix) :]
                else:
                    logger.warning(
                        "Native incremental: fresh tokenization (%d) shorter than "
                        "known prefix (%d) — falling back to fresh tokenization",
                        len(fresh_ids),
                        len(known_prefix),
                    )
                    input_ids = fresh_ids

        self._prev_msg_count = len(messages)

        # 3. Context length guard / truncation
        if len(input_ids) >= self.rollout_max_context_len:
            if self._truncation_strategy == "head_tail":
                # Graceful truncation: keep head + tail with [OMITTED MIDDLE] separator.
                # Episode continues with truncated context (matches terminal-rl behavior).
                input_ids = self._truncate_input_ids(input_ids)
            else:
                # Default: hard stop — raise exception caught by run_loop.
                # Episode ends immediately (math recipe behavior).
                self.truncated = True
                raise ContextLengthExceeded(
                    f"Input length {len(input_ids)} >= rollout_max_context_len {self.rollout_max_context_len}"
                )

        # 4. POST to SGLang /generate with return_logprob=True
        # Cap max_new_tokens to remaining context budget so that
        #   sample.tokens = prompt_ids + Σ(output_ids per turn)
        # never exceeds rollout_max_context_len (== max_tokens_per_gpu).
        # Without this cap, multi-turn episodes accumulate response_ids across
        # turns; sample.tokens can far exceed max_tokens_per_gpu, causing OOM
        # in Megatron's compute_log_prob microbatch packing.
        remaining_budget = self.rollout_max_context_len - len(input_ids) - 64
        dynamic_max_new_tokens = max(
            1,
            min(
                self.sampling_params.get("max_new_tokens", 8192),
                remaining_budget,
            ),
        )
        _sampling_params = {
            **self.sampling_params,
            "max_new_tokens": dynamic_max_new_tokens,
        }
        payload = {
            "input_ids": input_ids,
            "sampling_params": _sampling_params,
            "return_logprob": True,
        }
        t_infer = time.monotonic()
        if self._post_fn is not None:
            output: dict = await self._post_fn(self.url, payload)
        else:
            output = await _post_json(self.url, payload, client=self._http_client)
        _inference_ms = (time.monotonic() - t_infer) * 1000

        # 5. Check for abort
        finish_type: str = output.get("meta_info", {}).get("finish_reason", {}).get("type", "stop")
        if finish_type == "abort":
            self.aborted = True
            raise GenerationAborted("SGLang returned finish_reason='abort'")

        # 6. Extract token IDs and logprobs from output_token_logprobs
        output_token_logprobs = output.get("meta_info", {}).get("output_token_logprobs")
        if output_token_logprobs:
            output_token_ids: list[int] = [item[1] for item in output_token_logprobs]
            # Guard against None logprob values that SGLang occasionally emits for
            # special tokens (e.g. BOS). Replace with 0.0 — these tokens contribute
            # zero log-importance-ratio in GRPO, which is the safest conservative default.
            output_logprobs: list[float] = [
                float(item[0]) if item[0] is not None else 0.0 for item in output_token_logprobs
            ]
        else:
            # Fallback: no logprobs available (shouldn't happen with return_logprob=True)
            text_fallback: str = output.get("text", "")
            output_token_ids = self.tokenizer(text_fallback, add_special_tokens=False)["input_ids"]
            output_logprobs = [0.0] * len(output_token_ids)
            logger.warning("output_token_logprobs missing in SGLang response — using fallback")

        # 7. Decode text from output token IDs
        text: str = self.tokenizer.decode(output_token_ids, skip_special_tokens=False)

        # 8. Parse <tool_call> XML blocks from text (Qwen3 XML format)
        tool_calls: list[ToolCall] = _parse_qwen3_tool_calls(text)

        # 9. Build Usage from token counts
        usage = Usage(
            input_tokens=len(input_ids),
            output_tokens=len(output_token_ids),
        )

        # 10. Save side-channel capture
        self.step_captures.append(
            StepCapture(
                input_ids=input_ids,
                output_token_ids=output_token_ids,
                output_logprobs=output_logprobs,
                finish_reason=finish_type,
                inference_ms=_inference_ms,
            )
        )

        # 11. Map SGLang finish_type → HarnessX finish_reason
        # "end_turn" (no tool_calls) → run_loop breaks cleanly
        # "tool_use" → run_loop continues (executes tools, loops)
        # When SGLang returns "length" (max tokens hit), we use "end_turn" to
        # stop the run_loop.  The caller detects truncation via
        # step_captures[-1].finish_reason == "length" after Harness.run().
        if tool_calls:
            finish_reason = "tool_use"
        else:
            finish_reason = "end_turn"  # covers "stop" and "length" — both stop the loop

        return ModelResponseEvent(
            run_id=self._run_id,
            step_id=len(self.step_captures) - 1,
            content=text,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
        )

    def count_tokens(self, messages: list[Message]) -> int:
        """Token count using the provider's HF tokenizer.

        Concatenates all message text and encodes once to avoid per-message
        overhead.  More accurate than word-splitting for Chinese, LaTeX, and
        code content where word ≠ token.  Adds 4 tokens per message for
        role/format overhead (im_start/im_end/role/newline).
        """
        text_parts: list[str] = []
        for m in messages:
            if isinstance(m.content, str):
                text_parts.append(m.content)
            elif isinstance(m.content, list):
                for block in m.content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
        if not text_parts:
            return 4 * len(messages)
        full_text = "\n".join(text_parts)
        ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        return len(ids) + 4 * len(messages)

    def annotate_trajectory(self, trajectory: "StatefulTrajectory") -> None:
        """Populate TrajectoryStep.token_annotation from captured step data.

        Called automatically by Harness.run() after backfill_rewards().
        Delegates to backfill_token_annotations() which uses the side-channel
        step_captures to reconstruct per-step token sequences.
        """
        backfill_token_annotations(trajectory, self)

    async def aclose(self) -> None:
        """Close the persistent HTTP client.

        Call once after Harness.run() completes (and after annotate_trajectory).
        generate() in harness_rollout.py calls this in a finally block.

        No-op when post_fn was provided — the caller owns the client lifecycle.
        """
        if self._http_client is not None:
            await self._http_client.aclose()


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _messages_to_openai_dicts(messages: list[Message]) -> list[dict]:
    """Convert HarnessX Message objects to OpenAI-compatible dicts.

    Critical: Message.tool_calls uses field 'input' (HarnessX internal),
    but apply_chat_template for Qwen3 expects 'arguments' (OpenAI format).
    This function performs the conversion.
    """
    result = []
    for m in messages:
        d: dict = {
            "role": m.role,
            "content": m.content if isinstance(m.content, list) else (m.content or ""),
        }
        if m.tool_calls:
            # OpenAI format: "arguments" (JSON string), not "input" (dict)
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input),
                    },
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        result.append(d)
    return result


def _schemas_to_openai_tools(tools: list[ToolSchema]) -> list[dict]:
    """Convert ToolSchema list to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Tool call XML parsing (Qwen3 format)
# ---------------------------------------------------------------------------


def _parse_qwen3_tool_calls(text: str) -> list[ToolCall]:
    """Parse <tool_call>{"name":..., "arguments":{...}}</tool_call> blocks.

    Qwen3 XML tool call format (NOT OpenAI API native function calling).

    Two-pass JSON parsing to handle both common model output patterns:
      Pass 1 — json.loads(raw): handles valid JSON including structural newlines
               (JSON spec allows whitespace between tokens).  Covers:
               • single-line JSON (most common for SFT-trained Qwen3)
               • multi-line JSON with properly escaped \\n in string values
      Pass 2 — fallback: escape raw control chars (\\n, \\r) in string values.
               Handles the case where the model emits literal newlines inside
               a JSON string value (technically invalid JSON, but common when
               model generates multi-line code blocks in the arguments).
    """
    results: list[ToolCall] = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        json_str = m.group(1)
        data = None

        # Pass 1: direct parse — correct for all valid JSON (incl. structural newlines)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Pass 2: escape raw control characters and retry
        if data is None:
            try:
                data = json.loads(json_str.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n"))
            except json.JSONDecodeError:
                pass

        if data is None:
            continue

        try:
            name = data.get("name", "")
            args = data.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                results.append(
                    ToolCall(
                        id=str(uuid.uuid4()),
                        name=name,
                        input=args,  # ToolCall uses .input, NOT .arguments
                    )
                )
        except (KeyError, AttributeError):
            pass

    return results


# ---------------------------------------------------------------------------
# build_flat_sequence — reconstruct token sequence after Harness.run()
# ---------------------------------------------------------------------------


def _find_first_diff(a: list[int], b: list[int]) -> int:
    """Return index of first divergence between two lists (for diagnostics)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def build_flat_sequence(provider: SGLangProvider) -> FlatSequence:
    """
    Reconstruct flat token sequence from SGLangProvider.step_captures.

    Algorithm based on Training Correctness Invariant B:
        step_captures[t+1].input_ids
          == step_captures[t].input_ids
           + step_captures[t].output_token_ids   (model tokens)
           + tool_result_ids[t]                  (tool result tokens)

    Therefore:
        tool_result_ids[t] = step_captures[t+1].input_ids[
            len(step_captures[t].input_ids) + len(step_captures[t].output_token_ids):
        ]

    Returns FlatSequence with:
        - prompt_ids:       step_captures[0].input_ids
        - response_ids:     model_ids[0] + tool_ids[0] + model_ids[1] + tool_ids[1] + ...
        - loss_mask:        [1]*model + [0]*tool interleaved
        - rollout_logprobs: real logprobs for model, 0.0 for tool
    """
    caps = provider.step_captures
    if not caps:
        raise ValueError("No step captures — SGLangProvider.complete() was never called")

    prompt_ids = caps[0].input_ids
    resp: list[int] = []
    mask: list[int] = []
    logps: list[float] = []

    for t, cap in enumerate(caps):
        # ① Model generation tokens (loss=1, real logprobs)
        resp += cap.output_token_ids
        mask += [1] * len(cap.output_token_ids)
        logps += cap.output_logprobs

        # ② Tool result tokens (loss=0, logprob=0.0)
        # Extracted from the diff between consecutive input_ids
        if t + 1 < len(caps):
            base_len = len(cap.input_ids) + len(cap.output_token_ids)
            next_ids = caps[t + 1].input_ids

            # Flat sequence invariant — guaranteed by construction via incremental
            # tokenization in complete().  A violation here means something outside
            # the provider mutated the message sequence (e.g. compaction processor).
            if len(next_ids) < base_len:
                logger.error(
                    "Flat sequence invariant broken at step %d: "
                    "next input_ids (%d) shorter than input+output (%d). "
                    "Is a CompactionProcessor enabled? Skipping tool_ids.",
                    t,
                    len(next_ids),
                    base_len,
                )
                tool_ids = []
            else:
                expected_prefix = cap.input_ids + cap.output_token_ids
                actual_prefix = next_ids[:base_len]
                if expected_prefix != actual_prefix:
                    logger.error(
                        "Flat sequence prefix mismatch at step %d "
                        "(first divergence near token %d). "
                        "Incremental tokenization invariant was violated — "
                        "check inter_turn_formatter.",
                        t,
                        _find_first_diff(expected_prefix, actual_prefix),
                    )
                tool_ids = next_ids[base_len:]
            resp += tool_ids
            mask += [0] * len(tool_ids)
            logps += [0.0] * len(tool_ids)

    if not (len(resp) == len(mask) == len(logps)):
        raise RuntimeError(f"FlatSequence length mismatch: {len(resp)} resp / {len(mask)} mask / {len(logps)} logps")

    return FlatSequence(
        prompt_ids=prompt_ids,
        response_ids=resp,
        loss_mask=mask,
        rollout_logprobs=logps,
    )


# ---------------------------------------------------------------------------
# backfill_token_annotations — populate TrajectoryStep.token_annotation
# ---------------------------------------------------------------------------


def backfill_token_annotations(
    traj: StatefulTrajectory,
    provider: SGLangProvider,
) -> None:
    """
    Populate TrajectoryStep.token_annotation from provider.step_captures.

    Call once after Harness.run() returns, before build_flat_sequence().
    After this call, traj.has_token_annotations() is True, and
    traj.to_rl_records(fmt) can be used with any RLFormat.

    Algorithm mirrors build_flat_sequence() but writes per-step into the
    trajectory instead of aggregating into a single flat sequence:

        For step t:
            prompt_ids        = caps[t].input_ids
            model_ids         = caps[t].output_token_ids
            model_logprobs    = caps[t].output_logprobs
            tool_ids          = caps[t+1].input_ids[base_len:]   (or [] for last step)
            response_ids      = model_ids + tool_ids
            response_mask     = [1]*len(model_ids) + [0]*len(tool_ids)
            response_logprobs = model_logprobs + [0.0]*len(tool_ids)

    Invariant: len(traj.steps) == len(provider.step_captures)
    If they differ, a warning is logged and only the shorter range is filled.

    Args:
        traj:     StatefulTrajectory returned by Harness.run()
        provider: SGLangProvider used during the run
    """
    caps = provider.step_captures
    steps = traj.steps

    if len(steps) != len(caps):
        logger.warning(
            "backfill_token_annotations: traj.steps (%d) != step_captures (%d) — filling only %d steps",
            len(steps),
            len(caps),
            min(len(steps), len(caps)),
        )

    for t, (step, cap) in enumerate(zip(steps, caps)):
        model_ids = cap.output_token_ids
        model_logprobs = cap.output_logprobs

        # Tool result tokens: diff between consecutive input_ids
        if t + 1 < len(caps):
            base_len = len(cap.input_ids) + len(cap.output_token_ids)
            next_ids = caps[t + 1].input_ids
            if len(next_ids) < base_len:
                logger.error(
                    "backfill_token_annotations: flat-sequence invariant broken at step %d "
                    "(next input_ids %d < base_len %d). Tool tokens set to [].",
                    t,
                    len(next_ids),
                    base_len,
                )
                tool_ids = []
            else:
                tool_ids = next_ids[base_len:]
        else:
            tool_ids = []

        response_ids = model_ids + tool_ids
        response_mask = [1] * len(model_ids) + [0] * len(tool_ids)
        response_logprobs = model_logprobs + [0.0] * len(tool_ids)

        step.token_annotation = TokenAnnotation(
            prompt_ids=cap.input_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            token_reward=0.0,  # filled later by reward_func() PRM scoring
        )


def _flat_sequence_from_trajectory(traj: StatefulTrajectory) -> FlatSequence:
    """
    Reconstruct FlatSequence from trajectory token_annotations.

    Equivalent to build_flat_sequence(provider) but reads from the trajectory
    instead of the provider's step_captures side-channel.  Used to validate
    that backfill_token_annotations() produced consistent results.

    Requires traj.has_token_annotations() == True.
    """
    if not traj.steps:
        raise ValueError("_flat_sequence_from_trajectory: trajectory has no steps")

    prompt_ids = traj.steps[0].token_annotation.prompt_ids  # type: ignore[union-attr]
    resp: list[int] = []
    mask: list[int] = []
    logps: list[float] = []

    for step in traj.steps:
        ta = step.token_annotation  # type: ignore[union-attr]
        resp += ta.response_ids
        mask += ta.response_mask
        logps += ta.response_logprobs

    return FlatSequence(
        prompt_ids=prompt_ids,
        response_ids=resp,
        loss_mask=mask,
        rollout_logprobs=logps,
    )


def validate_token_annotation_consistency(
    traj: StatefulTrajectory,
    flat: FlatSequence,
) -> bool:
    """
    Assert trajectory token_annotations reconstruct to the same FlatSequence.

    Returns True if consistent, False (with error logs) if not.
    Used in dev/validation mode — gate with HARNESSX_VALIDATE_TOKENS=1.

    Args:
        traj: trajectory after backfill_token_annotations()
        flat: FlatSequence from build_flat_sequence(provider) (the reference)
    """
    if not traj.has_token_annotations():
        logger.error("validate_token_annotation_consistency: token_annotations missing")
        return False

    traj_flat = _flat_sequence_from_trajectory(traj)
    ok = True

    if traj_flat.prompt_ids != flat.prompt_ids:
        logger.error(
            "Token annotation validation FAILED: prompt_ids mismatch "
            "(traj=%d tokens vs captures=%d tokens, first diff at %d)",
            len(traj_flat.prompt_ids),
            len(flat.prompt_ids),
            _find_first_diff(traj_flat.prompt_ids, flat.prompt_ids),
        )
        ok = False

    if traj_flat.response_ids != flat.response_ids:
        logger.error(
            "Token annotation validation FAILED: response_ids mismatch "
            "(traj=%d tokens vs captures=%d tokens, first diff at %d)",
            len(traj_flat.response_ids),
            len(flat.response_ids),
            _find_first_diff(traj_flat.response_ids, flat.response_ids),
        )
        ok = False

    if traj_flat.loss_mask != flat.loss_mask:
        logger.error(
            "Token annotation validation FAILED: loss_mask mismatch (traj=%d vs captures=%d)",
            len(traj_flat.loss_mask),
            len(flat.loss_mask),
        )
        ok = False

    if traj_flat.rollout_logprobs != flat.rollout_logprobs:
        logger.error(
            "Token annotation validation FAILED: rollout_logprobs mismatch (traj=%d vs captures=%d)",
            len(traj_flat.rollout_logprobs),
            len(flat.rollout_logprobs),
        )
        ok = False

    if ok:
        logger.debug(
            "Token annotation validation passed (%d response tokens)",
            len(flat.response_ids),
        )

    return ok
