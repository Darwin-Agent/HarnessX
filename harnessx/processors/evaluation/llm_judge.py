# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""LLMJudgeProcessor — ground-truth-free structured verdict emitted on task end."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING, AsyncIterator, Callable, Iterable

from ...core.events import Message, TaskEndEvent
from ...core.processor import MultiHookProcessor


def _content_as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content) if content else ""


_FINAL_ANSWER_RE = re.compile(r"(?:final\s+answer)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
_ANSWER_IS_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:the\s+)?answer\s*(?:is)?\s*[:：]\s*(.+)",
    re.IGNORECASE,
)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


# Ported from benchmarks/gaia/evaluator.py::_extract_answer_from_messages.
# Keep in sync manually; do not import from benchmarks/ to preserve
# harnessx's "core never imports third-party benchmark libs" rule.
def default_answer_extractor(messages: "Iterable[Message]") -> str:
    """GAIA-style final answer extractor. Priority:
    1. `FINAL ANSWER:` marker (case-insensitive), last occurrence in assistant messages.
    2. `The answer is:` / `answer:` marker.
    3. Last non-empty line of the last assistant message.
    """
    msg_list = list(messages)

    for msg in reversed(msg_list):
        if msg.role != "assistant" or not msg.content:
            continue
        content = _content_as_text(msg.content)
        m = _FINAL_ANSWER_RE.search(content)
        if m:
            return _strip_markdown(m.group(1).strip().split("\n")[0].strip())

    for msg in reversed(msg_list):
        if msg.role != "assistant" or not msg.content:
            continue
        content = _content_as_text(msg.content)
        m = _ANSWER_IS_RE.search(content)
        if m:
            return _strip_markdown(m.group(1).strip().split("\n")[0].strip())

    for msg in reversed(msg_list):
        if msg.role != "assistant" or not msg.content:
            continue
        content = _content_as_text(msg.content)
        lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
        if lines:
            return _strip_markdown(lines[-1])

    return ""


def render_trajectory_summary(
    *,
    tool_trace: "list[tuple[int, str, str, str, int]]",
    final_messages: "Iterable[Message]",
    budget_chars: int = 4000,
) -> str:
    """Render a compact trajectory summary for the judge prompt.

    tool_trace is a list of (step_no, tool_name, input_summary, status, output_len).
    status is either "ok" or "error: <reason>".

    Format:
        ## Tool Trace
        Step 1: tool_name(input) → status, Nchars
        ...
        (middle folded as "Step i-j omitted" when over budget)
        ## Final Reasoning
        <last 1-2 assistant messages, truncated>
    """
    lines: list[str] = ["## Tool Trace"]
    if not tool_trace:
        lines.append("(no tools called)")
    else:
        trace_lines = [f"Step {s}: {name}({inp}) → {status}, {olen} chars" for s, name, inp, status, olen in tool_trace]
        # Fold middle if total would exceed half the budget
        trace_budget = budget_chars // 2
        total = sum(len(ln) + 1 for ln in trace_lines)
        if total > trace_budget and len(trace_lines) > 4:
            keep = 2
            head = trace_lines[:keep]
            tail = trace_lines[-keep:]
            mid_start = tool_trace[keep][0]
            mid_end = tool_trace[-keep - 1][0]
            trace_lines = head + [f"... Steps {mid_start}-{mid_end} omitted ..."] + tail
        lines.extend(trace_lines)

    lines.append("")
    lines.append("## Final Reasoning")

    msg_list = list(final_messages)
    reasoning_parts: list[str] = []
    remaining = budget_chars - sum(len(ln) + 1 for ln in lines)
    remaining = max(500, remaining)  # guarantee minimum
    for msg in reversed(msg_list):
        if msg.role != "assistant" or not msg.content:
            continue
        content = _content_as_text(msg.content)
        reasoning_parts.insert(0, content)
        if sum(len(p) for p in reasoning_parts) >= remaining:
            break
        if len(reasoning_parts) >= 2:
            break

    reasoning = "\n\n".join(reasoning_parts).strip()
    if len(reasoning) > remaining:
        reasoning = reasoning[: remaining - 20] + " ... [truncated]"
    lines.append(reasoning if reasoning else "(no assistant reasoning captured)")

    return "\n".join(lines)


_VALID_VERDICTS = frozenset(
    {
        "plausible",
        "unsupported",
        "hedging",
        "format_wrong",
        "refused",
        "no_answer",
    }
)

_JUDGE_SYSTEM = """\
You extract SIGNALS from an agent's trajectory for improving its configuration.
You do NOT grade correctness — ground truth is intentionally withheld from you.

Output STRICT JSON ONLY, no prose, no code fences. All 6 top-level fields
required; use empty string / empty object where a field is not applicable.

Schema:
{
  "verdict":    "plausible|unsupported|hedging|format_wrong|refused|no_answer",
  "confidence": <float 0.0 to 1.0>,
  "cause":      "<=120 chars: root cause, not symptom",
  "missing":    "<=80 chars: short tag for the absent capability; '' if n/a",
  "lesson":     "<=100 chars: transferable takeaway; '' if n/a",
  "missing_capability": {
    "present": <bool: true when the trajectory shows a gap that behaviour
                tuning won't fix — the agent needed a tool/helper/parser
                it did not have; false otherwise>,
    "summary": "<=240 chars, free text. Describe the missing capability
                as a generic tool/helper shape the meta-agent could author
                (e.g. 'a client that wraps <service> with retry + alt-
                endpoint fallback'; 'a parser for <file-type> tables';
                'a memory layer that persists across subprocesses').
                Do NOT name specific URLs or benchmark tasks. Do NOT
                prescribe implementation — describe shape of the gap.
                Leave '' when present=false.",
    "evidence_steps": <list of integer step indices from the trace that
                most clearly show the gap; keep to <=6 entries; [] when
                present=false>
  }
}

Verdict meanings:
- plausible:    answer specific AND trajectory supports it
- unsupported:  answer given but evidence doesn't back it (hallucination risk)
- hedging:      agent qualified/hedged instead of committing
- format_wrong: rambled without emitting a clean final answer
- refused:      agent said it can't do the task
- no_answer:    extractor pulled nothing

Examples (abbreviated):
- plausible  → cause="triangulated 3 independent sources"
                missing_capability={"present": false, "summary": "",
                                    "evidence_steps": []}
- unsupported→ cause="API throttled; agent degraded to guessing"
                missing="robust API client"
                missing_capability={"present": true,
                  "summary": "A client for this external service with
                    exponential backoff and an alt-endpoint fallback chain;
                    the agent repeatedly hit a single endpoint and kept
                    retrying without switching strategies.",
                  "evidence_steps": [8, 12, 17, 22]}
- refused    → cause="task needs PDF table parse; no such tool"
                missing="file parser"
                missing_capability={"present": true,
                  "summary": "A structured parser that extracts tables
                    from PDFs and returns rows/columns as JSON; current
                    tools can fetch the file but not decode its tables.",
                  "evidence_steps": [3, 4]}
- format_wrong→cause="agent narrated instead of FINAL ANSWER"
                missing_capability={"present": false, "summary": "",
                                    "evidence_steps": []}

Rules for missing_capability:
- It is NOT a duplicate of `missing`. `missing` is a short tag; summary
  is the richer, generic-tool-shape description.
- `present=true` ONLY when the gap is a CAPABILITY the agent lacked, not
  a BEHAVIOUR it chose wrongly. "Agent was lazy and gave up" is behaviour;
  "agent had no tool to retry under rate limit" is capability.
- The summary should describe a CLASS of tool, not a fix for one task.
"""


def build_judge_prompt(
    *,
    task_description: str,
    trajectory_summary: str,
    extracted_answer: str,
) -> str:
    task_desc = (task_description or "")[:1000]
    answer = extracted_answer if extracted_answer else "<none>"
    body = f"{_JUDGE_SYSTEM}\n\n## Question\n{task_desc}\n\n{trajectory_summary}\n\n## Extracted Answer\n{answer}\n"
    return body


def parse_judge_response(raw: str) -> dict:
    """Parse a judge model's JSON response into a canonical 5-field verdict.

    Tolerant of:
    - Leading / trailing ```json``` code fences
    - Missing optional fields (cause/missing/lesson default to '')
    - Out-of-range confidence (clamped to [0, 1])

    Strict on:
    - verdict must be one of _VALID_VERDICTS
    - response must parse as JSON

    Raises ValueError on unrecoverable input.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"response is not a JSON object: {type(obj).__name__}")

    verdict = str(obj.get("verdict") or "").strip()
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}; valid: {sorted(_VALID_VERDICTS)}")

    try:
        confidence = float(obj.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    def _str_field(k: str, max_len: int) -> str:
        v = obj.get(k) or ""
        s = str(v).strip()
        return s[:max_len]

    mc_raw = obj.get("missing_capability")
    missing_capability = _normalize_missing_capability(mc_raw)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "cause": _str_field("cause", 120),
        "missing": _str_field("missing", 80),
        "lesson": _str_field("lesson", 100),
        "missing_capability": missing_capability,
    }


def _empty_missing_capability() -> dict:
    return {"present": False, "summary": "", "evidence_steps": []}


def _normalize_missing_capability(raw: object) -> dict:
    """Coerce the judge's ``missing_capability`` value into a canonical dict.

    Tolerant of:
    - None / missing key → empty payload
    - Accidentally-flat ``summary`` string at the top level (some models
      drop the nested object if `present=false`)
    - ``evidence_steps`` items that are str-digits or floats
    - ``evidence_steps`` longer than the cap (trimmed to 6)
    - ``summary`` longer than the cap (trimmed to 240 chars)

    Strict: never raises — a malformed field degrades to the empty payload
    so the verdict pipeline keeps flowing.
    """
    if not isinstance(raw, dict):
        return _empty_missing_capability()

    present = bool(raw.get("present"))
    summary_raw = raw.get("summary") or ""
    summary = str(summary_raw).strip()[:240]

    steps_raw = raw.get("evidence_steps") or []
    steps: list[int] = []
    if isinstance(steps_raw, (list, tuple)):
        for item in steps_raw:
            if isinstance(item, bool):
                continue  # bool is a subclass of int; skip it
            try:
                steps.append(int(item))
            except (TypeError, ValueError):
                continue
            if len(steps) >= 6:
                break

    # Self-consistency: present=true demands a non-empty summary.
    # present=false demands an empty summary + empty steps. If the model
    # confuses these, normalize rather than discard.
    if present and not summary:
        present = False
    if not present:
        summary = ""
        steps = []

    return {"present": present, "summary": summary, "evidence_steps": steps}


if TYPE_CHECKING:
    from ...providers.base import BaseModelProvider


class LLMJudgeProcessor(MultiHookProcessor):
    """Placeholder — filled in by later tasks."""

    required_providers: frozenset = frozenset()

    _singleton_group = "llm_judge"
    __hx_skip_serialization__ = frozenset({"judge_provider", "answer_extractor"})

    def __init__(
        self,
        judge_provider: "BaseModelProvider | None" = None,
        *,
        judge_model: "str | None" = None,
        answer_extractor: "Callable[[list[Message]], str] | None" = None,
        trace_budget_chars: int = 4000,
        timeout_s: float = 30.0,
        verdict_sink: "dict | None" = None,
    ) -> None:
        if judge_provider is None and judge_model is None:
            raise ValueError("LLMJudgeProcessor: one of judge_provider or judge_model must be set")
        self._judge_provider = judge_provider
        self._judge_model = judge_model
        self._answer_extractor = answer_extractor
        self._trace_budget_chars = trace_budget_chars
        self._timeout_s = timeout_s
        self._verdict_sink: dict = verdict_sink if verdict_sink is not None else {}

    def _get_judge_provider(self) -> "BaseModelProvider":
        if self._judge_provider is not None:
            return self._judge_provider
        model = (self._judge_model or "").strip()
        if not model:
            raise ValueError("LLMJudgeProcessor: judge_model is empty; cannot construct provider")
        if model.startswith("anthropic/"):
            from ...providers.anthropic_provider import AnthropicProvider

            model_name = model[len("anthropic/") :]
            self._judge_provider = AnthropicProvider(
                model=model_name,
                base_url=os.environ.get("ANTHROPIC_API_BASE"),
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                extended_thinking=False,
                max_tokens=4096,
            )
            return self._judge_provider
        from ...providers.litellm_provider import LiteLLMProvider

        self._judge_provider = LiteLLMProvider(model)
        return self._judge_provider

    async def on_task_end(self, event: TaskEndEvent) -> AsyncIterator[TaskEndEvent]:
        extractor = self._answer_extractor or default_answer_extractor
        try:
            answer = extractor(event.final_messages)
        except Exception as exc:  # defensive
            self._verdict_sink[event.run_id] = {
                "verdict": {
                    "verdict": "no_answer",
                    "confidence": 1.0,
                    "cause": f"extractor error: {type(exc).__name__}",
                    "missing": "",
                    "lesson": "",
                    "missing_capability": _empty_missing_capability(),
                },
                "extracted_answer": "",
            }
            yield event
            return

        if not answer:
            self._verdict_sink[event.run_id] = {
                "verdict": {
                    "verdict": "no_answer",
                    "confidence": 1.0,
                    "cause": "extractor produced empty string",
                    "missing": "",
                    "lesson": "",
                    "missing_capability": _empty_missing_capability(),
                },
                "extracted_answer": "",
            }
            yield event
            return

        trace_summary = self._render_trace_from_event(event)
        prompt = build_judge_prompt(
            task_description=getattr(event, "task_description", "") or "",
            trajectory_summary=trace_summary,
            extracted_answer=answer,
        )
        verdict = await self._call_judge(prompt, self._get_judge_provider())

        self._verdict_sink[event.run_id] = {
            "verdict": verdict,
            "extracted_answer": answer,
        }
        yield event

    def _render_trace_from_event(self, event: TaskEndEvent) -> str:
        # Extract per-step tool info from state_snapshot if available;
        # fallback: use message roles to reconstruct a minimal trace.
        tool_trace: list[tuple[int, str, str, str, int]] = []
        snapshot = event.state_snapshot or {}
        msgs = snapshot.get("messages") or []
        step = 0
        for m in msgs:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if role == "tool":
                step += 1
                content_str = _content_as_text(content)
                tool_name = (m.get("name") if isinstance(m, dict) else getattr(m, "name", "")) or "tool"
                tool_trace.append(
                    (
                        step,
                        tool_name,
                        "(args omitted)",
                        "ok" if content_str else "empty",
                        len(content_str),
                    )
                )
        return render_trajectory_summary(
            tool_trace=tool_trace,
            final_messages=event.final_messages,
            budget_chars=self._trace_budget_chars,
        )

    async def _call_judge(self, prompt: str, provider: "BaseModelProvider | None" = None) -> dict:
        if provider is None:
            provider = self._get_judge_provider()
        messages = [Message(role="user", content=prompt)]

        async def _one_shot() -> str:
            response = await provider.complete(
                messages=messages,
                tools=[],
            )
            return _content_as_text(response.content)

        for attempt in range(2):
            try:
                raw = await asyncio.wait_for(_one_shot(), timeout=self._timeout_s)
            except asyncio.TimeoutError:
                return {
                    "verdict": "judge_error",
                    "confidence": 0.0,
                    "cause": f"judge timeout after {self._timeout_s}s",
                    "missing": "",
                    "lesson": "",
                    "missing_capability": _empty_missing_capability(),
                }
            except Exception as exc:
                return {
                    "verdict": "judge_error",
                    "confidence": 0.0,
                    "cause": f"{type(exc).__name__}: {str(exc)[:80]}",
                    "missing": "",
                    "lesson": "",
                    "missing_capability": _empty_missing_capability(),
                }

            try:
                return parse_judge_response(raw)
            except ValueError:
                if attempt == 0:
                    # Retry once with nudge; extend message chain
                    messages = [
                        Message(role="user", content=prompt),
                        Message(role="assistant", content=raw),
                        Message(
                            role="user",
                            content=(
                                "Previous output was not valid JSON matching the schema. "
                                "Output STRICT JSON ONLY, no prose, no code fences."
                            ),
                        ),
                    ]
                    continue

        return {
            "verdict": "judge_error",
            "confidence": 0.0,
            "cause": "judge returned invalid JSON twice",
            "missing": "",
            "lesson": "",
            "missing_capability": _empty_missing_capability(),
        }

    def get_verdict(self, run_id: str) -> "dict | None":
        return self._verdict_sink.get(run_id)
