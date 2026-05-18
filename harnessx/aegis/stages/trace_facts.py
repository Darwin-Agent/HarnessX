# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Layer A — mechanical trace-fact extraction.

Reads cleaned trajectory jsonl and returns structured, deterministic facts
(tool-call shape table, exit summary, repeated-call runs, tool-effectiveness
heuristic). The Digester LLM receives this block verbatim and is told NOT to
rewrite it — Layer B (pathology signals) and Layer C (diagnosis) build on top.

Design: every fact carries a trajectory anchor (`trajectories/<file>#step_N`)
so downstream agents can always back-cite to the raw event.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


_MULTIMODAL_MARKERS = (
    "[image displayed below]",
    "image displayed below",
    "[content omitted",
    "content omitted for length",
    "<image>",
    "<file>",
)


def _rollout_tag(path: Path) -> str:
    """Extract 'rN' from a cleaned trajectory filename like '<task>_r3.jsonl'."""
    stem = path.name.replace(".jsonl", "")
    if "_r" in stem:
        return "r" + stem.rsplit("_r", 1)[1]
    return stem


def _args_sha(args: object) -> str:
    try:
        blob = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        blob = repr(args)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _content_to_text(content: object) -> tuple[str, bool]:
    """Flatten assistant/tool content to a plain string. Returns (text, had_structured_blocks)."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts: list[str] = []
        structured = False
        for blk in content:
            if isinstance(blk, dict):
                if blk.get("type") in ("image", "file", "image_url", "input_image"):
                    structured = True
                    parts.append(f"<{blk.get('type')}>")
                else:
                    txt = blk.get("text") or blk.get("content") or ""
                    parts.append(str(txt))
            else:
                parts.append(str(blk))
        return "\n".join(parts), structured
    return str(content), False


def _classify_return(text: str, had_structured: bool) -> str:
    """text / multimodal / multimodal_coerced / short_marker / empty / error"""
    if not text.strip():
        return "empty"
    low = text.strip().lower()
    if low.startswith("error") or "traceback" in low[:200]:
        return "error"
    if had_structured:
        return "multimodal"
    if len(text) < 120 and any(m in low for m in (m.lower() for m in _MULTIMODAL_MARKERS)):
        return "multimodal_coerced"
    return "text"


@dataclass
class ToolCallFact:
    rollout: str
    step: int
    tool: str
    args_sha: str
    args_preview: str  # short JSON-preview of args, ≤ 120 chars
    return_type: str   # text | multimodal | multimodal_coerced | short_marker | empty | error
    return_len: int
    next_uses_result: bool | None  # heuristic; None if no next assistant step

    @property
    def anchor(self) -> str:
        return f"trajectories/<file_for_{self.rollout}>#step_{self.step}"


@dataclass
class ExitFact:
    rollout: str
    exit_reason: str
    total_steps: int
    passed: bool | None
    terminal_snippet: str  # last 200 chars of last assistant content (final answer locus)


@dataclass
class RepeatRun:
    rollout: str
    tool: str
    args_sha: str
    steps: list[int]


@dataclass
class ToolBurst:
    """A single tool was hammered in one rollout — likely a loop trap.

    The existing ``RepeatRun`` only fires when args are byte-identical
    between calls (and no other tool intervenes). That misses the common
    case where a tool is called many times with *different* args (e.g.
    SmartFetch 75× across 30 different URLs). Bursts are detected from
    aggregate counts so that pattern is surfaced.
    """
    rollout: str
    tool: str
    total_calls: int               # whole-rollout total
    max_calls_in_one_step: int     # peak parallel/iterated count within one step
    peak_step: int                 # step where the peak happened
    severity: str                  # "high" if total >= 30 or peak >= 15, else "medium"


# Thresholds chosen so a healthy multi-tool trajectory (typically ≤ 5–10
# calls of any one tool, ≤ 3 in any single step) never trips, but the
# observed loop-trap shape (e.g. 75 SmartFetch in 19 steps with peak ~30
# in step 1) reliably does.
_BURST_TOTAL_MIN = 20
_BURST_PEAK_MIN = 10


@dataclass
class TraceFacts:
    task_id: str
    rollouts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallFact] = field(default_factory=list)
    exits: list[ExitFact] = field(default_factory=list)
    repeats: list[RepeatRun] = field(default_factory=list)
    bursts: list[ToolBurst] = field(default_factory=list)
    trajectory_file_by_rollout: dict[str, str] = field(default_factory=dict)

    def anchor(self, rollout: str, step: int) -> str:
        fname = self.trajectory_file_by_rollout.get(rollout, f"<{rollout}>")
        return f"trajectories/{fname}#step_{step}"

    def to_markdown(self) -> str:
        """Render as a self-contained markdown section. Stable field order —
        Planner may parse cross-task statistics off this schema."""
        lines: list[str] = []
        lines.append("## Trace Facts (Layer A — mechanical; do not rewrite)")
        lines.append("")
        lines.append(
            "> These facts were extracted deterministically from trajectory jsonl. "
            "Layer B (Pathology signals) and Layer C (Diagnosis) below should treat "
            "them as ground truth evidence."
        )
        lines.append("")

        # --- Exits ---
        lines.append("### Exits")
        lines.append("")
        lines.append("| rollout | exit_reason | steps | passed | terminal |")
        lines.append("|---|---|---|---|---|")
        for e in self.exits:
            passed_str = "—" if e.passed is None else ("yes" if e.passed else "no")
            snip = e.terminal_snippet.replace("|", "/").replace("\n", " ")[:160]
            if len(e.terminal_snippet) > 160:
                snip += "…"
            lines.append(
                f"| {e.rollout} | {e.exit_reason} | {e.total_steps} | {passed_str} | {snip} |"
            )
        if not self.exits:
            lines.append("| (no episode_end events) |  |  |  |  |")
        lines.append("")

        # --- Tool calls per rollout ---
        lines.append("### Tool calls")
        lines.append("")
        for rollout in self.rollouts:
            calls = [c for c in self.tool_calls if c.rollout == rollout]
            if not calls:
                lines.append(f"**{rollout}** — no tool calls.")
                lines.append("")
                continue
            lines.append(
                f"**{rollout}** → `trajectories/{self.trajectory_file_by_rollout.get(rollout, '')}`"
            )
            lines.append("")
            lines.append(
                "| step | tool | args_sha | args_preview | return_type | return_len | next_uses_result |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for c in calls:
                nu = "—" if c.next_uses_result is None else ("yes" if c.next_uses_result else "**NO**")
                prev = c.args_preview.replace("|", "/").replace("\n", " ")
                lines.append(
                    f"| {c.step} | `{c.tool}` | {c.args_sha} | {prev} "
                    f"| {c.return_type} | {c.return_len} | {nu} |"
                )
            lines.append("")

        # --- Repeated runs ---
        lines.append("### Repeated tool calls (same args, no new tool between)")
        lines.append("")
        if not self.repeats:
            lines.append("_None detected._")
        else:
            for r in self.repeats:
                anchor = self.anchor(r.rollout, r.steps[0])
                lines.append(
                    f"- **{r.rollout}** `{r.tool}` args_sha={r.args_sha} at steps {r.steps} "
                    f"— {anchor}"
                )
        lines.append("")

        # --- Tool bursts (suspected loop trap) ---
        lines.append("### Tool burst (suspected loop trap — same tool hammered)")
        lines.append("")
        if not self.bursts:
            lines.append("_None detected._")
        else:
            lines.append(
                "_Heuristic: a single tool called >=20 times in a rollout OR "
                ">=10 times concentrated in one step. The args may differ "
                "between calls (e.g. iterating URLs) so this is the loop "
                "pattern the consecutive-same-args repeat detector misses. "
                "If a newly-shipped tool appears here, treat it as evidence "
                "the candidate enabled budget burn rather than progress._"
            )
            lines.append("")
            for b in self.bursts:
                anchor = self.anchor(b.rollout, b.peak_step)
                lines.append(
                    f"- **{b.rollout}** `{b.tool}` total={b.total_calls}, "
                    f"peak={b.max_calls_in_one_step} at step {b.peak_step} "
                    f"(severity={b.severity}) — {anchor}"
                )
        lines.append("")

        # --- Tool-effect missing shortlist ---
        effect_missing = [c for c in self.tool_calls if c.next_uses_result is False]
        lines.append("### Tool calls whose output the next step did NOT reference")
        lines.append("")
        if not effect_missing:
            lines.append("_None detected._")
        else:
            lines.append(
                "_Heuristic: assistant message after the tool call does not contain "
                "any ≥20-char substring from the tool output. This is the single most "
                "common way a tool call 'fires but fails to affect the model' — "
                "e.g. when protocol flattening drops multimodal content._"
            )
            lines.append("")
            for c in effect_missing:
                lines.append(
                    f"- **{c.rollout}** step {c.step} `{c.tool}` "
                    f"(return_type={c.return_type}, len={c.return_len}) — "
                    f"{self.anchor(c.rollout, c.step)}"
                )
        lines.append("")

        return "\n".join(lines)


def _shares_substring(a: str, b: str, min_len: int = 20) -> bool:
    """True iff some length-min_len substring of the shorter string appears
    in the longer one. Step=1 so no window is skipped."""
    if len(a) < min_len or len(b) < min_len:
        return False
    src, tgt = (a, b) if len(a) <= len(b) else (b, a)
    for i in range(len(src) - min_len + 1):
        if src[i : i + min_len] in tgt:
            return True
    return False


def extract_trace_facts(
    task_id: str,
    trajectory_paths: list[Path],
) -> TraceFacts:
    facts = TraceFacts(task_id=task_id)

    for path in trajectory_paths:
        rollout = _rollout_tag(path)
        facts.rollouts.append(rollout)
        facts.trajectory_file_by_rollout[rollout] = path.name

        events: list[dict] = []
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue

        # Build (tool_call_id → tool_name) index and tool_result index.
        id_to_tool: dict[str, str] = {}
        for ev in events:
            if ev.get("type") == "raw_assistant":
                msg = ev.get("message") or {}
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        id_to_tool[tc["id"]] = tc.get("name") or "?"

        # Walk events once, emit ToolCallFact on assistant tool_calls, pair with result.
        # We index events by step so we can look up the next assistant message.
        assistant_events = [e for e in events if e.get("type") == "raw_assistant"]
        tool_result_by_id: dict[str, dict] = {}
        for ev in events:
            if ev.get("type") == "raw_tool":
                msg = ev.get("message") or {}
                tid = msg.get("tool_call_id")
                if tid:
                    tool_result_by_id[tid] = ev

        per_rollout_calls: list[ToolCallFact] = []
        for idx, ev in enumerate(assistant_events):
            msg = ev.get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                continue
            step = int(ev.get("step", idx))
            # Determine the "next assistant text" once per assistant event (all
            # its tool_calls share the same downstream step).
            next_asst_text = ""
            for later in assistant_events[idx + 1 :]:
                lmsg = later.get("message") or {}
                txt, _ = _content_to_text(lmsg.get("content"))
                if txt.strip():
                    next_asst_text = txt
                    break
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tool = tc.get("name") or "?"
                args = tc.get("input") or tc.get("arguments") or {}
                sha = _args_sha(args)
                try:
                    preview = json.dumps(args, ensure_ascii=False, default=str)[:120]
                except Exception:
                    preview = str(args)[:120]
                tid = tc.get("id") or ""
                result_ev = tool_result_by_id.get(tid)
                if result_ev is None:
                    return_type = "missing"
                    return_len = 0
                    next_uses = None
                else:
                    rmsg = result_ev.get("message") or {}
                    rtext, structured = _content_to_text(rmsg.get("content"))
                    return_type = _classify_return(rtext, structured)
                    return_len = len(rtext)
                    if not next_asst_text:
                        next_uses = None
                    elif return_len < 20:
                        next_uses = None  # too short to tell
                    else:
                        next_uses = _shares_substring(rtext, next_asst_text, min_len=20)
                per_rollout_calls.append(
                    ToolCallFact(
                        rollout=rollout,
                        step=step,
                        tool=tool,
                        args_sha=sha,
                        args_preview=preview,
                        return_type=return_type,
                        return_len=return_len,
                        next_uses_result=next_uses,
                    )
                )

        facts.tool_calls.extend(per_rollout_calls)

        # Repeated-run detection: same (tool, args_sha) consecutive with no
        # different tool in between.
        run_tool = None
        run_sha = None
        run_steps: list[int] = []
        for c in per_rollout_calls:
            if c.tool == run_tool and c.args_sha == run_sha:
                run_steps.append(c.step)
            else:
                if run_tool is not None and len(run_steps) >= 2:
                    facts.repeats.append(
                        RepeatRun(rollout=rollout, tool=run_tool, args_sha=run_sha, steps=list(run_steps))
                    )
                run_tool, run_sha, run_steps = c.tool, c.args_sha, [c.step]
        if run_tool is not None and len(run_steps) >= 2:
            facts.repeats.append(
                RepeatRun(rollout=rollout, tool=run_tool, args_sha=run_sha, steps=list(run_steps))
            )

        # Burst detection: same tool called many times in one rollout, even
        # when args differ between calls (so the consecutive-same-args
        # repeat detector above misses it). Catches loop traps where a
        # newly-shipped tool is hammered in a tight retry loop and the
        # surrounding budget gets burned. A trace with >=20 calls of one
        # tool, OR >=10 calls of one tool concentrated in a single step,
        # is reported. Healthy multi-tool traces stay well under both.
        per_tool_total: dict[str, int] = {}
        per_tool_step_max: dict[str, tuple[int, int]] = {}  # (peak_count, peak_step)
        per_tool_step_running: dict[tuple[str, int], int] = {}
        for c in per_rollout_calls:
            per_tool_total[c.tool] = per_tool_total.get(c.tool, 0) + 1
            key = (c.tool, c.step)
            per_tool_step_running[key] = per_tool_step_running.get(key, 0) + 1
            n = per_tool_step_running[key]
            cur = per_tool_step_max.get(c.tool)
            if cur is None or n > cur[0]:
                per_tool_step_max[c.tool] = (n, c.step)
        for tool, total in per_tool_total.items():
            peak, peak_step = per_tool_step_max.get(tool, (0, 0))
            if total < _BURST_TOTAL_MIN and peak < _BURST_PEAK_MIN:
                continue
            severity = "high" if (total >= 30 or peak >= 15) else "medium"
            facts.bursts.append(
                ToolBurst(
                    rollout=rollout,
                    tool=tool,
                    total_calls=total,
                    max_calls_in_one_step=peak,
                    peak_step=peak_step,
                    severity=severity,
                )
            )

        # Exit summary
        end_ev = next((e for e in events if e.get("type") == "episode_end"), None)
        last_asst_text = ""
        for later in reversed(assistant_events):
            txt, _ = _content_to_text((later.get("message") or {}).get("content"))
            if txt.strip():
                last_asst_text = txt[-200:]
                break
        if end_ev is not None:
            facts.exits.append(
                ExitFact(
                    rollout=rollout,
                    exit_reason=str(end_ev.get("exit_reason", "?")),
                    total_steps=int(end_ev.get("total_steps", len(assistant_events))),
                    passed=end_ev.get("passed"),
                    terminal_snippet=last_asst_text,
                )
            )
        else:
            facts.exits.append(
                ExitFact(
                    rollout=rollout,
                    exit_reason="unknown",
                    total_steps=len(assistant_events),
                    passed=None,
                    terminal_snippet=last_asst_text,
                )
            )

    return facts
