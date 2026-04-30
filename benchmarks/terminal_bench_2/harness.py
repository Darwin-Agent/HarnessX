# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""TB2 harness config factory and processor implementations.

Public API
----------
``make_tb2_harness_config(*, timeout_seconds=None) -> HarnessConfig``
    Build a fully serializable ``HarnessConfig`` with no runtime slots.
    Persist it with ``cfg.to_yaml_file(path)`` for meta-harness evolution.

    Runtime-only slots (``sandbox_provider``, ``tracer``) are **not** included
    here — they are injected by the agent at run time via ``cfg.copy(...)``.

All processor classes are kept in this module so that YAML round-trip
deserialization (``HarnessConfig.from_yaml_file``) can resolve their
``_target_`` dotted paths.
"""

from __future__ import annotations

import dataclasses
import re
import time
import uuid

from harnessx.core.builder import HarnessBuilder
from harnessx.core.harness import HarnessConfig
from harnessx.core.events import (
    BeforeModelEvent,
    Message,
    ModelResponseEvent,
    StepStartEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
)
from harnessx.core.processor import MultiHookProcessor
from harnessx.processors.context.env_context_injector import EnvironmentContextInjector
from harnessx.processors.context.strategies.system_prompt.base import (
    BaseSystemPromptBuilder,
)
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.processors.control.bg_install_guard import BgInstallGuard
from harnessx.processors.control.compaction import CompactionProcessor
from harnessx.processors.control.parse_retry import ParseRetryProcessor
from harnessx.processors.control.tool_call_correction import ToolCallCorrectionLayer
from harnessx.tools.builtin import bash_tool
from harnessx.tools.inmemory import InMemoryToolRegistry

from .defaults import OUTPUT_LIMIT, WORKSPACE_PATH

_REDIRECT_WRITE_RE = re.compile(r"(?<![<>2&\d])>{1,2}\s*([^\s;&|><\n'\"]+)")
_SED_INPLACE_RE = re.compile(r"\bsed\s+(?:-[a-zA-Z]*i[a-zA-Z]*|-i\S*)\s+\S+\s+([^\s;&|><\n'\"]+)")
_TEE_WRITE_RE = re.compile(r"\btee\s+(?:-\S+\s+)*([^\s;&|><\n'\"]+)")
_SKIP_PATH_RE = re.compile(r"^(/dev/|/proc/|/sys/|-|\d+$)")


def _extract_written_files(cmd: str) -> list[str]:
    """Return deduplicated file paths that a bash command writes to."""
    if ">" not in cmd and "sed" not in cmd and "tee" not in cmd:
        return []
    paths: set[str] = set()
    for pattern in (_REDIRECT_WRITE_RE, _SED_INPLACE_RE, _TEE_WRITE_RE):
        for m in pattern.finditer(cmd):
            p = m.group(1).strip("'\"")
            if p and not _SKIP_PATH_RE.match(p):
                paths.add(p)
    return list(paths)


_EDIT_LIMIT_WARN = (
    "\n\n[EditDetection] File `{path}` has been modified more than {threshold} times. "
    "You are over-editing this file — step back and try a fundamentally different approach."
)


_SELF_VERIFY_TOOL = "_tb2_self_verify"
_SELF_VERIFY_ACK = "Verification check initiated. See the message above for instructions."
_SELF_VERIFY_MSG = """\
Before finishing, run through this checklist — do not skip any step even if you already verified:

1. **Re-read the task description now.** Does your solution address every requirement, including edge cases, accuracy thresholds, and exact output format?

2. **Check every required output file exists at its exact path:**
```bash
ls -lh /path/to/each/required/output/file
```
A script that ran without errors does NOT guarantee the file was written. Run `ls` explicitly for each required file.

3. **Inspect the actual file contents** — `cat` or `head` each output file and confirm the values are semantically correct, not just that the file exists or is non-empty.

4. **Validate your verification method.** Did your test actually exercise the real behavior? Tests that only check syntax, importability, or exit code 0 on a trivial case are NOT valid — they can pass even on a broken implementation.

5. **For running services:** confirm they are still alive and reachable right now, not just that they started earlier.

Fix anything that looks wrong before exiting. When all checks pass, end your final message with:
**SUCCESS: task complete. Output files confirmed at [list each required path].**\
"""


class CustomSelfVerifyProcessor(MultiHookProcessor):
    """Inject a one-shot verification prompt when the model tries to exit without tool calls.

    Fires at most once per task run.  On the next no-tool-call turn it stays silent.
    """

    _singleton_group = "tb2_self_verify"
    _order = 90

    def __init__(self) -> None:
        self._verified = False
        self._pending_message: str = ""

    async def on_task_start(self, event: TaskStartEvent):
        self._verified = False
        self._pending_message = ""
        yield event

    async def on_before_model(self, event: BeforeModelEvent):
        if not self._pending_message:
            yield event
            return
        msg = self._pending_message
        self._pending_message = ""
        # last message is a tool result (role≠user) → append exactly +1 user
        yield dataclasses.replace(
            event,
            messages=event.messages + (Message(role="user", content=msg),),
        )

    async def on_after_model(self, event: ModelResponseEvent):
        exit_intent = event.finish_reason in ("end_turn", "stop") and not event.tool_calls
        if exit_intent and not self._verified:
            self._verified = True
            self._pending_message = _SELF_VERIFY_MSG
            keepalive = ToolCall(
                id=f"sv-{uuid.uuid4().hex[:8]}",
                name=_SELF_VERIFY_TOOL,
                input={},
            )
            yield dataclasses.replace(event, tool_calls=(keepalive,))
        else:
            yield event

    async def on_before_tool(self, event: ToolCallEvent):
        if event.tool_name == _SELF_VERIFY_TOOL:
            yield dataclasses.replace(event, approved=False, synthetic_result=_SELF_VERIFY_ACK)
        else:
            yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._verified = False
        self._pending_message = ""
        yield event


class CustomEditToolProcessor(MultiHookProcessor):
    """Detect repeated Bash-based file edits and nudge the model to try a different approach.

    TB2 only exposes Bash, so write operations are detected via command patterns
    (redirects, sed -i, tee).  When the same file is edited more than
    ``threshold`` times, a warning is appended to the tool result and the
    per-file counter is reset.
    """

    _singleton_group = "bash_edit_detector"
    _order = 30

    def __init__(self, threshold: int = 7) -> None:
        self.threshold = threshold
        self._edit_counts: dict[str, int] = {}
        self._pending: dict[str, list[str]] = {}

    async def on_task_start(self, event: TaskStartEvent):
        self._edit_counts.clear()
        self._pending.clear()
        yield event

    async def on_before_tool(self, event: ToolCallEvent):
        if event.tool_name == "Bash":
            paths = _extract_written_files(event.tool_input.get("command", ""))
            if paths:
                self._pending[event.tool_call_id] = paths
        yield event

    async def on_after_tool(self, event: ToolResultEvent):
        paths = self._pending.pop(event.tool_call_id, [])
        injections: list[str] = []
        for path in paths:
            count = self._edit_counts.get(path, 0) + 1
            self._edit_counts[path] = count
            if count > self.threshold:
                injections.append(_EDIT_LIMIT_WARN.format(path=path, threshold=self.threshold))
                self._edit_counts[path] = 0
        if injections:
            yield dataclasses.replace(event, result=(event.result or "") + "".join(injections))
        else:
            yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._edit_counts.clear()
        self._pending.clear()
        yield event


# ---------------------------------------------------------------------------
# TaskTimeReminderProcessor
# ---------------------------------------------------------------------------

_TASK_TIME_WARN_70 = (
    "[TaskTimeReminder] You have used 70% of your time budget ({elapsed:.0f}s / {total:.0f}s). "
    "Stop exploring. Focus exclusively on implementing and writing all required output files. "
    "Run `ls -lh` on each expected output path to confirm they exist."
)
_TASK_TIME_WARN_90 = (
    "[TaskTimeReminder] CRITICAL: You have used 90% of your time budget ({elapsed:.0f}s / {total:.0f}s, "
    "{remaining:.0f}s left). "
    "If you have NOT yet written all required output files, do so IMMEDIATELY — "
    "the task will be terminated soon and any missing output files will cause an automatic score of 0. "
    "Do NOT start any new implementation. Write and verify output files NOW."
)


class TaskTimeReminderProcessor(MultiHookProcessor):
    _singleton_group = "task_time_reminder"
    _order = 6

    def __init__(
        self,
        timeout_seconds: float | None = None,
        warn_at: tuple[float, ...] = (0.70, 0.90),
    ) -> None:
        self._timeout = timeout_seconds
        self._warn_at = sorted(warn_at)
        self._triggered: set[float] = set()
        self._start: float = 0.0

    async def on_task_start(self, event: TaskStartEvent):
        self._triggered = set()
        self._start = time.monotonic()
        yield event

    async def on_step_start(self, event: StepStartEvent):
        if not self._timeout or self._timeout <= 0:
            yield event
            return

        elapsed = time.monotonic() - self._start
        pct = elapsed / self._timeout
        remaining = self._timeout - elapsed

        warn: str | None = None
        for threshold in reversed(self._warn_at):
            if pct >= threshold and threshold not in self._triggered:
                self._triggered.add(threshold)
                if threshold >= 0.90:
                    warn = _TASK_TIME_WARN_90.format(elapsed=elapsed, total=self._timeout, remaining=remaining)
                else:
                    warn = _TASK_TIME_WARN_70.format(elapsed=elapsed, total=self._timeout)
                break

        if warn:
            msg = Message(role="user", content=warn)
            yield dataclasses.replace(
                event,
                messages=event.messages + (msg,),
                raw_messages=event.raw_messages + (msg,),
            )
        else:
            yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._triggered = set()
        yield event


# ---------------------------------------------------------------------------
# PostCompactionRefreshProcessor
# ---------------------------------------------------------------------------


class PostCompactionRefreshProcessor(MultiHookProcessor):
    """After context compaction, re-inject a fresh workspace snapshot from the
    sandbox so the model knows what files currently exist rather than relying on
    compressed-away history.

    Detects compaction by a significant drop in message count between steps
    (same heuristic used by LoopDetectionProcessor).
    """

    _singleton_group = "tb2_post_compaction_refresh"
    _order = 11  # runs after CompactionProcessor (8)

    def __init__(self, drop_threshold: int = 5) -> None:
        self._drop_threshold = drop_threshold
        self._prev_count: int = 0

    async def on_task_start(self, event: TaskStartEvent):
        self._prev_count = 0
        yield event

    async def on_step_start(self, event: StepStartEvent):
        current_count = len(event.messages)
        prev = self._prev_count
        self._prev_count = current_count

        if not (prev > 0 and (prev - current_count) >= self._drop_threshold):
            yield event
            return

        from harnessx.sandbox.base import get_current_sandbox

        sandbox = get_current_sandbox()
        if sandbox is None:
            yield event
            return

        try:
            ls_out = await sandbox.exec("ls -la /app 2>/dev/null | head -30", timeout=8)
        except Exception:
            yield event
            return

        procs_out = ""
        try:
            procs_out = await sandbox.exec(
                "ps aux --no-headers 2>/dev/null | grep -v '\\[' "
                "| awk '{print $11}' | sort -u | grep -v '^ps$' | head -8",
                timeout=5,
            )
        except Exception:
            pass

        snapshot = f"[PostCompaction] Context was compressed. Current workspace state:\n```\n{ls_out.strip()}\n```"
        if procs_out and procs_out.strip():
            snapshot += f"\nRunning processes: {procs_out.strip()}"

        msg = Message(role="user", content=snapshot)
        yield dataclasses.replace(
            event,
            messages=event.messages + (msg,),
            raw_messages=event.raw_messages + (msg,),
        )

    async def on_task_end(self, event: TaskEndEvent):
        self._prev_count = 0
        yield event


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_TB2_SYSTEM = """\
# System
You are an expert software engineer solving terminal-based programming tasks in a Linux sandbox.

## Key Facts About This Environment
- You are inside an isolated Docker container.
- Your job is to implement the task correctly so the external verifier passes when it runs.

## Workflow
1. **Read the task carefully.** Before writing any code, understand:
   - Required output format, file paths, field names, CLI flags
   - Accuracy or size constraints
   - What a correct solution looks like from the outside
2. **Explore the environment first.**
```bash
   ls /app && find /app -name '*.md' -o -name '*.txt' -o -name '*.sh' 2>/dev/null | head -20
```
3. **For complex tasks, decompose before implementing.** Break the problem into concrete subtasks. Solve and verify each subtask before moving to the next — do not attempt everything at once.
4. **Implement the solution.** Match the task's specified interface exactly — file paths, output format, and CLI behavior all matter to the verifier.
5. **Write outputs to the exact paths the task specifies.** Re-read the task description to confirm paths before writing anything.
6. **Verify outputs explicitly** — do not assume a file was written just because a script ran without errors. Check with `ls` and inspect the contents:
```bash
   ls -lh /path/to/required/output   # confirm file exists and is non-empty
   head -20 /path/to/required/output  # confirm content is correct
```
7. **For background services**, keep them running after you exit and verify they are reachable:
```bash
   nohup python server.py > /tmp/server.log 2>&1 &
   sleep 1 && curl -sf http://localhost:8080/health
```
8. **Do not use unbounded background commands.** If a command's runtime is unpredictable, run it synchronously. Only use `&` / `nohup` for long-running services the verifier will connect to.
9. **Do not run `apt-get update`.** Package lists are pre-cached — `apt-get install -y <pkg>` works directly and completes in seconds.

## Definition of Done
- All required output files exist at the exact paths specified in the task — confirmed with `ls`
- Any required services are running and reachable
- You have inspected the actual output values and confirmed they are correct
- No temporary or debug files remain that could interfere with the verifier

When all checks pass, end your final message with:
**SUCCESS: task complete. Output files confirmed at [list each required path].**
"""


class _StaticSystemPromptBuilder(BaseSystemPromptBuilder):
    """Minimal TB2-specific system prompt — returns a fixed base identity prompt."""

    async def build(self, workspace=None) -> str:  # noqa: ARG002
        return _TB2_SYSTEM


def _build_tb2_tools() -> InMemoryToolRegistry:
    """TB2 tool set: Bash only — matches the official Harbor evaluation."""
    registry = InMemoryToolRegistry()
    registry.register(bash_tool)
    return registry


def make_tb2_harness_config(
    *,
    timeout_seconds: int | None = None,
    workspace_path: str = WORKSPACE_PATH,
    output_limit: int = OUTPUT_LIMIT,
) -> HarnessConfig:
    """Build a serializable ``HarnessConfig`` for Terminal Bench 2.0.

    This config contains only the processor pipeline and tool registry — no
    runtime-only slots (``sandbox_provider``, ``tracer``).  It can be saved to
    YAML via ``cfg.to_yaml_file(path)`` and evolved by the meta-harness.

    The agent injects the runtime slots at run time::

        base = make_tb2_harness_config(timeout_seconds=task_timeout)
        harness_config = base.copy(
            sandbox_provider=HarborSandboxProvider(environment, ...),
            tracer=HarnessJournal(...),
        )
        harness = ModelConfig(main=provider).agentic(harness_config)

    Args:
        timeout_seconds:  Task timeout disclosed to the model via
                          ``EnvironmentContextInjector`` and
                          ``TaskTimeReminderProcessor``.
        workspace_path:   Absolute path inside the container where task files live.
        output_limit:     Maximum characters captured per tool call (passed
                          through to ``HarborSandboxProvider`` at run time via
                          the agent; stored here for reference only).
    """
    _ = output_limit  # documented here; consumed by HarborSandboxProvider in agent.py

    return (
        HarnessBuilder()
        .add(SystemPromptProcessor(_StaticSystemPromptBuilder()), order=1)
        .add(
            EnvironmentContextInjector(
                working_dir=workspace_path,
                timeout_seconds=timeout_seconds,
                show_project_dir=False,
            ),
            order=2,
        )
        .add(TaskTimeReminderProcessor(timeout_seconds=timeout_seconds))
        .add(CompactionProcessor())
        .add(PostCompactionRefreshProcessor())
        .add(ToolCallCorrectionLayer())
        .add(ParseRetryProcessor())
        .add(BgInstallGuard())
        .add(CustomEditToolProcessor())
        .add(CustomSelfVerifyProcessor())
        .slot(tool_registry=_build_tb2_tools())
    ).build()
