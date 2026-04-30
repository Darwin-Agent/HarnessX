# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import re
import uuid
from pathlib import Path

from ...core.events import (
    BeforeModelEvent,
    Message,
    ModelResponseEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
)
from ...core.processor import MultiHookProcessor
from .._sp_utils import sp_append

_SYNTHETIC_TOOL = "_verify_keepalive"

# Patterns in the Bash *command* that indicate a test runner was invoked.
# Anchored to common invocation forms to reduce false positives.
_TEST_CMD_RE = re.compile(
    r"pytest|py\.test"  # Python pytest (any invocation)
    r"|python\s.*\btest"  # python -m pytest / python test_foo.py
    r"|uv\s+run\s+pytest|poetry\s+run\s+pytest"
    r"|make\s+test|make\s+check"
    r"|npm\s+(run\s+)?test|yarn\s+(run\s+)?test|pnpm\s+(run\s+)?test"
    r"|cargo\s+test"
    r"|go\s+test"
    r"|rspec|bundle\s+exec\s+rspec"
    r"|mocha|jest|vitest"
    r"|\.\/test\.sh|bash\s+test|sh\s+test",
    re.IGNORECASE,
)

# Patterns in the Bash *result* that confirm at least some tests ran and
# reported an outcome (pass or fail).  We want evidence of execution, not
# just invocation (e.g. import errors abort before any test runs).
_TEST_RESULT_RE = re.compile(
    r"\d+\s+passed"  # pytest: "3 passed"
    r"|\d+\s+failed"  # pytest: "2 failed"
    r"|OK\s*\(\d+\s+test"  # unittest: "OK (3 tests)"
    r"|FAILED\s*\(\w+=\d+"  # unittest: "FAILED (failures=1)"
    r"|Tests:\s+\d+"  # Jest/Vitest summary
    r"|passing|failing"  # Mocha
    r"|test result:\s+(ok|FAILED)"  # Cargo
    r"|ok\s+\d+\s+\-\-\s+PASS",  # Go test
    re.IGNORECASE,
)

_PROACTIVE = (
    "\n\n## Verification Requirements\n"
    "Before declaring the task complete, you MUST:\n"
    "1. Run all available tests (e.g. `pytest`, `make test`, `./test.sh`) and confirm they pass.\n"
    "2. Verify your solution against every requirement in the task description.\n"
    "3. Check that no TODOs, placeholders, or stub implementations remain."
)

_VERIFY_REASONING_BOOST = (
    "[ReasoningBudget:high] You are in the verification phase. "
    "Reason carefully and thoroughly — re-read the task requirements, check each "
    "deliverable, and confirm your solution is complete and correct before finishing. "
    "Do not ask the user for a new task; verify against the current conversation.\n\n"
)

_KEEPALIVE_ACK = "Verification check initiated. See the message above for what to verify."

_MSG_NO_TEST = """\
Before finishing, you have not run any tests yet.

First, check whether an official test suite exists:
  find / -maxdepth 4 -name 'test_*.py' -o -name '*_test.py' -o -name 'test.sh' -o -name 'pytest.ini' 2>/dev/null | grep -v __pycache__ | head -10

- If tests are found: run them (`python -m pytest <test_dir>/ -x -q` or `bash <test_dir>/test.sh`) and fix all failures before finishing.
- If no tests are found: the verifier runs separately — verify your output files exist at the exact paths the task requires, match the expected format, and that any required services are still running.\
"""

_MSG_CHECKLIST = """\
Please do a final check before finishing:

[ ] All required files have been created / modified correctly.
[ ] Tests pass — run `pytest` / `make test` / `./test.sh` to confirm.
[ ] No TODOs, placeholders, or stub implementations remain.
[ ] Your solution handles edge cases and matches the exact interface specified.

If everything is correct, provide a brief summary of what was done.\
"""


class SelfVerifyProcessor(MultiHookProcessor):
    """Intercept task exit and inject a user-turn verification message.

    Tracks:
    - ``_write_count``  — incremented on every Write/Edit tool call.
    - ``_test_ran``     — set to ``True`` when a Bash command matching a test
                          runner pattern is detected in ``on_before_tool``.

    Only fires when at least one file has been written (``_write_count > 0``).
    Pure Q&A / research tasks with no file writes exit immediately without
    triggering verification.

    On exit intent (with writes):
    1. A synthetic keep-alive tool call is injected to force one more model step.
    2. A verification user message (chosen based on activity state) is queued.
    3. On the next ``step_start``, the message is appended to ``event.messages``
       so the model reads it as a human reviewer asking for confirmation.

    Args:
        enabled: Set to ``False`` to disable without removing from the builder.
    """

    _singleton_group = "self_verify"
    _order = 90

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._verified = False  # True once verification has been triggered; never reset mid-task
        self._pending_user_message: str = ""
        self._write_count: int = 0
        self._test_ran: bool = False
        self._agent_home: str = str(Path.home() / ".harnessx")
        # Tracks the last Bash tool_call_id that matched a test command so we
        # can confirm execution by checking the corresponding tool result.
        self._pending_test_call_id: str = ""

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    async def on_task_start(self, event: TaskStartEvent):
        """Append the proactive verification instruction to the system prompt once."""
        if not self.enabled:
            yield event
            return
        yield dataclasses.replace(event, system_prompt=sp_append(event.system_prompt, _PROACTIVE))

    async def on_before_model(self, event: BeforeModelEvent):
        """Inject the pending verification user message, if any."""
        if not self.enabled or not self._pending_user_message:
            yield event
            return
        # _VERIFY_REASONING_BOOST is prepended to the message — dynamic boosts
        # travel as user messages, never via system-prompt mutation.
        content = _VERIFY_REASONING_BOOST + self._pending_user_message
        self._pending_user_message = ""
        yield dataclasses.replace(
            event,
            messages=event.messages + (Message(role="user", content=content),),
        )

    async def on_before_tool(self, event: ToolCallEvent):
        if event.tool_name in ("Write", "Edit"):
            path = str(event.tool_input.get("file_path", "") or event.tool_input.get("path", ""))
            if self._should_count_write(path):
                self._write_count += 1

        # Stage test-command call_id so on_after_tool can confirm the result
        if event.tool_name == "Bash":
            cmd = event.tool_input.get("command", "")
            if _TEST_CMD_RE.search(cmd):
                self._pending_test_call_id = event.tool_call_id

        # Intercept the synthetic keep-alive call; return a minimal ack
        if event.tool_name == _SYNTHETIC_TOOL:
            yield dataclasses.replace(
                event,
                approved=False,
                synthetic_result=_KEEPALIVE_ACK,
            )
        else:
            yield event

    async def on_after_tool(self, event: ToolResultEvent):
        # Confirm test execution: command matched AND result contains test output
        if (
            event.tool_call_id == self._pending_test_call_id
            and self._pending_test_call_id
            and _TEST_RESULT_RE.search(event.result or "")
        ):
            self._test_ran = True
            self._pending_test_call_id = ""

        yield event

    async def on_after_model(self, event: ModelResponseEvent):
        if not self.enabled:
            yield event
            return

        exit_intent = event.finish_reason in ("end_turn", "stop") and not event.tool_calls

        if exit_intent and not self._verified and self._write_count > 0:
            self._verified = True  # gate: at most one verification pass per task run

            # Choose verification message based on activity state
            if not self._test_ran:
                msg = _MSG_NO_TEST
            else:
                msg = _MSG_CHECKLIST

            self._pending_user_message = msg

            # Inject keep-alive tool call so the RunLoop executes one more step,
            # during which on_step_start will deliver the user message above.
            keepalive = ToolCall(
                id=f"kv-{uuid.uuid4().hex[:8]}",
                name=_SYNTHETIC_TOOL,
                input={},
            )
            yield dataclasses.replace(event, tool_calls=(keepalive,))

        else:
            yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._verified = False
        self._pending_user_message = ""
        self._write_count = 0
        self._test_ran = False
        self._pending_test_call_id = ""
        yield event

    def _should_count_write(self, file_path: str) -> bool:
        """Count only project-relevant writes for verification gating.

        Writes under AGENT_HOME metadata (memory/session internals) should not
        trigger test-verification turns; they are operational state updates, not
        user deliverables.
        """
        if not file_path:
            return True
        try:
            p = str(Path(file_path).expanduser().resolve())
        except Exception:
            p = file_path
        norm = p.replace("\\", "/")
        home_norm = self._agent_home.replace("\\", "/")

        if norm.startswith(f"{home_norm}/memory/"):
            return False
        if norm.startswith(f"{home_norm}/workspaces/"):
            return False

        # Default: count as a user-deliverable write.
        return True
