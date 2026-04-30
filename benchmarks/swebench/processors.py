"""SWE-bench specific processors for HarnessX."""

from __future__ import annotations

import dataclasses
import re
from typing import AsyncIterator

from harnessx.core.events import (
    Message,
    ModelResponseEvent,
    StepStartEvent,
    StepEndEvent,
    ToolCall,
)
from harnessx.core.processor import MultiHookProcessor


# Patterns that indicate the model is giving up instead of trying
_APOLOGY_PATTERNS = re.compile(
    r"(?:i'?m sorry|i cannot|i can'?t|unable to|i apologize|"
    r"i don'?t have|beyond my|not possible for me|"
    r"i wasn'?t able|unfortunately.*cannot|"
    r"i need more context|let me know if you)",
    re.IGNORECASE,
)


class SWEBenchWorkflowProcessor(MultiHookProcessor):
    """Nudge the model to follow the LOCATE -> FIX -> VERIFY workflow.

    Hooks: step_start, step_end, after_model

    Key behaviors:
    - Step-based nudges: push model to edit code and run git diff
    - Anti-apology: if model tries to give up, override with continuation
    - Progress tracking: detect edits and git diff output

    Args:
        max_steps: Maximum steps for the task (used to compute thresholds).
        repo_dir: Path to the repository (used in nudge messages).
    """

    _singleton_group = "swebench_workflow"
    _order = 5  # After SystemPromptProcessor (1), before TokenBudget (10)

    def __init__(self, max_steps: int = 40, repo_dir: str = "/tmp/swe_repo"):
        self.max_steps = max_steps
        self.repo_dir = repo_dir
        self._has_edit = False
        self._has_git_diff = False
        self._step_count = 0
        self._apology_count = 0

    async def on_step_start(self, event: StepStartEvent) -> AsyncIterator[StepStartEvent]:
        step = event.step_id
        self._step_count = step
        messages = list(event.messages)

        # Scan existing messages to track progress
        self._scan_messages(event.raw_messages)

        nudge = None

        # Halfway point: if no edits yet, nudge to start fixing
        halfway = self.max_steps // 2
        if step == halfway and not self._has_edit:
            nudge = (
                "[SYSTEM NOTICE] You are halfway through your allowed steps. "
                "You have NOT made any code changes yet. "
                "Stop exploring and start fixing the bug NOW using the Edit tool. "
                "Then run `cd {repo} && git diff` to show your changes."
            ).format(repo=self.repo_dir)

        # Early nudge: step 4+ without edits
        elif step >= 4 and not self._has_edit and step < halfway:
            nudge = (
                "[SYSTEM NOTICE] You have spent {n} steps exploring. Start making code changes NOW using the Edit tool."
            ).format(n=step)

        # Near end: demand git diff
        elif step >= self.max_steps - 5 and self._has_edit and not self._has_git_diff:
            nudge = (
                "[SYSTEM NOTICE] You are running out of steps. Run `cd {repo} && git diff` NOW to output your changes."
            ).format(repo=self.repo_dir)

        # Very near end: final warning
        elif step >= self.max_steps - 2:
            if not self._has_edit:
                nudge = (
                    "[SYSTEM NOTICE] FINAL WARNING: You have {left} steps left. "
                    "Make your best fix attempt with Edit, then run "
                    "`cd {repo} && git diff`."
                ).format(left=self.max_steps - step, repo=self.repo_dir)
            elif not self._has_git_diff:
                nudge = ("[SYSTEM NOTICE] FINAL STEP: Run `cd {repo} && git diff` NOW.").format(repo=self.repo_dir)

        if nudge:
            messages.append(Message(role="user", content=nudge))
            yield dataclasses.replace(event, messages=tuple(messages))
        else:
            yield event

    async def on_after_model(self, event: ModelResponseEvent) -> AsyncIterator[ModelResponseEvent]:
        """Detect and override premature stops and apology/refusal responses.

        Two cases:
        1. Model stops without tool calls AND hasn't edited anything yet
           -> Force it to continue (regardless of text content)
        2. Model responds with apology patterns
           -> Override with continuation prompt
        """
        content = event.content or ""
        is_stopping = not event.tool_calls and event.finish_reason in (
            "end_turn",
            "stop",
            None,
        )

        # Case 1: Premature stop — model trying to finish without any edits
        if (
            is_stopping and not self._has_edit and self._apology_count < 3  # Don't loop forever
        ):
            self._apology_count += 1
            override_content = (
                f"{content}\n\n"
                f"[SYSTEM OVERRIDE] You have NOT made any code changes yet. "
                f"You MUST edit at least one file before finishing. "
                f"Search for relevant code in {self.repo_dir}, "
                f"then use the Edit tool to fix it, then run "
                f"`cd {self.repo_dir} && git diff`."
            )
            # Vary the synthetic commands to avoid loop detection
            _explore_cmds = [
                f"cd {self.repo_dir} && find . -maxdepth 2 -name '*.py' -not -path './.git/*' | head -30",
                f"cd {self.repo_dir} && ls -la",
                f"cd {self.repo_dir} && find . -maxdepth 3 -name '*.py' -not -path './.git/*' | wc -l",
            ]
            cmd = _explore_cmds[min(self._apology_count - 1, len(_explore_cmds) - 1)]
            yield dataclasses.replace(
                event,
                content=override_content,
                finish_reason="tool_use",
                tool_calls=(
                    ToolCall(
                        id=f"synthetic_{self._step_count}_{self._apology_count}",
                        name="Bash",
                        input={"command": cmd},
                    ),
                ),
            )
            return

        # Case 2: Has edits but no git diff, and trying to stop
        if is_stopping and self._has_edit and not self._has_git_diff and self._apology_count < 5:
            self._apology_count += 1
            override_content = (
                f"{content}\n\n"
                f"[SYSTEM OVERRIDE] You have made edits but haven't run `git diff`. "
                f"Run `cd {self.repo_dir} && git diff` NOW to output your changes."
            )
            yield dataclasses.replace(
                event,
                content=override_content,
                finish_reason="tool_use",
                tool_calls=(
                    ToolCall(
                        id=f"synthetic_diff_{self._step_count}",
                        name="Bash",
                        input={"command": f"cd {self.repo_dir} && git diff"},
                    ),
                ),
            )
            return

        yield event

    async def on_step_end(self, event: StepEndEvent) -> AsyncIterator[StepEndEvent]:
        # Track tool calls from the step summary
        summary = event.tool_call_summary or ""
        if "Edit:" in summary or "edit_tool:" in summary:
            # Note: we also verify from messages that edit succeeded (not just called)
            pass
        if "git diff" in summary.lower():
            self._has_git_diff = True
        yield event

    def _scan_messages(self, messages: tuple[Message, ...]) -> None:
        """Scan raw messages to detect if edits or git diff have occurred.

        Important: we track edit SUCCESS, not just edit calls.
        An Edit tool call followed by an error result doesn't count.
        """
        prev_was_edit = False
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Check for Edit tool calls
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.name in ("Edit", "edit_tool"):
                        prev_was_edit = True
                    if tc.name in ("Bash", "bash_tool") and "git diff" in str(tc.input):
                        self._has_git_diff = True

            # Check tool results: did the edit succeed?
            if msg.role == "tool":
                if prev_was_edit:
                    # Edit failed if result contains specific failure messages
                    lower = content.lower()
                    edit_failed = (
                        "old_string not found" in lower
                        or "no match found" in lower
                        or lower.startswith("error:")
                        or "file not found" in lower
                        or not content.strip()
                    )
                    if not edit_failed:
                        self._has_edit = True
                    prev_was_edit = False

                # Check for diff output
                if "diff --git" in content:
                    self._has_git_diff = True
