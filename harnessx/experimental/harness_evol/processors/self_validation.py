"""
SelfValidationProcessor — exit-intent interceptor for completion-gated agents.

Fires when the model tries to end the task (finish_reason=end_turn/stop, no tool calls).
Runs through up to five phases before allowing the run to end:

  Phase 1 — Auto-check (optional):
    Calls config_validator(required_file) programmatically.  Catches import
    errors, instantiation failures, and basic runtime issues before the agent
    submits.  If the validator returns an error string, it is shown to the agent
    so it can fix the config.

  Phase 2 — Leakage / overfitting self-check:
    After auto-check passes, injects a targeted prompt listing every new
    processor file in the workspace and asking the agent to read each one and
    confirm it contains NO task-specific overfitting or answer leakage of any
    kind: task_description branching, hardcoded task identities, hardcoded
    domain constants, conditional shortcuts that bypass genuine reasoning,
    or statistical fingerprinting to identify and hint at specific tasks.
    The agent must output ``leakage_pass_marker`` (default "[LEAKAGE CLEAN]")
    to proceed.

  Phase 3 — Regression self-check:
    After leakage passes, injects a prompt asking the agent to read
    ``signals/all_tasks_summary.json`` from its workspace parent directory,
    locate the ``failure_pattern_clusters.all_pass`` list, and verify that
    any injected guidance is universally applicable — not domain-specific
    content that adds noise for tasks outside the targeted failure pattern.
    The check is generic: it asks whether the guidance is coherent for ALL
    tasks, not whether it will break any specific task by name.  If the
    signals file does not exist or the all_pass list is empty, the agent
    can confirm immediately.  The agent must output ``regression_pass_marker``
    (default "[REGRESSION CLEAN]") to proceed.

  Phase 4 — Agent self-review:
    Injects a structured review prompt asking the agent to check design quality
    and generalization.  The agent must output ``review_pass_marker``
    (default "[CHECK PASS]") to confirm it is satisfied.
    Until the marker is seen, the processor keeps intercepting exit attempts.

  Phase 5 — Pre-submission policy gate (optional, requires pre_submission_validator):
    When the agent calls ``completion_tool`` and all prior checks have passed,
    extracts the ``change_manifest`` from the tool call input and runs
    ``pre_submission_validator(target_config_path, manifest)``.  The validator
    should cover all remaining policy checks (param_drift, diff_not_empty,
    evidence_completeness, model_gap_filter, new_processors_registered,
    pattern_coverage, etc.).  Any failure is returned as an actionable error
    message so the agent can fix it and resubmit.

  Phase 6 — Completion tool:
    Once all checks pass, allows the agent to call ``completion_tool``.

Intervention mechanism:
  on_after_model  — detects exit intent AND scans content for pass markers.
                    Markers are only accepted AFTER the corresponding prompt has
                    been delivered (gated by _leakage_prompt_delivered /
                    _review_prompt_delivered flags) to prevent accidental bypass
                    from the model mentioning marker strings in discussion text.
                    Injects a keepalive fake tool call to prevent the runloop from
                    breaking.  Fires at most ``max_interventions`` times total.
  on_before_tool  — (a) short-circuits the fake keepalive tool.
                    (b) blocks completion_tool if files are missing, auto-check
                        has not passed, or self-review has not been confirmed.
  on_before_model — delivers the pending remediation message as a user turn.

Parameters
----------
completion_tool:
    Name of the tool the agent MUST call to complete the task.
required_files:
    Absolute paths that must exist before the task is considered done.
config_validator:
    Optional async callable ``(path: Path) -> str | None``.  Called with the
    first required_file when files are present and auto-check has not yet
    passed.  Returns None on success or an error string for the agent to fix.
    When None, Phase 1 is skipped.
leakage_pass_marker:
    String the agent must include to confirm leakage self-check passed.
    Default: "[LEAKAGE CLEAN]".
regression_pass_marker:
    String the agent must include to confirm regression self-check passed.
    Default: "[REGRESSION CLEAN]".
review_pass_marker:
    String the agent must include in a response to confirm self-review passed.
    Default: "[CHECK PASS]".
pre_submission_validator:
    Optional async callable ``(target: Path, manifest: dict) -> str | None``.
    Called just before the completion tool is allowed to proceed.  Should run
    all remaining policy checks (param_drift, diff_not_empty, evidence
    completeness, model_gap_filter, new_processors_registered, pattern_coverage,
    etc.).  Returns None on success or a human-readable error string for the
    agent to fix.  When None, Phase 5 is skipped.
max_interventions:
    Maximum exit-intent interceptions per task run (default 8).
"""
from __future__ import annotations

import dataclasses
import logging
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from harnessx.core.processor import MultiHookProcessor
from harnessx.core.events import (
    BeforeModelEvent,
    Message,
    ModelResponseEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCall,
    ToolCallEvent,
)

logger = logging.getLogger(__name__)

_FAKE_TOOL_NAME = "_self_validation_check"
_FAKE_TOOL_ACK = "Self-validation check initiated. See the instructions above."

_LEAKAGE_CHECK_PROMPT = """\
✅ Auto-check passed. Before the final design review, perform a **leakage / overfitting \
self-check** on every new processor file you wrote:

{file_list}

For each file above, read its full source and check ALL of the following:

1. **Task-description branching** — Does the processor read `task_description` / \
`event.task_description` / `task.description` and branch on its content \
(keyword matching, regex, substring checks) to decide what to inject or how to behave?
   — If YES: **benchmark overfitting**. Replace with a generic strategy that applies \
identically to all tasks.

2. **Hardcoded task identities** — Does the processor contain literal task names, \
task IDs, expected answers, or any string that only makes sense for a specific \
benchmark task?
   — If YES: strip them entirely.

3. **Hardcoded domain constants** — Does the processor embed concrete values that \
came from knowledge of what the benchmark tasks look like (e.g. specific numbers, \
strings, file paths, or identifiers that only appear in certain tasks)?
   — If YES: strip them. A processor must not encode answers or domain knowledge.

4. **Conditional shortcuts / cheats** — Does the processor detect task properties \
(category, difficulty, tool list, etc.) and take a shortcut that bypasses genuine \
reasoning — effectively handing the agent a pre-solved answer or a task-specific hint?
   — If YES: this is answer leakage. Remove the shortcut logic.

5. **Statistical fingerprinting** — Does the processor try to identify the current \
task from patterns in previous outputs, tool results, or trajectory history, and \
then inject task-specific guidance?
   — If YES: overfitting via indirect task identification. Remove the fingerprinting.

A processor MUST be **fully task-agnostic**: its behaviour must be identical \
regardless of which specific task is running. Generic improvements to reasoning \
strategy, tool usage, or error recovery are fine; anything that encodes knowledge \
of specific tasks or their answers is not.

If all files are clean (or there are no new processor files), end your response \
with exactly: **{marker}**

Otherwise fix every issue found first, then output the marker.\
"""

_REGRESSION_CHECK_PROMPT = """\
✅ Leakage check confirmed. Before the final design review, check whether your \
changes could regress previously-stable tasks.

**Step 1 — identify stable tasks.**
Read the file `signals/all_tasks_summary.json` in the directory one level above \
your workspace (i.e. `../signals/all_tasks_summary.json` relative to your workspace root). \
Look at `failure_pattern_clusters.all_pass` — this is the list of tasks that passed \
**all rollouts** with the current baseline config.

If the file does not exist, or `all_pass` is empty, there are no stable tasks to protect. \
Skip to the marker immediately: **{marker}**

**Step 2 — for each new processor file you wrote, answer:**

1. **Unconditional injection scope** — Does any hint, guidance text, or behaviour you \
inject apply only to a specific task domain (e.g. distributed training, hash cracking, \
dataset processing), yet is injected for ALL tasks regardless of context?
   — If YES: either gate the hint behind a runtime condition, remove it, or rewrite it \
as generic advice that is coherent and non-confusing for tasks in the all_pass list.

2. **Parameter tightening** — If you reduced a threshold or window (e.g. compaction \
retention, budget guard limit), could the new value starve tasks that were passing with \
the old, more generous setting?
   — If YES: revert the parameter or add a safety margin so stable tasks are not \
negatively affected.

3. **Processor interactions** — Does your change suppress or override the signal of \
another active processor in a way that could remove a guard that the stable tasks \
depended on?
   — If YES: restore the guard or narrow the override so it only fires for the \
targeted failure pattern.

**Key principle**: reason about whether the guidance category is universally applicable. \
You do NOT need to simulate each task by name — ask whether an agent working on any \
task in the all_pass list would receive guidance that is at least as coherent and \
non-distracting as before.

If all changes are safe (or there are no new processor files), \
end your response with exactly: **{marker}**

Otherwise fix the issues first, then output the marker.\
"""

_SELF_REVIEW_PROMPT = """\
✅ Regression check confirmed. Before calling `{completion_tool}`, perform a final design review:

1. **Re-read the digest report** failure patterns and your config changes. \
Does each change directly address a specific pattern with clear evidence?

2. **Check processor logic**: For any new or modified processor, does its detection \
signal precisely match the failure pattern — or is it an over-broad heuristic that \
could interfere with correct behavior?

3. **Check parameter values**: Are thresholds, window sizes, and other numeric \
parameters justified by evidence from the digest, or were they guessed?

If all checks pass, end your response with exactly: **{marker}** — then \
immediately call `{completion_tool}` to complete the task.

Otherwise, fix the issues you found, then output the marker and call `{completion_tool}`.\
"""


class SelfValidationProcessor(MultiHookProcessor):
    """
    Six-phase exit-intent interceptor:
    auto-check → leakage → regression → review → pre-submission policy gate → submit.
    """

    def __init__(
        self,
        completion_tool: str,
        required_files: list[str] | None = None,
        config_validator: Callable[[Path], Awaitable[str | None]] | None = None,
        leakage_pass_marker: str = "[LEAKAGE CLEAN]",
        regression_pass_marker: str = "[REGRESSION CLEAN]",
        review_pass_marker: str = "[CHECK PASS]",
        pre_submission_validator: Callable[[Path, dict], Awaitable[str | None]] | None = None,
        max_interventions: int = 8,
    ) -> None:
        self.completion_tool = completion_tool
        self._required_files: list[Path] = [Path(f) for f in (required_files or [])]
        self._config_validator = config_validator
        self.leakage_pass_marker = leakage_pass_marker
        self.regression_pass_marker = regression_pass_marker
        self.review_pass_marker = review_pass_marker
        self._pre_submission_validator = pre_submission_validator
        self.max_interventions = max_interventions

        self._completion_called: bool = False
        self._auto_check_passed: bool = False
        self._auto_check_error: str | None = None   # cached error from last run
        self._leakage_check_passed: bool = False
        self._leakage_prompt_delivered: bool = False
        self._regression_check_passed: bool = False
        self._pre_submission_passed: bool = False
        self._regression_prompt_delivered: bool = False
        self._review_passed: bool = False
        self._review_prompt_delivered: bool = False
        self._intervention_count: int = 0
        self._pending_message: str = ""

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def on_task_start(self, event: TaskStartEvent):
        self._completion_called = False
        self._auto_check_passed = False
        self._auto_check_error = None
        self._leakage_check_passed = False
        self._leakage_prompt_delivered = False
        self._regression_check_passed = False
        self._regression_prompt_delivered = False
        self._review_passed = False
        self._review_prompt_delivered = False
        self._pre_submission_passed = False
        self._intervention_count = 0
        self._pending_message = ""
        yield event

    # ── tool interception ──────────────────────────────────────────────────────

    async def on_before_tool(self, event: ToolCallEvent):
        # (a) Short-circuit the fake keepalive tool.
        if event.tool_name == _FAKE_TOOL_NAME:
            yield dataclasses.replace(
                event, approved=False, synthetic_result=_FAKE_TOOL_ACK
            )
            return

        # (b) Guard completion_tool against premature submission.
        if event.tool_name == self.completion_tool:
            block_reason = self._submission_block_reason()
            if block_reason:
                logger.warning(
                    "SelfValidation: blocked %s — %s", self.completion_tool, block_reason
                )
                # Proactively run the remediation logic so the model gets an
                # actionable prompt inline (e.g. auto-check, leakage check,
                # self-review) rather than a generic "not yet" message.
                # This handles the common case where the model never produces a
                # bare exit-intent and always calls the completion tool directly.
                remediation = await self._build_remediation_message()
                synthetic = (
                    f"❌ `{self.completion_tool}` rejected.\n\n{remediation}"
                    if remediation
                    else (
                        f"❌ `{self.completion_tool}` rejected: {block_reason}\n\n"
                        f"Fix the issue above, then call `{self.completion_tool}` again."
                    )
                )
                yield dataclasses.replace(
                    event,
                    approved=False,
                    synthetic_result=synthetic,
                )
                return

            # All soft checks passed — run pre-submission policy checks.
            if self._pre_submission_validator is not None and not self._pre_submission_passed:
                manifest = (event.tool_input or {}).get("change_manifest", {})
                target = self._required_files[0] if self._required_files else None
                policy_error = await self._pre_submission_validator(target, manifest) if target else None
                if policy_error:
                    logger.warning("SelfValidation: pre-submission policy check rejected submission")
                    yield dataclasses.replace(
                        event,
                        approved=False,
                        synthetic_result=(
                            f"❌ `{self.completion_tool}` rejected — policy checks failed.\n\n"
                            f"{policy_error}\n\n"
                            f"Fix the issues above, then call `{self.completion_tool}` again."
                        ),
                    )
                    return
                self._pre_submission_passed = True

            self._completion_called = True

        yield event

    # ── exit-intent interception ───────────────────────────────────────────────

    async def on_after_model(self, event: ModelResponseEvent):
        # Scan content for markers only after the corresponding prompt has been delivered.
        # This prevents the model from accidentally setting flags by mentioning marker
        # strings in discussion text before the actual check phase.
        content = event.content or ""
        if self._leakage_prompt_delivered and self.leakage_pass_marker in content:
            self._leakage_check_passed = True
            logger.info("SelfValidation: leakage_pass_marker detected — leakage check confirmed")
        if self._regression_prompt_delivered and self.regression_pass_marker in content:
            self._regression_check_passed = True
            logger.info("SelfValidation: regression_pass_marker detected — regression check confirmed")
        if self._review_prompt_delivered and self.review_pass_marker in content:
            self._review_passed = True
            logger.info("SelfValidation: review_pass_marker detected — self-review confirmed")

        exit_intent = (
            event.finish_reason in ("end_turn", "stop")
            and not event.tool_calls
        )
        if not exit_intent:
            yield event
            return

        # Already submitted or limit reached — let the run end.
        if self._completion_called or self._intervention_count >= self.max_interventions:
            yield event
            return

        msg = await self._build_remediation_message()
        if not msg:
            yield event
            return

        self._intervention_count += 1
        self._pending_message = msg
        logger.info(
            "SelfValidation: exit_intent intercepted (intervention %d/%d)",
            self._intervention_count,
            self.max_interventions,
        )

        keepalive = ToolCall(
            id=f"sv-{uuid.uuid4().hex[:8]}",
            name=_FAKE_TOOL_NAME,
            input={},
        )
        yield dataclasses.replace(event, tool_calls=(keepalive,))

    # ── pending message delivery ───────────────────────────────────────────────

    async def on_before_model(self, event: BeforeModelEvent):
        if not self._pending_message:
            yield event
            return
        msg = self._pending_message
        self._pending_message = ""
        yield dataclasses.replace(
            event,
            messages=event.messages + (Message(role="user", content=msg),),
        )

    # ── task end log ───────────────────────────────────────────────────────────

    async def on_task_end(self, event: TaskEndEvent):
        missing = [str(f) for f in self._required_files if not f.exists()]
        if missing:
            logger.warning(
                "SelfValidation: task ended — artifacts NOT produced: %s", missing
            )
        else:
            logger.info(
                "SelfValidation: task ended — auto_check=%s  review=%s  completion=%s",
                self._auto_check_passed,
                self._review_passed,
                self._completion_called,
            )
        yield event

    # ── internal helpers ───────────────────────────────────────────────────────

    def _submission_block_reason(self) -> str | None:
        """Return a one-line reason to block submission, or None if allowed."""
        missing = [str(f) for f in self._required_files if not f.exists()]
        if missing:
            return f"required file(s) not found: {', '.join(missing)}"
        if self._config_validator is not None and not self._auto_check_passed:
            err = self._auto_check_error or "auto-check has not passed yet"
            return err
        if not self._leakage_check_passed:
            return f"leakage self-check not confirmed (output {self.leakage_pass_marker!r} to confirm)"
        if not self._regression_check_passed:
            return f"regression self-check not confirmed (output {self.regression_pass_marker!r} to confirm)"
        if not self._review_passed:
            return f"self-review not confirmed (output {self.review_pass_marker!r} to confirm)"
        return None

    async def _build_remediation_message(self) -> str:
        """
        Build a targeted message based on actual state.
        Returns empty string if everything is in order.
        """
        # Priority 1: required files not written yet.
        missing = [str(f) for f in self._required_files if not f.exists()]
        if missing:
            files = "\n".join(f"  - {f}" for f in missing)
            return (
                f"🔴 **MISSING OUTPUT**: The following required file(s) have not been written:\n"
                f"{files}\n\n"
                f"Write these files before calling `{self.completion_tool}`. "
                f"Do not produce a plain text response."
            )

        # Priority 2: files present — run auto-check if not yet passed.
        if self._config_validator is not None and not self._auto_check_passed:
            target = self._required_files[0] if self._required_files else None
            if target is not None:
                try:
                    error = await self._config_validator(target)
                except Exception as e:
                    error = f"Validator raised an exception: {e}"
                if error is None:
                    self._auto_check_passed = True
                    self._auto_check_error = None
                    logger.info("SelfValidation: auto-check passed for %s", target)
                else:
                    self._auto_check_error = error
                    return (
                        f"⚠️ **AUTO-CHECK FAILED**: The config did not pass programmatic "
                        f"validation:\n\n{error}\n\n"
                        f"Fix the errors above and try again."
                    )

        # Priority 3: auto-check passed — run leakage self-check.
        if not self._leakage_check_passed:
            self._leakage_prompt_delivered = True  # start accepting leakage marker
            file_list = self._list_new_processor_files()
            return _LEAKAGE_CHECK_PROMPT.format(
                file_list=file_list,
                marker=self.leakage_pass_marker,
            )

        # Priority 4: leakage confirmed — run regression self-check.
        if not self._regression_check_passed:
            self._regression_prompt_delivered = True  # start accepting regression marker
            return _REGRESSION_CHECK_PROMPT.format(marker=self.regression_pass_marker)

        # Priority 5: regression confirmed — ask agent to self-review.
        if not self._review_passed:
            self._review_prompt_delivered = True  # start accepting review marker
            return _SELF_REVIEW_PROMPT.format(
                completion_tool=self.completion_tool,
                marker=self.review_pass_marker,
            )

        # Priority 6: self-review confirmed — remind to call completion tool.
        if not self._completion_called:
            return (
                f"🔴 **MANDATORY**: Self-review confirmed and all checks passed. "
                f"Call `{self.completion_tool}` now to complete the task."
            )

        return ""

    def _list_new_processor_files(self) -> str:
        """
        Return a formatted bullet list of new .py processor files in the workspace.
        Walks up from required_files[0] until it finds a processors/ subdirectory,
        so it works regardless of how deep required_files[0] is nested.
        Falls back to a generic message if none found.
        """
        if not self._required_files:
            return "  (no workspace detected — check all processor logic manually)"

        # Walk up from required_files[0] until we find a processors/ sibling dir.
        candidate = self._required_files[0].resolve()
        proc_dir: Path | None = None
        for parent in [candidate.parent, *candidate.parents]:
            maybe = parent / "processors"
            if maybe.is_dir():
                proc_dir = maybe
                break

        if proc_dir is None:
            return "  (no processors/ directory found — confirm no new processor files were written)"

        py_files = sorted(proc_dir.glob("*.py"))
        if not py_files:
            return "  (no .py files found in processors/ — confirm no new processor files were written)"

        return "\n".join(f"  - `{f}` — use Read tool to inspect" for f in py_files)

