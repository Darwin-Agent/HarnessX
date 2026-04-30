# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from ...core.events import Message, StepStartEvent, TaskEndEvent, TaskStartEvent
from ...core.processor import MultiHookProcessor

_FINAL_CONFIG_PATH_PATTERNS = (
    re.compile(r"final_config_path`:\s*`([^`]+)`"),
    re.compile(r"EXACT path:\s*`([^`]+)`"),
)


class StepDeadlineReminderProcessor(MultiHookProcessor):
    """Inject step-based convergence reminders for meta-agent deadline management.

    Supports two-phase reminders:
    - Phase 1 (``early_reminder_step``): soft warning — finish analysis, start
      drafting the deliverable.
    - Phase 2 (``reminder_step``): hard stop — no new exploration, write
      ``config.yaml`` immediately.

    Example: ``early_reminder_step=200``, ``reminder_step=350``,
    ``output_within_steps=100`` (max_steps=500).
    """

    _singleton_group = "meta_step_deadline_reminder"
    _order = 7

    def __init__(
        self,
        *,
        reminder_step: int = 260,
        output_within_steps: int = 30,
        early_reminder_step: "int | None" = None,
    ) -> None:
        self.reminder_step = max(1, int(reminder_step))
        self.output_within_steps = max(1, int(output_within_steps))
        self.early_reminder_step = max(1, int(early_reminder_step)) if early_reminder_step is not None else None
        self._triggered = False
        self._early_triggered = False
        self._final_config_path: Path | None = None

    async def on_task_start(self, event: TaskStartEvent):
        self._triggered = False
        self._early_triggered = False
        self._final_config_path = None
        yield event

    async def on_step_start(self, event: StepStartEvent):
        if self._final_config_path is None:
            self._final_config_path = self._discover_final_config_path(event)
        if self._final_config_path is not None and self._final_config_path.is_file():
            yield event
            return

        max_steps = getattr(event.task, "max_steps", None)

        # Phase 1: early soft warning
        if (
            self.early_reminder_step is not None
            and not self._early_triggered
            and event.step_id >= self.early_reminder_step
            and event.step_id < self.reminder_step
        ):
            self._early_triggered = True
            steps_left = (max_steps - event.step_id) if isinstance(max_steps, int) else "?"
            output_path_hint = (
                f"Final config target: `{self._final_config_path}`."
                if self._final_config_path is not None
                else "Final config target: `final_config_path` declared in TASK.md."
            )
            early_reminder = (
                f"[StepDeadlineReminder] ⚠️ Midpoint warning — step {event.step_id}"
                + (f"/{max_steps}" if isinstance(max_steps, int) else "")
                + f" ({steps_left} steps remaining).\n"
                "Trajectory analysis should be wrapping up. "
                "If you have not yet started drafting the output `config.yaml`, begin NOW. "
                "Hard stop follows at step "
                f"{self.reminder_step} — only {self.reminder_step - event.step_id} steps away.\n"
                "Before the hard stop you must produce ALL of: "
                "`config.yaml`, `_meta_scratch/candidates.md` (required when config changes — "
                "at least one `## Candidate C-NNN` section), and a journal entry in `memo_path` "
                "with `cited_candidates`. Start outlining your candidates now.\n"
                f"{output_path_hint}"
            )
            msg = Message(role="user", content=early_reminder)
            yield dataclasses.replace(
                event,
                messages=event.messages + (msg,),
                raw_messages=event.raw_messages + (msg,),
            )
            return

        # Phase 2: hard stop
        if self._triggered or event.step_id < self.reminder_step:
            yield event
            return

        self._triggered = True
        if isinstance(max_steps, int) and max_steps > 0:
            deadline_step = min(max_steps, self.reminder_step + self.output_within_steps)
            budget_hint = (
                f"Current step: {event.step_id}/{max_steps}. "
                f"Hard deadline: finish by step {deadline_step} "
                f"(within {self.output_within_steps} steps)."
            )
        else:
            deadline_step = self.reminder_step + self.output_within_steps
            budget_hint = (
                f"Current step: {event.step_id}. "
                f"Hard deadline: finish by step {deadline_step} "
                f"(within {self.output_within_steps} steps)."
            )

        output_path_hint = (
            f"Final config target: `{self._final_config_path}`."
            if self._final_config_path is not None
            else "Final config target: `final_config_path` declared in TASK.md."
        )

        reminder = (
            "[StepDeadlineReminder] 🚨 STOP analysis now. "
            "Do not start new broad investigations or extra sub-agent fan-out. "
            f"{budget_hint}\n"
            "You must produce ALL of the following before end_turn — missing any one hard-fails the round:\n"
            "  1. `config.yaml` — write or copy the final HarnessConfig\n"
            "  2. `_meta_scratch/candidates.md` — REQUIRED when config changes: "
            "at least one `## Candidate C-NNN` section with lens/lever/intent tag, "
            "signal, verified body evidence, retroactive check, and 'Why X not Y'\n"
            "  3. Journal entry appended to `memo_path` with `cited_candidates` frontmatter "
            "referencing your C-NNN IDs\n"
            "Write these now. Do not end your turn until all three exist.\n"
            f"{output_path_hint}"
        )
        msg = Message(role="user", content=reminder)
        yield dataclasses.replace(
            event,
            messages=event.messages + (msg,),
            raw_messages=event.raw_messages + (msg,),
        )

    async def on_task_end(self, event: TaskEndEvent):
        self._triggered = False
        self._early_triggered = False
        self._final_config_path = None
        yield event

    def _discover_final_config_path(self, event: StepStartEvent) -> Path | None:
        for msg in event.messages:
            if msg.role != "user" or not isinstance(msg.content, str):
                continue
            for pattern in _FINAL_CONFIG_PATH_PATTERNS:
                matched = pattern.search(msg.content)
                if not matched:
                    continue
                raw_path = matched.group(1).strip()
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    continue
                if path.suffix in {".yaml", ".yml"}:
                    return path
        return None
