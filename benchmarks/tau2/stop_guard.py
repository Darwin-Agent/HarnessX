"""
StopGuardUserSimulator — strips premature ###STOP### from confirmation messages.

Root cause: GPT user simulators sometimes send "Yes, please proceed.###STOP###"
in the same turn as confirming the agent's plan.  The tau2 orchestrator detects
###STOP### and terminates immediately, before the agent gets a chance to execute
the write tool.  The DB never changes → task fails.

Fix: if the user message (stripped of ###STOP###) starts with an affirmative
("yes", "yeah", "sure", "ok", "proceed", …) the task is NOT yet complete — the
agent still needs to execute.  Strip ###STOP### so the orchestrator forwards the
message to the agent for one more turn.

On the next turn the agent calls the write tool, confirms execution, and the user
simulator's follow-up ("No, that's all. Thanks.###STOP###") legitimately ends the
conversation.

False-positive analysis (retail Sonnet-4.5+ET run, 114 tasks):
  - Premature stops starting with "yes": 26/34 — all caught and stripped.
  - Legitimate stops (agent already confirmed execution): 56 total, NONE start
    with "yes" — all safely ignored.  Zero false positives.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    pass

# Matches messages whose meaningful content is an affirmative confirmation.
# Applied to the message text AFTER ###STOP### is removed.
_YES_RE = re.compile(r"^\s*yes\b", re.IGNORECASE)

STOP_TOKEN = "###STOP###"


def _strip_premature_stop(content: str) -> str:
    """Return content with ###STOP### removed if it is a premature confirmation."""
    if STOP_TOKEN not in content:
        return content
    stripped = content.replace(STOP_TOKEN, "").strip()
    if _YES_RE.match(stripped):
        return stripped
    return content


def build_stop_guard_class() -> type:
    """Return StopGuardUserSimulator, built after tau2 is importable."""
    from tau2.user.user_simulator import UserSimulator

    class StopGuardUserSimulator(UserSimulator):
        """UserSimulator that strips ###STOP### from premature confirmation messages."""

        def generate_next_message(self, message, state) -> Tuple:
            user_msg, new_state = super().generate_next_message(message, state)

            if user_msg.content and STOP_TOKEN in user_msg.content:
                patched = _strip_premature_stop(user_msg.content)
                if patched != user_msg.content:
                    # Re-create message without ###STOP### so orchestrator keeps going
                    from tau2.data_model.message import UserMessage

                    user_msg = UserMessage(
                        role="user",
                        content=patched,
                        cost=user_msg.cost,
                        usage=user_msg.usage,
                        raw_data=getattr(user_msg, "raw_data", None),
                    )
                    # Keep new_state in sync
                    new_state.messages[-1] = user_msg

            return user_msg, new_state

    StopGuardUserSimulator.__name__ = "StopGuardUserSimulator"
    StopGuardUserSimulator.__qualname__ = "StopGuardUserSimulator"
    return StopGuardUserSimulator


try:
    StopGuardUserSimulator = build_stop_guard_class()
except ImportError:
    StopGuardUserSimulator = None  # tau2 not yet importable (e.g. during unit tests)
