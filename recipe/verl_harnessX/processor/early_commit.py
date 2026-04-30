# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""
Early-commit nudge processor for verl_harnessX agent loop.

Injects increasingly urgent commit messages at tool-response boundaries
based on the fraction of assistant turns consumed:
  - 1/2 of max_turns: gentle reminder to start converging
  - 3/4 of max_turns: urgent warning, must commit soon
  - last turn (max_turns - 1): final ultimatum — answer NOW

Each nudge level fires at most once per agent run.
Nudge tokens are appended as user messages and masked (mask=0) during RL.
"""

from __future__ import annotations


class EarlyCommitProcessor:
    NUDGE_HALF = (
        "[System Notice] You have used over half of your allowed turns. "
        "Start narrowing your search and focus on synthesizing your answer. "
        "If you already have strong evidence, provide your answer in "
        "<answer>...</answer> tags soon. "
        "Remember: an educated best answer is ALWAYS better than no answer."
    )

    NUDGE_THREE_QUARTER = "[System Notice] WARNING: You are running low on turns! "

    NUDGE_FINAL = (
        "[System Notice] LAST CHANCE: This is your final opportunity to answer. "
        "You MUST output your answer in <answer>...</answer> tags "
        "in your NEXT response. "
        "Use your best judgment based on everything you've gathered. "
        "DO NOT make any more tool calls — just give your answer immediately."
    )

    def __init__(self, max_turns: int):
        self.max_turns = max(max_turns, 1)
        self._fired: set[str] = set()

    def check(self, assistant_turns: int) -> str | None:
        """Return a nudge message if the current turn count triggers one.

        Args:
            assistant_turns: number of assistant generation turns completed so far.

        Returns:
            Nudge text to inject, or None if no nudge is due.
        """
        if self.max_turns <= 2:
            if assistant_turns >= self.max_turns - 1 and "final" not in self._fired:
                self._fired.add("final")
                return self.NUDGE_FINAL
            return None

        if assistant_turns >= self.max_turns - 1 and "final" not in self._fired:
            self._fired.add("final")
            return self.NUDGE_FINAL

        # if assistant_turns >= self.max_turns * 3 // 4 and "three_quarter" not in self._fired:
        #     self._fired.add("three_quarter")
        #     return self.NUDGE_THREE_QUARTER

        if assistant_turns >= self.max_turns // 2 and "half" not in self._fired:
            self._fired.add("half")
            return self.NUDGE_HALF

        return None

    def reset(self):
        """Reset fired state for a new agent run."""
        self._fired.clear()
