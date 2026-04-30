# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..core.builder import HarnessBuilder
from ..processors.control.sycophancy_detector import SycophancyDetector


def make_contrarian(
    streak_threshold: int = 3,
    judge_key: str = "judge",
    adversarial_key: str = "adversarial",
    task_mode_tools: frozenset[str] | set[str] = frozenset(),
    lookback_steps: int = 3,
) -> HarnessBuilder:
    """Return a customised contrarian harness bundle.

    Secondary providers are resolved from the harness's sub-harnesses registry
    at call time via ``judge_key`` / ``adversarial_key``.  If a key is absent,
    the feature degrades gracefully: no judge → regex-only detection;
    no adversarial → instruction injection.

    Args:
        streak_threshold:  Consecutive-agreement turns before contrarian mode
                           fires (default: 3).
        judge_key:         Providers registry key for the LLM judge.  When the
                           key resolves, the judge *confirms* regex hits (two-
                           layer detection) and classifies chat vs. task when
                           ``task_mode_tools`` is empty.  Default: ``"judge"``.
        adversarial_key:   Providers registry key for the adversarial critique
                           provider.  When the key resolves, an out-of-band
                           call is forked on streak and the result is appended
                           under "Devil's Advocate" in the current response.
                           When absent, a ``[CONTRARIAN MODE]`` instruction is
                           injected into the next turn's system prompt instead.
                           Default: ``"adversarial"``.
        task_mode_tools:   Tool names whose presence in recent history suppresses
                           contrarian mode.  Empty = always-on (pure chat).
                           When empty and a judge is registered, an LLM
                           classifier decides chat vs. task automatically.
        lookback_steps:    How many recent assistant turns to scan for task-mode
                           tool signals (default: 3).
    """
    return HarnessBuilder().add(
        SycophancyDetector(
            streak_threshold=streak_threshold,
            judge_key=judge_key,
            adversarial_key=adversarial_key,
            task_mode_tools=frozenset(task_mode_tools),
            lookback_steps=lookback_steps,
        )
    )


contrarian: HarnessBuilder = make_contrarian()
"""Contrarian bundle with default parameters (always-on, instruction injection only).

Plug into any ``HarnessBuilder`` via ``| contrarian``.

Secondary providers (judge, adversarial) are resolved from the registry at call
time.  If not registered, the bundle degrades gracefully to regex-only detection
and instruction injection.

.. note::
    This module-level singleton holds a single ``SycophancyDetector`` instance.
    For concurrent multi-user scenarios or independent harness instances that
    must not share per-conversation state (agree streak, pending flags), call
    ``make_contrarian()`` to get a fresh instance each time.
"""
