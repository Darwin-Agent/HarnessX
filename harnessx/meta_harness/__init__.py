# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Meta-harness — one meta-agent, one task: evolve a HarnessConfig.

Public API:

- :class:`MetaAgent` — class-based API; instantiate once per benchmark
  run, call ``await agent.evolve(...)`` per round.
- :func:`compute_changeset` — shallow diff of two canonical
  ``HarnessConfig`` objects (tools / processors / kwargs / templates).
- :func:`build_meta_agent_harness_config` — low-level factory exposed
  for tests and advanced callers that want the raw ``HarnessConfig``
  without the orchestration layer.

The meta-agent is an ordinary HarnessX agent: its identity, boundaries,
and workflow live in ``harnessx/meta_harness/workspace/`` (SOUL.md,
skills/). The orchestration logic — per-round brief, post-flight gates,
changeset computation, replay gate wiring — lives in ``agent.py``.
Cross-round memory sits in ``journal.py``; the synthetic-task replay
gate in ``replay.py``; and ``validate_workflow.py`` owns the two-phase
post-flight validator (validity → policy, plus an advisory literals
scan) and exposes agent-facing self-check CLIs via
``python -m harnessx.meta_harness.validate_workflow <subcommand>``.

This package is benchmark-agnostic — callers (``recipe/<bench>/run.py``)
own the benchmark loop, gating, and round indexing.
"""

from .agent import (
    MetaAgent,
    build_meta_agent_harness_config,
    compute_changeset,
)

__all__ = [
    "MetaAgent",
    "build_meta_agent_harness_config",
    "compute_changeset",
]
