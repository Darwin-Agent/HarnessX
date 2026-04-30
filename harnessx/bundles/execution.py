# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..core.builder import HarnessBuilder


def make_execution(
    sandbox_provider=None,
    workspace=None,
    workspace_template=None,
    init_workspace=None,
) -> HarnessBuilder:
    """Return an execution environment bundle.

    All arguments map directly to ``HarnessConfig`` slots.  Omit any that
    the default runtime already handles.

    Args:
        sandbox_provider:   Object implementing the sandbox execution interface.
                            When ``None``, the harness uses the default local
                            subprocess executor.
        workspace:          Pre-constructed workspace handle passed to each run.
        workspace_template: Template object cloned to create a fresh workspace
                            per run (mutually exclusive with ``workspace``).
        init_workspace:     ``async callable(config) -> None`` invoked once
                            before the first step.
    """
    slots: dict = {}
    if sandbox_provider is not None:
        slots["sandbox_provider"] = sandbox_provider
    if workspace is not None:
        slots["workspace"] = workspace
    if workspace_template is not None:
        slots["workspace_template"] = workspace_template
    if init_workspace is not None:
        slots["init_workspace"] = init_workspace
    if not slots:
        return HarnessBuilder()
    return HarnessBuilder().slot(**slots)
