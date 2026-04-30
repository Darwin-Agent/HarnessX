# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolRegistryConfig:
    """Declarative descriptor for a tool registry.

    ``builtin`` lists tool names to register from the standard builtin set
    (e.g. ``["Bash", "Read", "Write"]``).  ``custom`` lists fully-qualified
    import paths of custom tool functions (``"my.module.my_tool"``).
    """

    builtin: list[str] = field(default_factory=list)
    custom: list[str] = field(default_factory=list)


@dataclass
class TracerConfig:
    """Declarative descriptor for a HarnessJournal tracer.

    ``_target_`` allows swapping the tracer implementation.
    All other fields are forwarded as constructor kwargs.
    """

    _target_: str = "harnessx.tracing.journal.HarnessJournal"
    export_jsonl: bool = True
    silent: bool = False
    session_id: Optional[str] = None
    base_dir: Optional[str] = None


@dataclass
class NullTracerConfig(TracerConfig):
    """Declarative descriptor for NullTracer (suppresses all tracing output)."""

    _target_: str = "harnessx.tracing.null_tracer.NullTracer"
    export_jsonl: bool = False
    silent: bool = True


@dataclass
class WorkspaceConfig:
    """Declarative descriptor for a Workspace.

    When ``home`` is set and ``root`` is *None*, the workspace root is
    auto-derived as ``home/workspaces/{agent_id}/{project}/``.
    """

    root: Optional[str] = None
    agent_id: str = "hxagent"
    project: str = "default"
    mode: Optional[str] = "isolated"
    home: Optional[str] = None
    parent_id: Optional[str] = None


@dataclass
class SandboxConfig:
    """Declarative descriptor for a sandbox provider.

    ``_target_`` is resolved via importlib at instantiation time.
    """

    _target_: str = "harnessx.sandbox.local.LocalSandboxProvider"


@dataclass
class PluginConfig:
    """Declarative descriptor for a HarnessPlugin.

    Two loading strategies:
    - Python class: set ``_target_`` to a dotted import path.
    - Directory plugin: set ``path`` to a dir containing ``plugin.json``.

    Init kwargs are serialized as additional flat fields alongside ``_target_``
    and restored at instantiation time.
    """

    _target_: str = ""
    path: Optional[str] = None


__all__ = [
    "ToolRegistryConfig",
    "TracerConfig",
    "NullTracerConfig",
    "WorkspaceConfig",
    "SandboxConfig",
    "PluginConfig",
]
