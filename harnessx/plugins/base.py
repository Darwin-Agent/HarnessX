# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.harness import HarnessConfig


class HarnessPlugin:
    """Base class for HarnessX plugins.

    Two-phase lifecycle
    -------------------
    **Build time** (``HarnessBuilder.plugin(p)``):
      ``p.processors`` and ``p.tools`` are read and merged into the builder.
      No I/O, no network calls.  These lists are typically set once in
      ``__init__`` and never mutated afterward.

    **Runtime** (``Harness.__init__(config)``):
      ``p.setup(config)`` is called with the fully assembled ``HarnessConfig``.
      Use this to open connections, warm caches, register dynamic tools, etc.
      ``p.stop()`` is called when the harness is cleaned up
      (once per ``Harness.cleanup()``).

    Subclass recipe
    ---------------
    ::

        class MyPlugin(HarnessPlugin):
            name = "my-plugin"
            version = "0.1.0"
            description = "Example dimension plugin"

            def __init__(self, window: int = 8):
                super().__init__()
                self.processors = [SlidingWindowProcessor(window)]

            def setup(self, config):
                # Optional: open DB connection, etc.
                pass

            def stop(self):
                # Optional: close resources
                pass

    Attributes:
        name:            Unique plugin identifier (e.g. ``"memory-boost"``).
        version:         Semver string.
        description:     Human-readable description.

        processors:      List of ``MultiHookProcessor`` instances.  Hook points
                         are derived from the processor's ``on_*`` method names.
                         **Set as an instance attribute in ``__init__``**, not as
                         a class attribute, to prevent accidental shared-state
                         mutation across instances.
        tools:           List of ``Tool`` instances to register in the tool registry.
        slash_commands:  Mapping ``"/command"`` → slot key (``str``) or ``None``.
                         Slot-based: registry sets ``state.slots[slot_key]`` and
                         a processor handles it at ``on_task_start``.
                         Direct: registry calls ``_handle_<cmd>()`` immediately.
        commands:        Claude Code-compatible prompt-injection commands.
                         ``[{"name": "cmd", "description": "...", "prompt": "..."}]``
                         When the user invokes ``/cmd``, the prompt is prepended to
                         ``event.system_prompt`` by ``CommandInjectionProcessor``.
        skill_dirs:      List of ``Path`` objects pointing to directories that each
                         contain a ``SKILL.md`` file.  Installed into the workspace
                         ``skills/`` directory at ``setup()`` time via ``SkillManager``.
        mcp_servers:     List of MCP server specs — same format as Claude Code's
                         ``mcpServers`` object entries:
                         ``[{"name": "sqlite", "transport": "stdio",
                            "command": "uvx mcp-server-sqlite --db ./db.sqlite"}]``
                         Mounted into the single ``McpRuntimePlugin`` runtime.
                         The runtime hot-reloads on ``task_start`` and injects
                         tools into the harness ``tool_registry``.
        lifecycle_hooks: Shell commands to run at specific HarnessX lifecycle
                         events.  Keys map Claude Code hook event names to
                         HarnessX hook points:
                         ``{"Stop": ["bash ./hooks/stop.sh"],
                            "PreToolUse": [...], "PostToolUse": [...]}``
                         Executed by ``ShellHookProcessor``.
    """

    name: str = ""
    version: str = "0.1.0"
    description: str = ""

    # Class-level defaults — subclasses that declare these as class attributes
    # (e.g. ``processors = [MyProc()]``) are supported but discouraged because
    # class-level lists are shared across instances.  Prefer setting them in
    # ``__init__`` as shown in the recipe above.
    processors: list = []
    tools: list = []
    slash_commands: dict[str, str | None] = {}
    commands: list[dict] = []

    skill_dirs: list[Path] = []
    mcp_servers: list[dict] = []
    lifecycle_hooks: dict = {}

    def __init__(self) -> None:
        # Ensure every instance gets its own list copies so that
        # ``self.processors.append(...)`` never mutates the class attribute.
        # Subclasses that call ``super().__init__()`` get this for free.
        cls = type(self)
        if "processors" not in self.__dict__:
            self.processors = list(cls.processors)
        if "tools" not in self.__dict__:
            self.tools = list(cls.tools)
        if "commands" not in self.__dict__:
            self.commands = list(cls.commands)
        if "skill_dirs" not in self.__dict__:
            self.skill_dirs = list(cls.skill_dirs)
        if "mcp_servers" not in self.__dict__:
            self.mcp_servers = list(cls.mcp_servers)
        if "slash_commands" not in self.__dict__:
            self.slash_commands = dict(cls.slash_commands)
        if "lifecycle_hooks" not in self.__dict__:
            self.lifecycle_hooks = dict(cls.lifecycle_hooks)

    def setup(self, config: "HarnessConfig") -> None:
        """Called once when the Harness is instantiated (after all processors are wired).

        Override to perform one-time initialisation.  ``config`` is the fully
        assembled ``HarnessConfig``; ``config.workspace`` is available if set.

        Called in ``Harness.__init__``, so it runs before the first ``run()``.
        """

    def stop(self) -> "Any | None":
        """Called once when ``Harness.cleanup()`` runs.

        Override to release harness-scoped resources.
        May return an awaitable for async teardown.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} v{self.version}>"
