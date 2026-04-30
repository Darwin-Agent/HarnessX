# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...base import HarnessPlugin
from .processors import LightMemoryCaptureProcessor, LightMemoryRetrievalProcessor

if TYPE_CHECKING:
    pass


class LightMemoryPlugin(HarnessPlugin):
    """File-based markdown memory with keyword search and exponential decay.

    Zero dependencies beyond the Python stdlib — uses git + the filesystem
    as the memory infrastructure and reuses whatever tools the agent already
    has available.
    """

    name = "memory.light"
    version = "0.1.0"
    description = (
        "File-based markdown memory: agent-driven recall + daily capture + "
        "background organisation.  Zero external dependencies."
    )

    def __init__(
        self,
        memory_root: str | None = None,
        user_id: str = "user",
        top_k: int = 15,
        half_life_days: int = 30,
        organization_enabled: bool = True,
        organization_interval_ms: int = 1_800_000,
        organization_timeout_ms: int = 30_000,
        decay_enabled: bool = True,
        git_mode: str = "optional",
        auto_recall: bool = True,
        auto_capture: bool = False,
        auto_commit: bool = True,
    ) -> None:
        super().__init__()
        self._memory_root = memory_root
        self._user_id = user_id
        self._top_k = top_k
        self._half_life_days = half_life_days
        self._organization_enabled = organization_enabled
        self._organization_interval_ms = organization_interval_ms
        self._organization_timeout_ms = organization_timeout_ms
        self._decay_enabled = decay_enabled
        self._git_mode = git_mode
        self._auto_recall = auto_recall
        self._auto_capture = auto_capture
        self._auto_commit = auto_commit

        # Processors created eagerly so HarnessBuilder can extract them at
        # build time.  Configured (via .configure()) in setup().
        self._retrieval = LightMemoryRetrievalProcessor()
        self._capture = LightMemoryCaptureProcessor()
        self.processors = [self._retrieval, self._capture]

    # ── Tool requirements ──────────────────────────────────────────────────────
    # This plugin is agent-driven: the agent reads and writes memory files
    # using its built-in tools.  Without these tools the operation guidance
    # injected into the system prompt cannot be acted upon.

    _REQUIRED_TOOLS: frozenset[str] = frozenset({"Read", "Write", "Edit"})
    _RECOMMENDED_TOOLS: frozenset[str] = frozenset({"Bash"})

    @classmethod
    def _check_tools(cls, config: Any) -> None:
        """Validate that the required file-system tools are available.

        Raises
        ------
        RuntimeError
            When the agent has a tool registry but none of the required
            file-system tools (Read, Write, Edit) are registered.
        """
        import warnings

        registry = getattr(config, "tool_registry", None)
        if registry is None:
            # No registry at all — external tool surface, skip check.
            warnings.warn(
                "LightMemoryPlugin: no tool_registry found on harness config; "
                "cannot verify that the agent has file-system tools available. "
                "The plugin requires Read, Write, and Edit tools to function correctly.",
                stacklevel=4,
            )
            return

        try:
            registered: set[str] = set(registry.list_names())
        except Exception:
            return  # registry doesn't support listing — skip

        if not registered:
            return  # empty registry — tools may be added later externally

        present = cls._REQUIRED_TOOLS & registered
        missing = cls._REQUIRED_TOOLS - registered

        if not present:
            raise RuntimeError(
                "LightMemoryPlugin requires the agent to have file-system tools "
                "so it can read and write memory files directly during the conversation. "
                "None of the required tools were found in the tool registry.\n"
                f"  Required : {sorted(cls._REQUIRED_TOOLS)}\n"
                f"  Registered: {sorted(registered) or '(none)'}\n"
                "Add the built-in file tools to your harness:\n"
                "    from harnessx.tools.builtin import Read, Write, Edit, Bash\n"
                "    builder.add_tool(Read).add_tool(Write).add_tool(Edit).add_tool(Bash)"
            )

        if missing:
            warnings.warn(
                f"LightMemoryPlugin: some required file-system tools are missing: "
                f"{sorted(missing)}.  The agent may not be able to perform all memory "
                f"operations.  Consider adding: "
                f"{', '.join(sorted(missing))}",
                stacklevel=4,
            )

        # Recommended tools check (soft warning only)
        missing_rec = cls._RECOMMENDED_TOOLS - registered
        if missing_rec:
            warnings.warn(
                f"LightMemoryPlugin: recommended tool(s) not found: "
                f"{sorted(missing_rec)}.  "
                "'Bash' is useful for git operations and browsing memory directories.",
                stacklevel=4,
            )

    # ── setup ──────────────────────────────────────────────────────────────────

    def setup(self, config: Any) -> None:
        """Wire providers and initialise the memory repository."""
        self._check_tools(config)

        from ._core.backend import ensure_git_repo, ensure_memory_repo
        from ._core.types import PluginConfig

        # Resolve memory root: explicit arg > AGENT_HOME/memory/
        if self._memory_root:
            root = self._memory_root
        else:
            try:
                from ....home import agent_home

                root = str(agent_home() / "memory")
            except Exception:
                import os

                root = os.path.expanduser("~/.harnessx/memory")

        cfg = PluginConfig(
            memory_root=root,
            user_id=self._user_id,
            top_k=self._top_k,
            access_half_life_days=self._half_life_days,
            organization_enabled=self._organization_enabled,
            organization_interval_ms=max(60_000, int(self._organization_interval_ms)),
            organization_timeout_ms=max(10_000, int(self._organization_timeout_ms)),
            decay_enabled=self._decay_enabled,
            git_mode=self._git_mode,
            auto_commit=self._auto_commit,
            auto_recall=self._auto_recall,
            auto_capture=self._auto_capture,
        )
        ensure_memory_repo(cfg)
        ensure_git_repo(cfg)

        # Extract the main provider from the harness config (best-effort).
        provider = None
        try:
            provider = config.model_config.main
        except AttributeError:
            pass

        self._retrieval.configure(cfg)
        self._capture.configure(cfg, provider)
