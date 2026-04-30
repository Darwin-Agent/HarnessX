# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .harness import HarnessConfig
    from .processor import Processor
    from ..tools.base import Tool

# Sentinel — distinguishes "not provided" from explicit None
_UNSET: Any = object()

# Valid HarnessConfig scalar slots managed by .slot()
_SCALAR_SLOTS = frozenset(
    {
        "tool_registry",
        "tracer",
        "workspace",
        "workspace_template",
        "init_workspace",
        "sandbox_provider",
        "sandbox_hint_id",
        "step_snapshots",
    }
)


# ---------------------------------------------------------------------------
# _ProcEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProcEntry:
    """Metadata wrapper for a registered Processor."""

    processor: Any  # Processor instance
    hook: str  # "context_ready" | "before_model" | ... | "*"
    order: int = 0  # lower = earlier within the same hook
    singleton_group: str | None = None  # at most one entry per group across merged builders
    after: tuple[str, ...] = ()  # soft ordering deps by singleton_group within same hook


# ---------------------------------------------------------------------------
# HarnessConflictError
# ---------------------------------------------------------------------------


class HarnessConflictError(Exception):
    """Raised by HarnessBuilder.merge() when incompatible builders are combined.

    ``error.conflicts`` is the full list of conflict descriptions so callers
    can programmatically inspect or log them.
    """

    def __init__(self, conflicts: list[str]) -> None:
        self.conflicts = list(conflicts)
        lines = "\n".join(f"  [{i + 1}] {c}" for i, c in enumerate(conflicts))
        super().__init__(f"{len(conflicts)} conflict(s) detected:\n{lines}")


# ---------------------------------------------------------------------------
# HarnessBuilder
# ---------------------------------------------------------------------------


class HarnessBuilder:
    """Immutable, composable factory for :class:`~harnessx.core.harness.HarnessConfig`.

    All mutation methods return a **new** ``HarnessBuilder``; the receiver is
    never modified.  This lets you define a bundle once and compose it into
    multiple downstream builders without side-effects.
    """

    # ── construction ────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._entries: list[_ProcEntry] = []
        self._slots: dict[str, Any] = {}  # scalar HarnessConfig slots
        self._tools: list[Any] = []  # Tool instances bundled with this harness
        self._plugins: list[Any] = []  # HarnessPlugin objects for lifecycle calls
        self._model_config: Any = None  # optional ModelConfig from descriptor.model

    # ── fluent builder API ──────────────────────────────────────────────────

    def add(
        self,
        proc: "Processor",
        *,
        hook: str | None = None,
        order: int | None = None,
        singleton_group: Any = _UNSET,
    ) -> "HarnessBuilder":
        """Register *proc* under a hook point.

        If the processor class declares ``_hook``, ``_order``, and/or
        ``_singleton_group`` class attributes, they are used as defaults.
        Explicit keyword arguments take precedence.

        Args:
            proc:             Processor instance to register.
            hook:             Hook key (``"step_start"``, ``"before_model"``,
                              ``"after_model"``, ``"before_tool"``,
                              ``"after_tool"``, ``"step_end"``, ``"task_end"``,
                              or ``"*"``).  Required if the class has no ``_hook``.
            order:            Position within the hook list (lower = earlier).
            singleton_group:  Conflict-detection group.  Pass ``None`` explicitly
                              to disable singleton checking for this entry.
        """
        from .processor import MultiHookProcessor

        cls = type(proc)
        resolved_hook = hook or getattr(cls, "_hook", None)
        if not resolved_hook and isinstance(proc, MultiHookProcessor):
            resolved_hook = "*"
        if not resolved_hook:
            raise ValueError(
                f"{cls.__name__} has no _hook class attribute and is not a "
                "MultiHookProcessor subclass; pass hook= explicitly to .add()"
            )
        resolved_order = order if order is not None else getattr(cls, "_order", 0)
        resolved_sg = singleton_group if singleton_group is not _UNSET else getattr(cls, "_singleton_group", None)
        resolved_after = tuple(getattr(cls, "_after", []))
        new = self._copy()
        new._entries.append(_ProcEntry(proc, resolved_hook, resolved_order, resolved_sg, resolved_after))
        return new

    def slot(self, **kwargs: Any) -> "HarnessBuilder":
        """Set one or more scalar HarnessConfig slots.

        Valid keys: ``tool_registry``, ``tracer``,
        ``workspace``, ``workspace_template``, ``init_workspace``.
        """
        invalid = set(kwargs) - _SCALAR_SLOTS
        if invalid:
            raise ValueError(f"Unknown slot(s): {invalid}. Valid: {_SCALAR_SLOTS}")
        new = self._copy()
        new._slots.update(kwargs)
        return new

    def add_tool(self, tool: "Tool") -> "HarnessBuilder":
        """Bundle a tool with this harness.

        Tools are registered into the ``tool_registry`` slot at :meth:`build`
        time.  If no registry was supplied via ``.slot(tool_registry=...)``, an
        ``InMemoryToolRegistry`` is created automatically.

        This is intended for **harness-defined convenience tools** — e.g. the
        ``todo_write`` tool paired with ``TodoCheck``, or the ``task``
        delegation tool for sub-agent bundles.  Workspace-bound tools (Bash,
        Read, Write …) and MCP tools (via ``MCPToolRegistry``) are registered
        by the caller, not through ``add_tool()``.

        Merging two builders that both register a tool with the same name raises
        :exc:`HarnessConflictError`.
        """
        new = self._copy()
        new._tools.append(tool)
        return new

    # ── composition ─────────────────────────────────────────────────────────

    def __or__(self, other: "HarnessBuilder") -> "HarnessBuilder":
        """Merge two builders.  Conflicts raise :exc:`HarnessConflictError`."""
        return HarnessBuilder.merge(self, other)

    @classmethod
    def merge(cls, *builders: "HarnessBuilder") -> "HarnessBuilder":
        """Merge an arbitrary number of builders left-to-right.

        All conflicts are collected before raising so the caller sees the
        complete picture in a single :exc:`HarnessConflictError`.
        """
        if not builders:
            return cls()
        result = builders[0]._copy()
        conflicts: list[str] = []
        for b in builders[1:]:
            _merge_into(result, b, conflicts)
        if conflicts:
            raise HarnessConflictError(conflicts)
        return result

    def plugin(self, source: "Any") -> "HarnessBuilder":
        """Register a plugin, merging all its capabilities into this builder.

        Handles every capability declared by the plugin:

        - ``processors``       → ``builder.add()``
        - ``tools``            → ``builder.add_tool()``
        - ``commands``         → shared ``CommandInjectionProcessor``
        - ``mcp_servers``      → ``McpRuntimePlugin`` (single runtime, hot-reload style)
        - ``lifecycle_hooks``  → ``ShellHookProcessor`` (if any hooks declared)
        - ``skill_dirs``       → installed via ``SkillManager`` at first run
          (deferred — requires workspace, which may not exist at build time)

        Args:
            source: A ``HarnessPlugin`` instance/class, a filesystem path to a
                    directory containing ``plugin.json``, or a dotted Python
                    import path string.

        Returns:
            A new builder with the plugin's capabilities merged in.
        """
        from ..plugins.loader import load_plugin
        from ..plugins.builtins.shell_hook import build_shell_hook_processor
        from ..plugins.dimensions.mcp_runtime import McpRuntimePlugin

        loaded = load_plugin(source)
        new = self

        for proc in loaded.processors:
            # Plugin owns these instances (may hold state/connections); mark as
            # runtime-only so build() keeps the instance instead of serializing it.
            if not getattr(proc, "__hx_runtime_only__", False):
                proc.__hx_runtime_only__ = True
            new = new.add(proc)
        for tool in loaded.tools:
            new = new.add_tool(tool)

        if loaded.commands:
            from ..plugins.builtins.command_injection import CommandInjectionProcessor

            existing = next(
                (e.processor for e in new._entries if isinstance(e.processor, CommandInjectionProcessor)),
                None,
            )
            if existing is not None:
                existing.add_commands(loaded.commands)
            else:
                cmd_proc = CommandInjectionProcessor()
                cmd_proc.add_commands(loaded.commands)
                new = new.add(cmd_proc)

        if loaded.mcp_servers:
            runtime_plugin = next((p for p in new._plugins if isinstance(p, McpRuntimePlugin)), None)
            if runtime_plugin is None:
                runtime_plugin = McpRuntimePlugin(
                    mcp_config={"source": "inline", "servers": loaded.mcp_servers},
                    ensure_primary=False,
                )
                for runtime_proc in runtime_plugin.processors:
                    new = new.add(runtime_proc)
                new._plugins.append(runtime_plugin)
            else:
                runtime_plugin.add_inline_servers(loaded.mcp_servers)

        if loaded.lifecycle_hooks:
            plugin_root = getattr(loaded, "_plugin_root", None)
            if plugin_root is not None:
                hook_proc = build_shell_hook_processor(
                    loaded.lifecycle_hooks,
                    plugin_root=plugin_root,
                    plugin_name=loaded.name or "unknown",
                )
                if hook_proc is not None:
                    new = new.add(hook_proc)

        # skill_dirs: plugin skills are discovered at runtime by SkillIndex
        # via collect_plugin_skill_dirs() — no build-time wiring needed.

        # Store the plugin object for lifecycle calls (setup/stop).
        new._plugins.append(loaded)

        return new

    # ── terminal ─────────────────────────────────────────────────────────────

    def build(self, journal_dir: "str | None" = None) -> "HarnessConfig":
        """Return a :class:`~harnessx.core.harness.HarnessConfig` from this builder.

        Args:
            journal_dir: Override the HarnessJournal ``base_dir``.  When set,
                         a ``TracerConfig(base_dir=journal_dir)`` is slotted
                         in, taking precedence over any tracer already in the
                         builder.  When *None* (default), the existing tracer
                         slot is used and the journal directory is resolved
                         automatically in ``Harness.__init__`` (workspace root
                         or agent_home-derived path).
        """
        from .harness import HarnessConfig, _serialize_processor
        from .config_schema import TracerConfig

        # Produce a flat ordered list of _target_ dicts.
        # Runtime-only processors (returned None by _serialize_processor) are
        # kept as instances; to_yaml() filters them out, _instantiate_runtime
        # routes them through _route_processors as usual.
        processors: list = []
        hook_entries: dict[str, list[_ProcEntry]] = {}
        for entry in self._entries:
            hook_entries.setdefault(entry.hook, []).append(entry)
        for hook, entries in hook_entries.items():
            for entry in _topological_sort_entries(entries):
                proc = entry.processor
                serialized = _serialize_processor(proc)
                if serialized is not None:
                    natural = getattr(proc, "_hook", None) or getattr(type(proc), "_hook", None)
                    if natural != hook:
                        serialized["_hook_"] = hook
                    processors.append(serialized)
                else:
                    if not hasattr(proc, "__hx_hook_override__"):
                        natural = getattr(proc, "_hook", None) or getattr(type(proc), "_hook", None)
                        if natural != hook:
                            proc.__hx_hook_override__ = hook
                    processors.append(proc)

        # ── tool_registry ─────────────────────────────────────────────────────
        slots = dict(self._slots)
        if self._tools:
            registry = slots.get("tool_registry")
            if registry is None:
                from ..tools.inmemory import InMemoryToolRegistry

                registry = InMemoryToolRegistry()
            for tool in self._tools:
                registry.register(tool)
            slots["tool_registry"] = registry

        # ── tracer ────────────────────────────────────────────────────────────
        if journal_dir is not None:
            slots["tracer"] = TracerConfig(base_dir=str(journal_dir))

        return HarnessConfig(
            processors=processors,
            plugins=list(self._plugins),
            **slots,
        )

    # ── internal helpers ─────────────────────────────────────────────────────

    def _copy(self) -> "HarnessBuilder":
        new = HarnessBuilder.__new__(HarnessBuilder)
        new._entries = list(self._entries)
        new._slots = dict(self._slots)
        new._tools = list(self._tools)
        new._plugins = list(self._plugins)
        new._model_config = self._model_config
        return new


# ---------------------------------------------------------------------------
# Module-level merge helper (keeps HarnessBuilder class body clean)
# ---------------------------------------------------------------------------


def _merge_into(
    target: HarnessBuilder,
    source: HarnessBuilder,
    conflicts: list[str],
) -> None:
    """Mutate *target* by absorbing *source*; append conflict descriptions."""

    # 1. scalar slots — conflict when both set to different objects
    for key, val in source._slots.items():
        if key in target._slots and target._slots[key] is not val:
            conflicts.append(
                f"slot '{key}': set in both builders ({type(target._slots[key]).__name__} vs {type(val).__name__})"
            )
        else:
            target._slots[key] = val

    # 2. tools — conflict when same name appears in both builders
    used_tool_names = {t.name for t in target._tools}
    for tool in source._tools:
        if tool.name in used_tool_names:
            conflicts.append(f"tool '{tool.name}': registered in both builders")
        else:
            target._tools.append(tool)
            used_tool_names.add(tool.name)

    # 3. plugins — accumulate (no conflict detection; same instance allowed once)
    seen_plugin_ids = {id(p) for p in target._plugins}
    for plugin in source._plugins:
        if id(plugin) not in seen_plugin_ids:
            target._plugins.append(plugin)
            seen_plugin_ids.add(id(plugin))

    # 4. processors — singleton_group collision
    used_groups = {e.singleton_group for e in target._entries if e.singleton_group}
    for entry in source._entries:
        if entry.singleton_group and entry.singleton_group in used_groups:
            # Find the colliding entry for a helpful message
            existing = next(e for e in target._entries if e.singleton_group == entry.singleton_group)
            conflicts.append(
                f"singleton_group='{entry.singleton_group}': "
                f"{type(existing.processor).__name__} (existing) conflicts with "
                f"{type(entry.processor).__name__} (incoming)"
            )
        else:
            target._entries.append(entry)
            if entry.singleton_group:
                used_groups.add(entry.singleton_group)


# ---------------------------------------------------------------------------
# Short-name registry + universal instantiator
# ---------------------------------------------------------------------------

#: Short name → fully-qualified class path.
#: Extend at import time: ``from harnessx.core.builder import NAMES; NAMES["my.proc"] = "my.Proc"``
NAMES: dict[str, str] = {
    # ── Memory strategies ─────────────────────────────────────────────────────
    "sliding_window": "harnessx.processors.memory.strategies.sliding_window.SlidingWindowMemory",
    "summarization": "harnessx.processors.memory.strategies.summarization.SummarizationMemory",
    # ── System prompt builders ────────────────────────────────────────────────
    "default_system": "harnessx.processors.context.strategies.system_prompt.default.DefaultSystemPromptBuilder",
    "template_system": "harnessx.processors.context.strategies.system_prompt.template.TemplateSystemPromptBuilder",
    "null_system": "harnessx.processors.context.strategies.system_prompt.null.NullSystemPromptBuilder",
    # ── User wrappers ─────────────────────────────────────────────────────────
    "xml_format": "harnessx.processors.context.strategies.user_wrapper.xml_format.XMLFormatWrapper",
    "cot": "harnessx.processors.context.strategies.user_wrapper.cot.ChainOfThoughtWrapper",
    # ── Evaluators ────────────────────────────────────────────────────────────
    "llm_judge": "harnessx.processors.evaluation.strategies.evaluators.llm_judge.LLMJudgeEvaluator",
    "self_verify": "harnessx.processors.evaluation.strategies.evaluators.self_verify.SelfVerifyEvaluator",
    # ── PRM strategies ────────────────────────────────────────────────────────
    "prm.terminal": "harnessx.processors.evaluation.strategies.evaluators.prm.TerminalPRM",
    "prm.discounted": "harnessx.processors.evaluation.strategies.evaluators.prm.DiscountedPRM",
    "prm.tool_success": "harnessx.processors.evaluation.strategies.evaluators.prm.ToolSuccessPRM",
    "prm.llm_judge": "harnessx.processors.evaluation.strategies.evaluators.prm.LLMJudgePRM",
    # ── Model providers ───────────────────────────────────────────────────────
    "anthropic": "harnessx.providers.anthropic_provider.AnthropicProvider",
    "litellm": "harnessx.providers.litellm_provider.LiteLLMProvider",
    "provider_group": "harnessx.providers.group.ProviderGroup",
    # ── Context processors ────────────────────────────────────────────────────
    "system_prompt": "harnessx.processors.context.system_prompt.SystemPromptProcessor",
    "user_wrapper": "harnessx.processors.context.user_wrapper.UserWrapperProcessor",
    "env_injection": "harnessx.processors.context.env_context_injector.EnvironmentContextInjector",
    # ── Memory processors ─────────────────────────────────────────────────────
    "memory_retrieval": "harnessx.processors.memory.memory_retrieval.MemoryRetrievalProcessor",
    "memory_extraction": "harnessx.processors.memory.memory_extraction.MemoryExtractionProcessor",
    # ── Control processors ────────────────────────────────────────────────────
    "loop_detection": "harnessx.processors.control.loop_detection.LoopDetectionProcessor",
    "tool_call_correction": "harnessx.processors.control.tool_call_correction.ToolCallCorrectionLayer",
    "parse_retry": "harnessx.processors.control.parse_retry.ParseRetryProcessor",
    "self_verify_proc": "harnessx.processors.control.self_verify.SelfVerifyProcessor",
    "todo_check": "harnessx.processors.control.todo_check.TodoWriteEnforcer",
    "repeated_edit_detector": "harnessx.processors.control.repeated_edit_detector.RepeatedFileEditDetector",
    "bg_install_guard": "harnessx.processors.control.bg_install_guard.BgInstallGuard",
    "cost_guard": "harnessx.processors.control.cost_guard.CostGuardProcessor",
    "compaction": "harnessx.processors.control.compaction.CompactionProcessor",
    "token_budget": "harnessx.processors.control.token_budget.TokenBudgetProcessor",
    "tool_failure_guard": "harnessx.processors.control.tool_failure_guard.ToolFailureGuard",
    "sycophancy_detector": "harnessx.processors.control.sycophancy_detector.SycophancyDetector",
    "model_router": "harnessx.processors.multi_model.model_router.ModelRouterProcessor",
    # ── Evaluation processors ─────────────────────────────────────────────────
    "evaluation_proc": "harnessx.processors.evaluation.evaluation.EvaluationProcessor",
    # ── Observability processors ──────────────────────────────────────────────
    "otel": "harnessx.processors.observability.otel_proc.OTelProcessor",
    "checkpoint": "harnessx.processors.observability.checkpoint.CheckpointProcessor",
    # ── Tools processors ──────────────────────────────────────────────────────
    "skill_loader": "harnessx.processors.tools.skill_loader.ProgressiveSkillLoader",
}


def _expand_type(cfg: dict) -> dict:
    """Expand ``{"type": "short_name", ...}`` to ``{"_target_": "...", ...}``.

    Returns *cfg* unchanged when it already has ``_target_``.
    """
    if "_target_" in cfg:
        return cfg
    if "type" not in cfg:
        raise ValueError(f"Config dict must have '_target_' or 'type': {cfg!r}")
    cfg = dict(cfg)
    short = cfg.pop("type")
    if short not in NAMES:
        raise KeyError(
            f"Unknown short name {short!r}. "
            f"Available names: {sorted(NAMES)}. "
            "Use the full '_target_' path or add the name to harnessx.core.builder.NAMES."
        )
    cfg["_target_"] = NAMES[short]
    return cfg


def _instantiate(cfg: "dict | None", default_factory=None) -> "Any":
    """Instantiate a ``{"_target_": "fully.qualified.ClassName", **kwargs}`` dict.

    Also accepts ``{"type": "short_name", **kwargs}`` — resolved via :data:`NAMES`.

    Args:
        cfg:             Config dict with ``_target_`` (or ``type``), or ``None``.
        default_factory: Zero-arg callable used when *cfg* is ``None``.

    Returns:
        The instantiated object, or ``None``.
    """
    import importlib as _importlib

    if cfg is None:
        return default_factory() if default_factory is not None else None

    cfg = _expand_type(dict(cfg))

    # Recursively instantiate nested _target_ / type dicts in kwargs.
    # Keys starting with "_" are metadata (for example "_code_hash") and
    # should never be forwarded to class constructors.
    resolved: dict = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and ("_target_" in v or "type" in v):
            try:
                resolved[k] = _instantiate(v)
            except Exception:
                # Graceful degradation for specs that survived serialization
                # but lost essential state (e.g. ModelConfig without providers).
                _log.warning("_instantiate: nested spec for %r failed, using None", k)
                resolved[k] = None
        else:
            resolved[k] = v

    def _parse_file_target(_target: str) -> tuple[str, str]:
        spec = _target[len("file://") :]
        path_part, sep, class_name = spec.rpartition("::")
        if not sep or not path_part.strip() or not class_name.strip():
            raise ValueError("invalid file target; expected 'file:///abs/path.py::ClassName'")
        return path_part, class_name.strip()

    target = cfg["_target_"]
    # File-based direct target:
    #   file:///abs/path/to/processor.py::MyProcessor
    if isinstance(target, str) and target.startswith("file://"):
        import uuid as _uuid

        file_path, class_name = _parse_file_target(target)
        spec = _importlib.util.spec_from_file_location(
            f"_hx_custom_file_{_uuid.uuid4().hex}",
            str(file_path),
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot build import spec from {file_path}")
        mod = _importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name)
    else:
        module_path, class_name = target.rsplit(".", 1)
        try:
            mod = _importlib.import_module(module_path)
        except ModuleNotFoundError:
            # Retry once after injecting legacy managed custom-processor path.
            # Legacy support for historical targets:
            #   hx_custom_processors.<slug>.processor.<ClassName>
            if module_path.startswith("hx_custom_processors."):
                import sys as _sys
                from pathlib import Path as _Path

                legacy_root = _Path.home() / ".harnessx" / "lab" / "custom_processors" / "py"
                p = str(legacy_root.resolve())
                if p not in _sys.path:
                    _sys.path.insert(0, p)
                mod = _importlib.import_module(module_path)
            else:
                raise
        cls = getattr(mod, class_name)
    obj = cls(**resolved)
    try:
        setattr(obj, "__hx_target__", str(target))
    except Exception:
        pass
    # Preserve constructor kwargs for stable round-trip serialization.
    # Some processors store transformed values under different attribute names
    # (e.g. blocked_patterns -> _patterns), so __init__ params cannot always
    # be reconstructed by attribute introspection alone.
    try:
        import copy as _copy

        setattr(obj, "__hx_init_kwargs__", _copy.deepcopy(resolved))
    except Exception:
        try:
            setattr(obj, "__hx_init_kwargs__", dict(resolved))
        except Exception:
            pass
    return obj


def build_from_config(d: dict) -> "HarnessConfig":
    """Build a :class:`~harnessx.core.harness.HarnessConfig` from a flat processor-list dict.

    The dict must have a ``processors:`` key containing a list of
    ``_target_`` dicts (or ``type:`` short-name dicts).  Each processor is
    instantiated and added to a :class:`HarnessBuilder` via ``builder.add()``.

    Optional top-level keys:

    ``plugins:``
        List of plugin specs forwarded to ``builder.plugin()``.

    Model is intentionally **excluded** — always build :class:`~harnessx.core.model_config.ModelConfig`
    separately and combine via::

        harness_config = build_from_config(d)
        agent = model_config.agentic(harness_config)

    Example::

        from harnessx.core.builder import build_from_config

        config = build_from_config({
            "processors": [
                {"type": "system_prompt"},
                {"type": "loop_detection"},
                {
                    "_target_": "harnessx.processors.memory.memory_retrieval.MemoryRetrievalProcessor",
                    "memory": {"type": "sliding_window", "n": 20},
                    "top_k": 10,
                },
            ],
        })
    """
    from .harness import HarnessConfig  # noqa: F401 (TYPE_CHECKING guard)
    from ..processors.control.todo_check import TodoWriteEnforcer, make_todo_tool

    builder = HarnessBuilder()

    procs_node = d.get("processors", [])
    if isinstance(procs_node, dict):
        raise ValueError(
            "'processors:' must be a flat list of _target_ dicts, not a "
            "hook-bucket dict. Use: "
            "processors: [{_target_: ...}, {_target_: ...}]. "
            "Processors self-declare their hook via _hook or "
            "MultiHookProcessor. "
            f"Got dict with keys {sorted(procs_node)!r}."
        )

    for proc_cfg in procs_node:
        _target = proc_cfg.get("_target_", "") if isinstance(proc_cfg, dict) else ""
        try:
            proc = _instantiate(proc_cfg)
        except (ImportError, ModuleNotFoundError) as exc:
            # Narrow legacy allowance: CLI-local classes (e.g. _CLIToolPrinter)
            # whose qualname contains "<locals>" can only exist in a function
            # scope and never round-trip. Anything else is a real config error
            # and must surface — silent drops have caused capabilities to
            # disappear invisibly across YAML round-trips.
            if "<locals>" in _target or "._<locals>" in _target:
                _log.warning("build_from_config: skipping CLI-local processor %s", _target)
                continue
            raise ImportError(f"cannot import processor {_target!r}: {type(exc).__name__}: {exc}") from exc
        if proc is not None:
            builder = builder.add(proc)
            # TodoCheck (aka TodoWriteEnforcer) requires its companion tool
            if isinstance(proc, TodoWriteEnforcer):
                builder = builder.add_tool(make_todo_tool())

    if d.get("plugins"):
        for plugin_spec in d["plugins"]:
            builder = builder.plugin(plugin_spec)

    for key, val in d.get("slots", {}).items():
        if key not in _SCALAR_SLOTS:
            _log.warning("build_from_config: unknown slot '%s', skipping", key)
            continue
        if isinstance(val, dict) and ("_target_" in val or "type" in val):
            val = _instantiate(val)
        builder = builder.slot(**{key: val})

    return builder.build()


def _topological_sort_entries(entries: list[_ProcEntry]) -> list[_ProcEntry]:
    """Sort entries by _order, then topologically within equal-order buckets via _after.

    Raises :exc:`HarnessConflictError` when:
    - An ``_after`` dep points to a processor with a *higher* ``_order`` value
      (the constraint can never be satisfied — fail fast at build time).
    - ``_after`` edges within the same ``_order`` bucket form a cycle.

    Soft deps: ``_after`` references to unregistered singleton_groups are silently
    ignored, so processors can declare optional ordering without coupling to
    plugins that may or may not be present.
    """
    from collections import defaultdict, deque

    # Build group → entry lookup (only entries that have a singleton_group)
    group_map: dict[str, _ProcEntry] = {e.singleton_group: e for e in entries if e.singleton_group}

    # Detect cross-bucket contradictions: _after target has a *higher* order value
    conflicts: list[str] = []
    for entry in entries:
        for after_group in entry.after:
            target = group_map.get(after_group)
            if target is None:
                continue
            if target.order > entry.order:
                conflicts.append(
                    f"{type(entry.processor).__name__} (_order={entry.order}) declares "
                    f"_after=['{after_group}'] but {type(target.processor).__name__} "
                    f"has _order={target.order} — contradictory: target runs later"
                )
    if conflicts:
        raise HarnessConflictError(conflicts)

    # Group by order value and sort buckets
    order_buckets: dict[int, list[_ProcEntry]] = {}
    for entry in entries:
        order_buckets.setdefault(entry.order, []).append(entry)

    result: list[_ProcEntry] = []
    for order_val in sorted(order_buckets):
        bucket = order_buckets[order_val]
        n = len(bucket)
        if n == 1:
            result.append(bucket[0])
            continue

        # Kahn's topological sort within this bucket
        entry_to_idx = {id(e): i for i, e in enumerate(bucket)}
        adj: dict[int, list[int]] = defaultdict(list)  # i → [j]: i must come before j
        in_deg = [0] * n

        for j, entry in enumerate(bucket):
            for after_group in entry.after:
                target = group_map.get(after_group)
                if target is None or id(target) not in entry_to_idx:
                    continue  # not in this bucket; cross-bucket order handles it
                i = entry_to_idx[id(target)]
                adj[i].append(j)
                in_deg[j] += 1

        queue = deque(i for i in range(n) if in_deg[i] == 0)
        sorted_bucket: list[_ProcEntry] = []
        while queue:
            i = queue.popleft()
            sorted_bucket.append(bucket[i])
            for j in adj[i]:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)

        if len(sorted_bucket) != n:
            cycle_procs = [type(bucket[i].processor).__name__ for i in range(n) if in_deg[i] > 0]
            raise HarnessConflictError(
                [f"Cycle in _after dependencies at _order={order_val}: {', '.join(cycle_procs)}"]
            )

        result.extend(sorted_bucket)

    return result
