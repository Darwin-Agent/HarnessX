# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import atexit
import asyncio
import copy
import inspect
import logging
import weakref
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

_log = logging.getLogger(__name__)

from .config_schema import (
    ToolRegistryConfig,
    TracerConfig,
    NullTracerConfig,
    WorkspaceConfig,
    SandboxConfig,
)
from .events import make_run_id
from .processor import Processor
from .runloop import run_loop
from .state import State

# Sentinel for missing attributes in processor serialization
_MISSING = object()
_PRIMITIVES = (bool, int, float, str, bytes, type(None))
_RUNTIME_ONLY_INIT_FIELDS = {
    # Bound at runtime; not part of persisted harness behaviour config.
    "model_config",
    "harness_config",
    "tool_registry",
    "workspace",
    "sub_harnesses",
    "verdict_sink",
}

_ACTIVE_HARNESSES: "weakref.WeakSet[Harness]" = weakref.WeakSet()


def _best_effort_cleanup_active_harnesses() -> None:
    """Best-effort process-exit cleanup for still-live harness instances."""
    live = [h for h in list(_ACTIVE_HARNESSES) if getattr(h, "_closed", True) is False]
    if not live:
        return

    async def _cleanup_all() -> None:
        for h in live:
            try:
                await h.cleanup()
            except Exception:
                pass

    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass

    try:
        asyncio.run(_cleanup_all())
    except Exception:
        pass


atexit.register(_best_effort_cleanup_active_harnesses)


def _hash_processor_code(proc: Any) -> str:
    """Compute a semantic hash of the processor's class source code.

    Uses ``ast.parse`` + ``ast.unparse`` to produce a canonical, comment-free
    and docstring-free representation before hashing.  This means:

    - Changing comments or docstrings → hash unchanged
    - Changing any method logic (including helpers called by on_xxx) → hash changes
    - Reformatting / renaming local variables → hash changes

    Works for both built-in HarnessX processors and developer-defined ones.
    Returns an empty string if source cannot be retrieved (e.g. C extensions).
    """
    import ast as _ast
    import hashlib as _hashlib
    import inspect as _inspect
    import textwrap as _textwrap

    try:
        source = _textwrap.dedent(_inspect.getsource(type(proc)))
        tree = _ast.parse(source)

        # Strip docstrings from all scopes so documentation changes
        # don't affect the semantic hash.
        for node in _ast.walk(tree):
            if not isinstance(
                node,
                (_ast.Module, _ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef),
            ):
                continue
            if (
                node.body
                and isinstance(node.body[0], _ast.Expr)
                and isinstance(node.body[0].value, _ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)

        canonical = _ast.unparse(tree)
        return "sha256:" + _hashlib.sha256(canonical.encode()).hexdigest()[:16]
    except Exception:
        return ""


def _serialize_processor(proc: Any) -> "dict | None":
    """Serialize a processor instance to a _target_-format dict for YAML export.

    Inspects ``__init__`` parameters and reads matching instance attributes,
    trying both ``self.name`` and ``self._name`` conventions.  Callables and
    objects that cannot be represented as plain scalars are omitted unless they
    can themselves be recursively serialized.
    """
    import inspect as _inspect

    # Runtime-only processors (e.g. MCP task_start hot-reload trigger) should
    # never be persisted into descriptor YAML.
    if bool(getattr(proc, "__hx_runtime_only__", False)) or bool(getattr(type(proc), "__hx_runtime_only__", False)):
        return None

    custom_target = getattr(proc, "__hx_target__", None)
    if isinstance(custom_target, str) and custom_target.strip():
        target = custom_target.strip()
    else:
        cls = type(proc)
        module = getattr(cls, "__module__", None) or ""
        qualname = getattr(cls, "__qualname__", None) or ""
        if not module or module.startswith("_"):
            return None
        # Skip non-importable local classes (e.g. _chat.<locals>._CLIToolPrinter).
        if "<" in qualname:
            return None
        target = f"{module}.{qualname}"

    cls = type(proc)
    result: dict = {"_target_": target}

    # Semantic code hash: detects logic changes regardless of comments/formatting.
    # Covers the entire class so helper methods called by on_xxx are also tracked.
    code_hash = _hash_processor_code(proc)
    if code_hash:
        result["_code_hash"] = code_hash

    try:
        sig = _inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return result
    init_kwargs = getattr(proc, "__hx_init_kwargs__", None)
    if not isinstance(init_kwargs, dict):
        init_kwargs = {}

    skip_params: frozenset = getattr(type(proc), "__hx_skip_serialization__", frozenset())
    for pname, _ in sig.parameters.items():
        if pname == "self":
            continue
        if pname.startswith("_") or pname in _RUNTIME_ONLY_INIT_FIELDS or pname in skip_params:
            continue
        # Try self.name then self._name (two common storage conventions)
        val = getattr(proc, pname, _MISSING)
        if val is _MISSING:
            val = getattr(proc, f"_{pname}", _MISSING)
        if val is _MISSING and pname in init_kwargs:
            val = init_kwargs[pname]
        if val is _MISSING:
            continue
        # ModelConfig is supplied separately from HarnessConfig and must not be
        # embedded inside processor kwargs in exported harness YAML.
        if type(val).__name__ == "ModelConfig" and type(val).__module__.endswith("core.model_config"):
            continue
        # Skip callables (functions, lambdas) AND bare type/class objects.
        if callable(val):
            continue
        # Skip None — it means "not provided / use default"; serialising null
        # into YAML is noisy and can cause reload failures when the constructor
        # validates that the param is provided.
        if val is None:
            continue
        if isinstance(val, _PRIMITIVES):
            result[pname] = val
        elif isinstance(val, (list, tuple)):
            # Drop lists containing callables (functions, classes, lambdas).
            items = list(val)
            if any(callable(x) for x in items):
                continue
            result[pname] = items
        elif isinstance(val, (set, frozenset)):
            if all(isinstance(x, _PRIMITIVES) for x in val):
                result[pname] = sorted(val) if all(isinstance(x, str) for x in val) else list(val)
        elif isinstance(val, dict):
            # Only serialize dicts whose keys and values are YAML-safe primitives.
            if all(isinstance(k, str) for k in val) and all(isinstance(v, _PRIMITIVES) for v in val.values()):
                result[pname] = dict(val)
        else:
            # Strategy / nested processor — recurse
            nested = _serialize_processor(val)
            if nested:
                result[pname] = nested

    return result


def _serialize_plugin(plugin: Any) -> "dict | None":
    """Serialize a HarnessPlugin instance to a _target_-format dict.

    Directory plugins (have ``_plugin_root``) are serialized as
    ``{"path": ...}``.  Python class plugins are serialized as
    ``{"_target_": "module.ClassName", **init_kwargs}``.
    Plugins defined in ``__main__`` or local scopes are skipped.
    """
    import inspect as _inspect

    plugin_root = getattr(plugin, "_plugin_root", None)
    if plugin_root is not None:
        return {"path": str(plugin_root)}

    cls = type(plugin)
    module = getattr(cls, "__module__", None) or ""
    qualname = getattr(cls, "__qualname__", None) or ""
    if not module or module.startswith("_") or "<" in qualname:
        return None

    result: dict = {"_target_": f"{module}.{qualname}"}

    try:
        sig = _inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return result

    for pname, _ in sig.parameters.items():
        if pname == "self" or pname.startswith("_"):
            continue
        val = getattr(plugin, pname, _MISSING)
        if val is _MISSING:
            val = getattr(plugin, f"_{pname}", _MISSING)
        if val is _MISSING:
            continue
        if isinstance(val, _PRIMITIVES):
            result[pname] = val
        elif isinstance(val, (list, tuple)):
            if not any(callable(x) for x in val):
                result[pname] = list(val)
        elif isinstance(val, dict):
            if all(isinstance(k, str) for k in val) and all(isinstance(v, _PRIMITIVES) for v in val.values()):
                result[pname] = dict(val)

    return result


def _write_resolved_config(
    config: "HarnessConfig",
    model_config: "Any",
    workspace_root: "Any",
) -> str:
    """Write the resolved harness config for reproduction and wake() recovery.

    Writes only harness-specific configuration (tools, processors, workspace,
    sandbox).  Model config is intentionally excluded — it lives separately in
    ``~/.harnessx/model_config.yaml``.

    Two files are produced:

    ``~/.harnessx/configs/{sha256}.yaml``
        Global content-addressed store.  Written once, never overwritten.
        Session index files reference this path so ``wake_config()`` can
        reconstruct the harness config after a process restart regardless of
        which workspace is active.

    ``{workspace_root}/harness_config.yaml``
        Workspace-local runtime snapshot.  Overwritten on every fresh run so
        it always reflects the active config for *this* workspace execution.
        This is separate from the agent-shared default config at
        ``AGENT_HOME/workspaces/{agent_id}/harness_config.yaml`` used by Lab UI
        and CLI default loading.

    Returns:
        sha256 hex digest of the config content (stored as config_hash in
        session state snapshots).
    """
    import hashlib as _hashlib
    from pathlib import Path as _Path

    # Use to_yaml() so runtime objects (InMemoryToolRegistry, Workspace, etc.)
    # are properly serialized before calling OmegaConf.structured().
    yaml_content = config.to_yaml()

    config_hash = _hashlib.sha256(yaml_content.encode()).hexdigest()

    # 1. Global content-addressed store — immutable once written.
    from ..home import agent_configs_dir as _acd

    hashed_path = _acd() / f"{config_hash}.yaml"
    if not hashed_path.exists():
        hashed_path.write_text(yaml_content, encoding="utf-8")

    # 2. Workspace-local runtime snapshot — scoped to this workspace so
    #    library/test calls never pollute AGENT_HOME shared defaults.
    ws_root = _Path(workspace_root)
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "harness_config.yaml").write_text(yaml_content, encoding="utf-8")

    return config_hash


def _find_journal(tracer: "Any") -> "Any | None":
    """Walk the tracer chain and return the first HarnessJournal found.

    Tracers may be wrapped (e.g. SSETracer wraps HarnessJournal via ``_inner``).
    harness.run() needs to interact with the journal directly for session
    management (session_id, wake(), config_hash) regardless of wrapping.
    """
    if tracer is None:
        return None
    from ..tracing.journal import HarnessJournal as _HJ

    if isinstance(tracer, _HJ):
        return tracer
    return _find_journal(getattr(tracer, "_inner", None))


if TYPE_CHECKING:
    from .events import TaskEndEvent, ToolCall
    from .trajectory import StatefulTrajectory


# ---------------------------------------------------------------------------
# Runtime container — created from HarnessConfig by _instantiate_runtime()
# ---------------------------------------------------------------------------


@dataclass
class _HarnessRuntime:
    """Live objects instantiated from a HarnessConfig at Harness.__init__ time."""

    tool_registry: Any  # InMemoryToolRegistry | MCPToolRegistry
    tracer: Any  # HarnessJournal | NullTracer | SSETracer
    processors: dict  # dict[str, list[Processor]] — hook-keyed
    workspace: Any  # Workspace | None
    sandbox_provider: Any  # SandboxProvider
    plugins: list  # list[HarnessPlugin]


def _instantiate_proc(d: dict) -> "Any | None":
    """Instantiate a processor from a _target_ dict using builder._instantiate."""
    try:
        from .builder import _instantiate

        return _instantiate(d)
    except Exception:
        return None


def _route_processors(flat: list) -> "dict[str, list]":
    """Route a flat list of Processor instances into a hook-keyed dict."""
    from .processor import MultiHookProcessor

    result: dict = {}
    for proc in flat:
        # Explicit hook override wins (set at build time or from _hook_ dict field)
        hook_override = getattr(proc, "__hx_hook_override__", None)
        if hook_override:
            result.setdefault(hook_override, []).append(proc)
        elif isinstance(proc, MultiHookProcessor):
            result.setdefault("*", []).append(proc)
        else:
            hook = getattr(proc, "_hook", None) or getattr(type(proc), "_hook", None) or "*"
            result.setdefault(hook, []).append(proc)
    return result


def _runtime_registry_to_config(registry: Any) -> "ToolRegistryConfig":
    """Convert a runtime InMemoryToolRegistry to a ToolRegistryConfig.

    Preserves the *original* target of each custom tool when available:

    * ``__hx_target__`` set by :func:`_build_tool_registry_from_config`
      takes precedence — this lets configs loaded from
      ``file:///abs/path.py::symbol`` round-trip back to the same URI
      instead of a synthetic, unimportable module name.
    * Otherwise, fall back to ``{fn.__module__}.{fn.__qualname__}`` for
      tools backed by real, importable modules.
    * Tools backed by a synthetic module (``_hx_custom_*``) without a
      recorded target are surfaced as a WARNING rather than silently
      misclassified as ``builtin`` — previously they ended up being
      dropped on the next load, because the default builtin set
      doesn't contain them.
    """
    if isinstance(registry, ToolRegistryConfig):
        return registry
    builtin_names: list = []
    custom_targets: list = []
    tools_attr = getattr(registry, "_tools", {})
    if isinstance(tools_attr, dict):
        from ..tools.builtin import build_default_tools as _bdt

        try:
            _default_names: set = set(getattr(_bdt(), "_tools", {}).keys())
        except Exception:
            _default_names = set()
        for name, tool_obj in tools_attr.items():
            recorded_target = getattr(tool_obj, "__hx_target__", None)
            fn = getattr(tool_obj, "fn", None)
            mod = getattr(fn, "__module__", None) or "" if fn else ""
            qual = getattr(fn, "__qualname__", None) or "" if fn else ""

            if isinstance(recorded_target, str) and recorded_target:
                # Custom tool we loaded ourselves — preserve the exact
                # target (file:// URI or dotted path) so a round-trip
                # through YAML reproduces it.
                custom_targets.append(recorded_target)
            elif name in _default_names:
                builtin_names.append(name)
            elif fn and mod and not mod.startswith("_") and "<" not in qual:
                custom_targets.append(f"{mod}.{qual}")
            else:
                # Synthetic module (file-loaded tool registered outside
                # our loader) or otherwise not round-trippable.
                _log.warning(
                    "tool_registry: tool %r has no recorded __hx_target__ and "
                    "its function lives in a non-importable module "
                    "(module=%r, qualname=%r); serializing as a builtin "
                    "name-only entry, which will fail to load on YAML "
                    "round-trip unless %r is added to the default builtin set.",
                    name,
                    mod,
                    qual,
                    name,
                )
                builtin_names.append(name)
    return ToolRegistryConfig(builtin=builtin_names, custom=custom_targets)


def _runtime_workspace_to_config(workspace: Any) -> "WorkspaceConfig":
    """Convert a runtime Workspace instance to a WorkspaceConfig."""
    root = getattr(workspace, "root", None)
    home = getattr(workspace, "home", None)
    return WorkspaceConfig(
        root=str(root) if root is not None else None,
        agent_id=getattr(workspace, "agent_id", "hxagent"),
        project=getattr(workspace, "project", "default"),
        mode=getattr(workspace, "mode", None),
        home=str(home) if home is not None else None,
        parent_id=getattr(workspace, "parent_id", None),
    )


def _parse_file_tool_target(target: str) -> tuple[str, str]:
    """Parse a ``file:///abs/path.py::symbol`` tool target.

    Mirrors the file-URI form supported for processors in
    :mod:`harnessx.core.builder`. Raises ``ValueError`` on malformed input.
    """
    spec = target[len("file://") :]
    path_part, sep, sym_name = spec.rpartition("::")
    if not sep or not path_part.strip() or not sym_name.strip():
        raise ValueError(f"invalid file target: {target!r} (expected 'file:///abs/path.py::symbol')")
    return path_part, sym_name.strip()


def _build_tool_registry_from_config(cfg: ToolRegistryConfig) -> Any:
    """Build an InMemoryToolRegistry from a ToolRegistryConfig descriptor.

    Supported ``custom`` target forms:
      * ``module.path.sub.symbol`` — dotted import path.
      * ``file:///abs/path/to/file.py::symbol`` — absolute-path file URI,
        loaded via :func:`importlib.util.spec_from_file_location` (same
        mechanism the processor loader uses in
        :mod:`harnessx.core.builder`).

    Load failures are logged at WARNING level rather than silently
    discarded so the evolver / runner can surface missing tools instead
    of handing the model an incomplete tool schema.
    """
    from ..tools.inmemory import InMemoryToolRegistry

    registry = InMemoryToolRegistry()
    if cfg.builtin:
        from ..tools.builtin import build_default_tools

        defaults = build_default_tools()
        all_tools = getattr(defaults, "_tools", {})
        for name in cfg.builtin:
            tool = all_tools.get(name)
            if tool is not None:
                registry.register(tool)
            else:
                _log.warning(
                    "tool_registry.builtin: %r is not in the default builtin "
                    "set (%s); skipping. Move it to tool_registry.custom with "
                    "a module path or file:// URI if it is a custom tool.",
                    name,
                    sorted(all_tools.keys()),
                )
    if cfg.custom:
        import importlib as _importlib
        import importlib.util as _importlib_util
        import uuid as _uuid

        for target in cfg.custom:
            try:
                if isinstance(target, str) and target.startswith("file://"):
                    path_part, sym_name = _parse_file_tool_target(target)
                    spec = _importlib_util.spec_from_file_location(f"_hx_custom_tool_{_uuid.uuid4().hex}", path_part)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"cannot build import spec from {path_part!r}")
                    mod = _importlib_util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    fn = getattr(mod, sym_name)
                else:
                    mod_path, sym_name = target.rsplit(".", 1)
                    mod = _importlib.import_module(mod_path)
                    fn = getattr(mod, sym_name)
                registry.register(fn)
                # Record the original target so later serialization
                # (via _runtime_registry_to_config) can reproduce the
                # same ``custom:`` entry, including file:// URIs whose
                # synthetic module name is meaningless outside this
                # process.
                try:
                    setattr(fn, "__hx_target__", target)
                except Exception:
                    # Tool is a dataclass; setattr should work, but if
                    # someone passed a frozen/slotted object we still
                    # want the registration itself to stand.
                    pass
            except Exception as exc:
                _log.warning(
                    "tool_registry.custom: failed to load %r: %s",
                    target,
                    exc,
                )
    return registry


def _instantiate_runtime(config: "HarnessConfig") -> _HarnessRuntime:
    """Instantiate all runtime objects from a HarnessConfig."""
    from ..tools.inmemory import InMemoryToolRegistry
    from ..tracing.journal import HarnessJournal
    from ..sandbox.local import LocalSandboxProvider

    # ── Tool registry ────────────────────────────────────────────────────────
    tr = config.tool_registry
    if tr is None:
        tool_registry = InMemoryToolRegistry()
    elif isinstance(tr, ToolRegistryConfig):
        tool_registry = _build_tool_registry_from_config(tr)
    else:
        tool_registry = tr

    # ── Tracer ───────────────────────────────────────────────────────────────
    tc = config.tracer
    if tc is None:
        tracer = HarnessJournal()
    elif isinstance(tc, TracerConfig):
        target = getattr(tc, "_target_", "harnessx.tracing.journal.HarnessJournal")
        if target == "harnessx.tracing.journal.HarnessJournal":
            kw: dict = dict(export_jsonl=tc.export_jsonl, silent=tc.silent, session_id=tc.session_id)
            if tc.base_dir is not None:
                kw["base_dir"] = tc.base_dir
            tracer = HarnessJournal(**kw)
        else:
            # NullTracerConfig or custom tracer — dispatch via _target_
            import importlib as _il

            _mod_path, _cls_name = target.rsplit(".", 1)
            tracer = getattr(_il.import_module(_mod_path), _cls_name)()
    else:
        tracer = tc

    # ── Processors ───────────────────────────────────────────────────────────
    flat: list = []
    # config.processors contains only _target_ dicts (enforced by __post_init__).
    for p in config.processors or []:
        if isinstance(p, dict) and "_target_" in p:
            hook_override = p.get("_hook_")
            inst = _instantiate_proc(p)
            if inst is not None:
                if hook_override:
                    inst.__hx_hook_override__ = hook_override
                flat.append(inst)
    # _rt_procs holds runtime-only processor instances that cannot be serialized.
    for p in getattr(config, "_rt_procs", None) or []:
        flat.append(p)
    proc_dict = _route_processors(flat)

    # ── Workspace ────────────────────────────────────────────────────────────
    wc = config.workspace
    if wc is None:
        workspace = None
    elif isinstance(wc, WorkspaceConfig):
        from ..workspace.workspace import Workspace
        from pathlib import Path as _Path

        if wc.root is not None:
            workspace = Workspace(
                root=_Path(wc.root),
                agent_id=wc.agent_id,
                project=wc.project,
                mode=wc.mode,
                home=_Path(wc.home) if wc.home is not None else None,
                parent_id=wc.parent_id,
            )
        elif wc.home is not None:
            workspace = Workspace(
                agent_id=wc.agent_id,
                project=wc.project,
                mode=wc.mode,
                home=_Path(wc.home),
                parent_id=wc.parent_id,
            )
        else:
            workspace = None
    else:
        workspace = wc

    # ── Sandbox provider ─────────────────────────────────────────────────────
    sp = getattr(config, "_rt_sandbox", None) or config.sandbox_provider
    if sp is None:
        sandbox_provider = LocalSandboxProvider()
    elif isinstance(sp, SandboxConfig):
        try:
            import importlib as _importlib

            mod_path, cls_name = sp._target_.rsplit(".", 1)
            mod = _importlib.import_module(mod_path)
            sandbox_provider = getattr(mod, cls_name)()
        except Exception:
            sandbox_provider = LocalSandboxProvider()
    else:
        sandbox_provider = sp

    # ── Plugins ──────────────────────────────────────────────────────────────
    plugins: list = []
    for p in config.plugins or []:
        if not isinstance(p, dict):
            plugins.append(p)
            continue
        path = p.get("path") or ""
        target = p.get("_target_") or ""
        kwargs = {k: v for k, v in p.items() if k not in ("_target_", "path", "_code_hash")}
        try:
            from ..plugins.loader import load_plugin

            if path:
                plugins.append(load_plugin(path))
            elif target:
                import importlib as _importlib

                mod_path, cls_name = target.rsplit(".", 1)
                cls = getattr(_importlib.import_module(mod_path), cls_name)
                plugins.append(cls(**kwargs))
        except Exception:
            pass

    # Wire processor-providing plugins that were instantiated from YAML dicts.
    # Plugins mounted via _mount_plugin() already have their processors in
    # _rt_procs, so they appear in proc_dict.  Only add processors that are
    # not already present (checked by object identity) to avoid duplicates.
    existing_proc_ids = {id(p) for procs in proc_dict.values() for p in procs}
    for plugin in plugins:
        for proc in getattr(plugin, "processors", []) or []:
            if id(proc) not in existing_proc_ids:
                for hook, hook_procs in _route_processors([proc]).items():
                    proc_dict.setdefault(hook, []).extend(hook_procs)
                existing_proc_ids.add(id(proc))

    return _HarnessRuntime(
        tool_registry=tool_registry,
        tracer=tracer,
        processors=proc_dict,
        workspace=workspace,
        sandbox_provider=sandbox_provider,
        plugins=plugins,
    )


# ---------------------------------------------------------------------------
# Task / Result
# ---------------------------------------------------------------------------


@dataclass
class BaseTask:
    description: str | list  # str for text-only; list = Anthropic content blocks
    success_criteria: str = ""
    max_steps: int = 50
    token_budget: int | None = None
    max_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    interrupt_on: list[str] = field(default_factory=list)
    """Tool names that trigger an interrupt/pause.
    When the model calls one of these tools, run_loop exits with
    exit_reason='interrupted' and HarnessResult.interrupted_at set.
    Resume by calling harness.run(task, _resume_state=result.resume_state).
    """
    force_compact: bool = False
    spawn_depth: int = 0

    def is_done(self, state: State) -> bool:
        """Subclasses can override to check state-based completion."""
        return False


@dataclass
class HarnessResult:
    """
    Harness.run() unified return value.
    trajectory is a first-class product, not a log side-product.
    """

    task_end: "TaskEndEvent"
    trajectory: "StatefulTrajectory"
    interrupted_at: "ToolCall | None" = None
    resume_state: "State | None" = None

    def __getattr__(self, name: str) -> object:
        task_end = object.__getattribute__(self, "task_end")
        if hasattr(task_end, name):
            return getattr(task_end, name)
        raise AttributeError(f"'HarnessResult' object has no attribute '{name}'")

    @property
    def is_interrupted(self) -> bool:
        return self.interrupted_at is not None


# ---------------------------------------------------------------------------
# HarnessConfig
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """Harness configuration — pure behaviour pipeline.

    Defines *what the agent does* (processors, tools, workspace, tracer) but
    carries no model information.  Combine with a ModelConfig to produce a
    runnable agent::

        agent = model_config.agentic(harness_config)

    All fields are serializable config objects.  Runtime objects are never
    stored here; ``_instantiate_runtime()`` converts them to live instances.
    """

    tool_registry: "Optional[ToolRegistryConfig]" = None
    tracer: "Optional[TracerConfig]" = None
    # Flat list of _target_ dicts only.  Runtime processor instances are kept
    # in the non-field attribute _rt_procs (set by __post_init__).
    processors: "list[dict]" = field(default_factory=list)
    workspace: "Optional[WorkspaceConfig]" = None

    workspace_template: str = "default"
    init_workspace: bool = True

    # step_snapshots=False skips storing large message data in trajectory steps:
    #   • FullStateSnapshot.messages = ()  (avoids O(n²) memory across n steps)
    #   • StepEndEvent.state_snapshot = None
    #   • TrajectoryStep.step_start_event = None
    # Disable in RL training where reward_func never reads step snapshots.
    step_snapshots: bool = True

    sandbox_provider: "Optional[SandboxConfig]" = None
    # Stable ID for warm-pool sandbox reuse across Harness.run() calls.
    sandbox_hint_id: "Optional[str]" = None

    # list of PluginConfig dicts — populated by HarnessBuilder.build().
    # setup(config) called in Harness.__init__(); stop() during cleanup().
    plugins: "list[Any]" = field(default_factory=list)

    # ── Gate: normalize all fields at construction time ───────────────────────

    def __post_init__(self) -> None:
        # workspace: runtime Workspace → WorkspaceConfig
        ws = self.workspace
        if ws is not None and not isinstance(ws, WorkspaceConfig):
            self.workspace = _runtime_workspace_to_config(ws)

        # tracer: convert only the exact standard types to their config equivalents.
        # Subclasses may have custom on_event behaviour and must not be replaced.
        tc = self.tracer
        if tc is not None and not isinstance(tc, TracerConfig):
            from ..tracing.journal import HarnessJournal as _HJ
            from ..tracing.null_tracer import NullTracer as _NT

            if type(tc) is _HJ:
                self.tracer = TracerConfig(
                    base_dir=str(tc.base_dir) if tc.base_dir is not None else None,
                    export_jsonl=tc.export_jsonl,
                    silent=tc.silent,
                    session_id=getattr(tc, "session_id", None),
                )
            elif type(tc) is _NT:
                self.tracer = NullTracerConfig()
            # All other runtime tracers (including subclasses) are kept as-is.

        # sandbox_provider: stash non-SandboxConfig runtime providers so
        # OmegaConf never sees them. _build_runtime_harness reads _rt_sandbox
        # when instantiating the actual sandbox.
        sp = self.sandbox_provider
        if sp is not None and not isinstance(sp, SandboxConfig):
            self._rt_sandbox = sp
            self.sandbox_provider = None
        elif not hasattr(self, "_rt_sandbox"):
            self._rt_sandbox = None

        # processors: separate serializable dicts from runtime-only instances.
        # Runtime instances live in the non-dataclass attribute _rt_procs so
        # OmegaConf / to_yaml() never sees them.
        existing_rt: list = [] if not hasattr(self, "_rt_procs") else list(self._rt_procs)
        dicts: list = []
        for p in self.processors:
            if isinstance(p, dict):
                dicts.append(p)
            else:
                existing_rt.append(p)
        self.processors = dicts
        self._rt_procs: list = existing_rt

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def required_model_keys(self) -> frozenset:
        required: set = set()
        for proc in getattr(self, "_rt_procs", None) or []:
            rk = getattr(type(proc), "required_model_keys", None)
            if rk:
                required.update(rk)
        return frozenset(required)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_yaml(self) -> str:
        """Serialize to a YAML string (harness config only, no model info).

        Reload with ``HarnessConfig.from_yaml(yaml_str)``.
        """
        from omegaconf import OmegaConf
        import dataclasses as _dc

        header = (
            "# HarnessX HarnessConfig (behaviour pipeline — no model)\n"
            "# Combine: model_config.agentic(HarnessConfig.from_yaml(this))\n\n"
        )
        # Serialize plugins: keep dicts as-is, convert runtime instances via _serialize_plugin
        serialized_plugins = []
        for p in self.plugins:
            if isinstance(p, dict):
                serialized_plugins.append(p)
            else:
                d = _serialize_plugin(p)
                if d is not None:
                    serialized_plugins.append(d)

        # __post_init__ normalized workspace and known tracer types.
        # tool_registry may still be a runtime object when tests pass
        # InMemoryToolRegistry directly and register tools after construction.
        tr = self.tool_registry
        if tr is not None and not isinstance(tr, ToolRegistryConfig):
            tr = _runtime_registry_to_config(tr)

        # Custom tracers (not HarnessJournal/NullTracer exact types) are kept
        # as-is by __post_init__; omit them from YAML rather than failing.
        tc = self.tracer if isinstance(self.tracer, TracerConfig) else None

        # workspace is always WorkspaceConfig after __post_init__ (or None).
        clean = _dc.replace(
            self,
            tool_registry=tr,
            tracer=tc,
            # processors is always list[dict] after __post_init__
            processors=self.processors,
            plugins=serialized_plugins,
        )
        return header + OmegaConf.to_yaml(OmegaConf.structured(clean))

    def to_yaml_file(self, path: "Any") -> None:
        """Write YAML to *path* (creates parent directories if needed)."""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "HarnessConfig":
        """Restore a HarnessConfig from a YAML string.

        Top-level ``model:`` and ``mcp_config:`` keys are silently stripped —
        they are not HarnessConfig fields.  ``mcp_config`` is the input for
        McpRuntimePlugin and is handled by the caller (Lab API run route)
        which mounts the plugin explicitly.
        """
        from omegaconf import OmegaConf

        # Keys stripped before OmegaConf structured-merge:
        #   model      — caller resolves model separately (ModelConfig)
        #   mcp_config — shorthand for McpRuntimePlugin; caller mounts the plugin
        #   <list field>: null — OmegaConf rejects null for non-Optional list fields;
        #                  the dataclass default (empty list) is the correct fallback.
        _STRIP = {"model", "mcp_config"}
        lines = [ln for ln in yaml_str.splitlines() if not ln.startswith("#")]
        override = OmegaConf.create("\n".join(lines))
        strip = _STRIP & set(override.keys())
        for key in list(override.keys()):
            if key not in strip and override[key] is None:
                strip = strip | {key}
        if strip:
            override = OmegaConf.masked_copy(override, [k for k in override if k not in strip])
        return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), override))

    @classmethod
    def from_yaml_file(cls, path: "Any") -> "HarnessConfig":
        """Load a HarnessConfig from a YAML file."""
        from pathlib import Path

        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    def copy(self, **kwargs: Any) -> "HarnessConfig":
        """Return a shallow copy with slot overrides applied."""
        new = copy.copy(self)
        new.processors = list(self.processors)
        new.plugins = list(self.plugins)
        # Copy non-dataclass attributes so the two objects are independent.
        new._rt_procs = list(getattr(self, "_rt_procs", []))
        new._rt_sandbox = getattr(self, "_rt_sandbox", None)
        for key, value in kwargs.items():
            setattr(new, key, value)
        # Re-run gate normalization so any runtime objects passed via kwargs
        # are converted to their config equivalents immediately.
        new.__post_init__()
        return new

    def canonicalize(self) -> "HarnessConfig":
        """Return a stable config copy for meta-harness / YAML round-trips.

        :func:`harnessx.meta_harness.evolve` calls this after
        ``HarnessConfig.from_yaml_file`` to validate a candidate config.
        Deduplicate identical processor dict entries (order-preserving, first wins).
        """
        new = self.copy()
        seen: set[str] = set()
        out: list = []
        for p in new.processors:
            key = repr(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        new.processors = out
        return new


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Harness:
    """Top-level Harness entry point. Composes ModelConfig + HarnessConfig and runs the RunLoop.

    Do not instantiate directly — use :meth:`ModelConfig.agentic`::

        agent = model_config.agentic(harness_config)
        result = await agent.run(task)
    """

    def __init__(
        self,
        model_config: "Any",  # ModelConfig
        config: HarnessConfig,
        extra_processors: dict[str, list[Processor]] | None = None,
    ):
        from .model_config import ModelConfig as _MC

        if not isinstance(model_config, _MC):
            raise TypeError(
                f"Harness expects a ModelConfig as the first argument, got {type(model_config).__name__}. "
                "Use: model_config.agentic(harness_config)"
            )
        self.model_config = model_config
        self.config = config
        self.child_harness_config: "HarnessConfig | None" = None
        self._closed = False
        self._sandbox = None
        _ACTIVE_HARNESSES.add(self)

        # Instantiate all runtime objects from config
        self._rt = _instantiate_runtime(config)

        if extra_processors:
            for key, procs in extra_processors.items():
                self._rt.processors.setdefault(key, []).extend(procs)

        # Resolve HarnessJournal base_dir to an absolute path so traces never
        # land in whatever CWD the caller happens to be in.
        from ..tracing.journal import HarnessJournal as _HJ

        tracer = self._rt.tracer
        if isinstance(tracer, _HJ) and tracer.base_dir == "sessions":
            if self._rt.workspace is not None:
                # Standard path: workspace was explicitly set (CLI / Lab UI).
                tracer.base_dir = str(self._rt.workspace.root / "sessions")
            else:
                # No explicit workspace: derive the default workspace root from
                # AGENT_HOME so traces go to
                #   agent_home()/workspaces/{agent_id}/{project}/sessions/
                # matching the layout used when workspace IS set.
                from ..home import agent_workspace_root

                tracer.base_dir = str(agent_workspace_root() / "sessions")

        # Build minimal sub-harnesses for each non-"main" key in ModelConfig
        # and bind them to all MultiHookProcessors via _bind_sub_harnesses().
        # Sub-harnesses use NullTracer so their internal model responses
        # (e.g. router classifier JSON) don't leak into the user-facing output.
        from ..tracing.null_tracer import NullTracer as _NullTracer

        sub_harnesses: dict[str, "Harness"] = {}
        for key, provider in self.model_config.models.items():
            if key == "main":
                continue
            sub_model = _MC(main=provider)
            sub_config = HarnessConfig(tracer=_NullTracer())
            sub_harnesses[key] = Harness(sub_model, sub_config)
        self._sub_harnesses = sub_harnesses

        from .processor import MultiHookProcessor

        for procs in self._rt.processors.values():
            for proc in procs:
                if isinstance(proc, MultiHookProcessor):
                    proc._bind_sub_harnesses(sub_harnesses)
                    proc._bind_tool_registry(self._rt.tool_registry)
                    proc._bind_model_config(self.model_config)
                    proc._bind_harness_config(self.config)
                    proc._bind_runtime(self._rt)

        # Two-phase plugin lifecycle: setup() runs after all processors are wired.
        for plugin in self._rt.plugins:
            try:
                plugin.setup(self.config)
                # Give plugin access to the runtime tool registry (InMemoryToolRegistry)
                # after setup() so it can register tools. config.tool_registry is
                # ToolRegistryConfig (serialisable form); _rt.tool_registry is the live one.
                if getattr(plugin, "_tool_registry", None) is not self._rt.tool_registry:
                    if hasattr(plugin, "_tool_registry"):
                        plugin._tool_registry = self._rt.tool_registry
            except Exception as exc:
                warnings.warn(
                    f"Plugin '{plugin.name}' setup() raised {type(exc).__name__}: {exc}. "
                    "The plugin's runtime initialisation was skipped.",
                    stacklevel=2,
                )

    async def run(
        self,
        task: BaseTask,
        *,
        session_id: "str | None" = None,
        parent_run_id: "str | None" = None,
        _resume_state: "State | None" = None,
        stream_callback: "object | None" = None,
        tracer_override: "Any | None" = None,
    ) -> HarnessResult:
        """
        Run the harness on a task.

        Args:
            task:           The task to run.
            session_id:     Session identifier for multi-turn continuity.  Auto-generated
                            (uuid4) if not provided.  HarnessJournal uses this as the
                            persistent key: writes a session index and enables
                            ``wake(session_id)`` recovery after a restart.
                            When a workspace + HarnessJournal are configured, a previous
                            session is automatically resumed from disk when possible.
            parent_run_id:  Parent run_id for sub-agent tracing; set internally.
            _resume_state:  Performance shortcut: pass the ``State`` returned by a
                            previous ``HarnessResult.resume_state`` to skip the
                            disk-based ``wake()`` call.  The state's ``run_id`` is
                            always preserved unchanged — run_id rotation happens
                            only via ``SegmentBoundaryEvent`` inside RunLoop.

        Returns HarnessResult(task_end, trajectory).
        """
        if self._closed:
            raise RuntimeError("Harness is closed. Create a new harness instance before running tasks.")

        import uuid as _uuid
        from .events import Message
        from ..sandbox.base import _sandbox_ctx
        from ..tracing.journal import HarnessJournal as _HJ

        # tracer_override (e.g. SSETracer) is active for this run only; it does
        # not mutate HarnessConfig or _rt so subsequent runs are unaffected.
        _active_tracer = tracer_override if tracer_override is not None else self._rt.tracer

        # Locate the HarnessJournal anywhere in the tracer chain (e.g. inside
        # an SSETracer wrapper).  All session-management operations target the
        # journal directly regardless of how many layers of wrapping exist.
        _journal = _find_journal(_active_tracer)

        # session_id is the primary key — always set, even for the first turn.
        # Priority: explicit arg > tracer.session_id already set > new uuid4.
        if _journal is not None:
            if session_id is not None:
                _journal.session_id = session_id
            elif _journal.session_id:
                session_id = _journal.session_id
            else:
                session_id = str(_uuid.uuid4())
                _journal.session_id = session_id
        else:
            if session_id is None:
                session_id = str(_uuid.uuid4())

        # Auto-resume: if no explicit _resume_state was given and HarnessJournal is
        # configured, try to wake() the previous state from disk.
        # Prefer workspace.root when available, but fall back to the journal's
        # own base_dir parent. Some integrations keep session files under a
        # channel-scoped subtree (e.g. <workspace>/<channel>/sessions) while the
        # workspace root remains at <workspace>; in that case the fallback path
        # is the only one that can resolve the index/state files.
        resume_state = _resume_state
        if resume_state is None and _journal is not None:
            from pathlib import Path as _Path

            _wake_roots: list[str] = []
            if self._rt.workspace is not None:
                _wake_roots.append(str(self._rt.workspace.root))
            _journal_root = str(_Path(_journal.base_dir).resolve().parent)
            if _journal_root not in _wake_roots:
                _wake_roots.append(_journal_root)

            for _wake_root in _wake_roots:
                try:
                    resume_state = _HJ.wake(session_id, _wake_root)
                    break
                except (FileNotFoundError, KeyError, ValueError):
                    continue

        if self._rt.workspace is not None and self.config.init_workspace and resume_state is None:
            from ..workspace.initializer import WorkspaceInitializer

            await WorkspaceInitializer().initialize(
                self._rt.workspace,
                template=self.config.workspace_template,
            )

        # Export fully-resolved harness config for reproducibility.
        # Written once per fresh run (not on resume) to allow exact reproduction
        # and to serve as the config reference for wake() recovery.
        _config_hash: str = ""
        if self._rt.workspace is not None and resume_state is None:
            _ws = self.config.workspace
            # Compute the agent-level root (not project-level) for harness_config.yaml
            if isinstance(_ws, WorkspaceConfig):
                if _ws.home is not None:
                    _cfg_root = _ws.home + "/workspaces/" + _ws.agent_id
                elif _ws.root is not None:
                    _cfg_root = _ws.root
                else:
                    _cfg_root = str(self._rt.workspace.root)
            else:
                # Runtime Workspace object — use home/workspaces/agent_id if available
                _ws_rt = self._rt.workspace
                if getattr(_ws_rt, "home", None) is not None:
                    _cfg_root = str(_ws_rt.home / "workspaces" / _ws_rt.agent_id)
                else:
                    _cfg_root = str(_ws_rt.root)
            _config_hash = _write_resolved_config(self.config, self.model_config, _cfg_root)

        # Inject config_hash into HarnessJournal (through any wrapper).
        if _journal is not None and _config_hash:
            _journal.config_hash = _config_hash

        # Sandbox hint: prefer explicit config override, then session_id for
        # stable sandbox affinity across turns, then a fresh uuid.
        _hint_id = self.config.sandbox_hint_id or session_id or str(_uuid.uuid4())
        if self._sandbox is None:
            self._sandbox = await self._rt.sandbox_provider.acquire(
                hint_id=_hint_id,
                workspace=self._rt.workspace,
            )
        _sandbox = self._sandbox
        _sandbox_token = _sandbox_ctx.set(_sandbox)

        try:
            if resume_state is not None:
                state = resume_state
                # Preserve restored effective context exactly as recovered.
                # Fallback only when callers pass a partial state manually.
                if not state.messages and state.raw_messages:
                    state.messages = list(state.raw_messages)
                # Keep state.run_id as-is — whether restored from disk or passed
                # in-memory.  A new run_id is only ever created by a
                # SegmentBoundaryEvent (compaction, system-prompt change, etc.)
                # emitted from within RunLoop.  Reassigning it here would break
                # journal continuity and produce orphaned JSONL segments.
                state.max_steps = state.step + task.max_steps
                if task.token_budget is not None:
                    state.token_budget = state.cumulative_tokens + task.token_budget
                if task.max_cost_usd is not None:
                    state.max_cost_usd = state.cumulative_cost_usd + task.max_cost_usd
                # Append the new user turn to the restored conversation.
                # Without this, the model sees the prior history but not the
                # current user input — causing completely unrelated responses.
                user_content = task.description or ""
                if user_content:
                    state.add_raw_message(Message(role="user", content=user_content))
            else:
                # Truly fresh session — generate the initial run_id.
                # Subsequent turns inherit this via resume_state.run_id.
                state = State(
                    run_id=make_run_id(),
                    max_steps=task.max_steps,
                    token_budget=task.token_budget,
                    max_cost_usd=task.max_cost_usd,
                    spawn_depth=task.spawn_depth,
                )
                user_content = task.description or ""
                if user_content:
                    state.add_raw_message(Message(role="user", content=user_content))

            # Transfer any pending plugin side-channels so processors / run_loop
            # can read them.  Command prompts and allowed_tools are stored as
            # plain attributes on the state object (not in state.slots, which
            # requires StateSlot objects).  Slash slots use set_slot().
            pending_cmd_prompt = getattr(self, "_pending_command_prompt", None)
            if pending_cmd_prompt is not None:
                state._pending_command_prompt = pending_cmd_prompt  # type: ignore[attr-defined]
                try:
                    del self._pending_command_prompt  # type: ignore[attr-defined]
                except AttributeError:
                    pass

            pending_allowed_tools = getattr(self, "_pending_command_allowed_tools", None)
            if pending_allowed_tools is not None:
                state._pending_command_allowed_tools = pending_allowed_tools  # type: ignore[attr-defined]
                try:
                    del self._pending_command_allowed_tools  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            else:
                # Clear any stale value from a previous command that had restrictions
                try:
                    del state._pending_command_allowed_tools  # type: ignore[attr-defined]
                except AttributeError:
                    pass

            pending_slots = getattr(self, "_pending_slash_slots", {})
            if pending_slots:
                for slot_key, slot_val in pending_slots.items():
                    state.set_slot(slot_key, "slash_command", slot_val)
                self._pending_slash_slots = {}  # type: ignore[attr-defined]

            route_slot_keys: list[str] = []
            for procs in self._rt.processors.values():
                for proc in procs:
                    if getattr(type(proc), "_singleton_group", None) != "model_router":
                        continue
                    key = getattr(proc, "slot_key", "model.route")
                    if isinstance(key, str) and key and key not in route_slot_keys:
                        route_slot_keys.append(key)
            if not route_slot_keys:
                route_slot_keys = ["model.route"]

            def _select_model_provider(current_state: State):
                for route_slot_key in route_slot_keys:
                    slot = current_state.get_slot(route_slot_key)
                    if slot is None or not isinstance(slot.content, dict):
                        continue
                    selected_key = slot.content.get("selected_key")
                    if not isinstance(selected_key, str) or not selected_key:
                        continue
                    try:
                        return self.model_config.get(selected_key)
                    except Exception:
                        continue
                return self.model_config.main

            end_event, trajectory, interrupted_at = await run_loop(
                task=task,
                state=state,
                model_provider=self.model_config.main,
                model_selector=_select_model_provider,
                tool_registry=self._rt.tool_registry,
                tracer=_active_tracer,
                processors=self._rt.processors,
                workspace=self._rt.workspace,
                parent_run_id=parent_run_id,
                step_snapshots=self.config.step_snapshots,
                stream_callback=stream_callback,
                model_config=self.model_config,
                harness_config=self.config,
                child_harness_config=self.child_harness_config,
            )

            # Auto-backfill terminal reward into all trajectory steps.
            # EvaluationProcessor writes eval_result onto task_end; backfill_rewards
            # propagates it to every TrajectoryStep so callers don't need to do it.
            if end_event.eval_result is not None:
                trajectory.backfill_rewards(end_event.eval_result)

            # Token annotation: RL providers populate TrajectoryStep.token_annotation
            # from captured token data; standard providers use the Protocol no-op.
            self.model_config.main.annotate_trajectory(trajectory)

            # ModelRouterProcessor state is task-scoped. Ensure route slots don't
            # leak into the next turn even if task_end cleanup was skipped.
            for route_slot_key in route_slot_keys:
                slot = state.get_slot(route_slot_key)
                if slot is not None and slot.slot_type == "model_route":
                    state.delete_slot(route_slot_key)

            return HarnessResult(
                task_end=end_event,
                trajectory=trajectory,
                interrupted_at=interrupted_at,
                resume_state=state,
            )
        finally:
            _sandbox_ctx.reset(_sandbox_token)

    async def cleanup(self) -> None:
        """Release harness-scoped resources (plugins, sandbox, sub-harnesses)."""
        if self._closed:
            return
        self._closed = True

        # Sub-harnesses are internal helpers for non-main model roles.
        for sub in reversed(list(getattr(self, "_sub_harnesses", {}).values())):
            try:
                await sub.cleanup()
            except Exception:
                pass

        for plugin in reversed(self._rt.plugins):
            try:
                result = plugin.stop()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                warnings.warn(
                    f"Plugin '{plugin.name}' stop() raised {type(exc).__name__}: {exc}.",
                    stacklevel=2,
                )

        sandbox = self._sandbox
        self._sandbox = None
        if sandbox is not None:
            try:
                await self._rt.sandbox_provider.release(sandbox)
            except Exception as exc:
                warnings.warn(
                    f"Sandbox release raised {type(exc).__name__}: {exc}.",
                    stacklevel=2,
                )
        _ACTIVE_HARNESSES.discard(self)
