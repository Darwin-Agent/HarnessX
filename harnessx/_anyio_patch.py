# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Defensive monkey patch for anyio's asyncio CancelScope.__exit__.

Background
----------
When ``anyio._backends._asyncio.CancelScope.__exit__`` is invoked from a task
other than the one that entered the scope, anyio raises::

    RuntimeError("Attempted to exit cancel scope in a different task than it was
                  entered in")

This RuntimeError is raised BEFORE the cleanup block (line 444 vs the
``try/finally`` starting at line 468 in anyio 4.13). The scope is left with
``_active=True``, ``_tasks`` still containing the host task and ``_cancel_handle``
potentially pointing at a self-rescheduling ``call_soon`` chain.

If something later cancels that orphaned scope (e.g. a parent ``wait_for``
deadline fires), ``_deliver_cancellation`` runs and finds ``_tasks`` non-empty —
even though the host task is already done — so ``should_retry`` becomes ``True``
and ``call_soon`` re-arms forever. The result is a 100% CPU hot loop in the
event loop that no debugger output can interrupt.

This patch wraps ``__exit__`` so that, if the cross-task ``RuntimeError`` is
raised, we forcibly drop the scope from anyio's bookkeeping (``_tasks``,
``_child_scopes``, ``_cancel_handle``, ``_active``) before re-raising. The
RuntimeError still propagates, so callers that swallow it learn about the bug
through the warning emitted here, but no orphan state is left to be re-armed.

This is defense-in-depth: HarnessX routes MCP connect/disconnect through a
single supervisor task so the cross-task path should not be hit during normal
operation. The patch is a safety net for third-party code paths we do not own.
"""

from __future__ import annotations

import warnings


_PATCHED_FLAG = "__harnessx_cross_task_patched__"


def _force_cleanup(scope) -> None:
    """Best-effort cleanup of a CancelScope after a cross-task __exit__ error.

    The original ``__exit__`` raises before any cleanup, so we replicate just
    enough of the bookkeeping that ``_deliver_cancellation`` can no longer
    self-reschedule on this scope.
    """
    try:
        handle = getattr(scope, "_cancel_handle", None)
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
            scope._cancel_handle = None
    except Exception:
        pass

    try:
        host = getattr(scope, "_host_task", None)
        tasks = getattr(scope, "_tasks", None)
        if host is not None and tasks is not None:
            try:
                tasks.discard(host)
            except Exception:
                pass
        if tasks is not None:
            try:
                tasks.clear()
            except Exception:
                pass
    except Exception:
        pass

    try:
        from anyio._backends._asyncio import _task_states  # type: ignore[attr-defined]

        if host is not None:
            ts = _task_states.get(host)
            if ts is not None and getattr(ts, "cancel_scope", None) is scope:
                ts.cancel_scope = getattr(scope, "_parent_scope", None)
    except Exception:
        pass

    try:
        parent = getattr(scope, "_parent_scope", None)
        if parent is not None:
            children = getattr(parent, "_child_scopes", None)
            if children is not None:
                try:
                    children.discard(scope)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        scope._active = False
        scope._cancel_called = False
        scope._pending_uncancellations = None
    except Exception:
        pass


def apply() -> bool:
    """Install the patch on anyio's asyncio CancelScope. Idempotent.

    Returns ``True`` if the patch was applied (or already in place), ``False``
    if anyio is unavailable or its internals don't match what we expect.
    """
    try:
        from anyio._backends import _asyncio as _anyio_asyncio
    except Exception:
        return False

    cs_cls = getattr(_anyio_asyncio, "CancelScope", None)
    if cs_cls is None:
        return False

    if getattr(cs_cls.__exit__, _PATCHED_FLAG, False):
        return True

    original_exit = cs_cls.__exit__

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-redef]
        try:
            return original_exit(self, exc_type, exc_val, exc_tb)
        except RuntimeError as exc:
            if "different task" in str(exc):
                _force_cleanup(self)
                warnings.warn(
                    "anyio CancelScope.__exit__ called from a task different "
                    "from the entering task; HarnessX forcibly cleaned up the "
                    "orphaned scope state. This indicates a cross-task lifecycle "
                    f"bug in caller code: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raise

    setattr(__exit__, _PATCHED_FLAG, True)
    cs_cls.__exit__ = __exit__  # type: ignore[method-assign]
    return True


_applied: bool = apply()
