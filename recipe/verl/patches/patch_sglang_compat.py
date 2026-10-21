# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Patch SGLang 0.5.11 to expose _launch_subprocesses at module level.

verl 0.7.1 imports:
    from sglang.srt.entrypoints.http_server import _launch_subprocesses

But SGLang 0.5.11 moved it to Engine._launch_subprocesses (class method).
This patch adds a module-level shim that delegates to Engine._launch_subprocesses.
"""
_patched = False


def apply():
    global _patched
    if _patched:
        return
    _patched = True

    try:
        from sglang.srt.entrypoints import http_server
        if hasattr(http_server, "_launch_subprocesses"):
            return  # already available, no patch needed

        from sglang.srt.entrypoints.engine import Engine
        if hasattr(Engine, "_launch_subprocesses"):
            http_server._launch_subprocesses = Engine._launch_subprocesses
    except ImportError:
        pass
