"""harnessx.sandbox — pluggable execution environment for agent tools.

Default (always available):
    LocalSandboxProvider  — subprocess + open() on the host filesystem

Optional backends (install the matching extra first):
    DockerSandboxProvider — pip install harnessx
    E2BSandboxProvider    — pip install harnessx
"""

from .base import Mount, Sandbox, SandboxProvider, _sandbox_ctx, get_current_sandbox
from .local import LocalSandbox, LocalSandboxProvider

__all__ = [
    "Mount",
    "Sandbox",
    "SandboxProvider",
    "get_current_sandbox",
    "_sandbox_ctx",
    "LocalSandbox",
    "LocalSandboxProvider",
    # Optional — imported lazily to avoid hard dependency
    "DockerSandboxProvider",
    "E2BSandboxProvider",
]


def __getattr__(name: str):
    if name == "DockerSandboxProvider":
        from .docker import DockerSandboxProvider

        return DockerSandboxProvider
    if name == "E2BSandboxProvider":
        from .e2b import E2BSandboxProvider

        return E2BSandboxProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
