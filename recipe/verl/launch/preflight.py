"""Fail early on missing dependencies and unsupported training interfaces."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import os

from packaging.specifiers import SpecifierSet

from recipe.verl.bootstrap import DEFAULT_DESTINATION, check_installation


def check_versions(version_lookup=None):
    if version_lookup is None:
        version_lookup = importlib.metadata.version
    requirements = {"sglang": "==0.5.6", "torch": "==2.9.1", "transformers": ">=4.51,<5"}
    for name, requirement in requirements.items():
        try:
            installed = version_lookup(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Missing training dependency: {name}. See recipe/verl/README.md") from error
        if installed not in SpecifierSet(requirement):
            raise RuntimeError(f"Expected {name}{requirement}, found {installed}. See recipe/verl/README.md")


def main():
    check_installation()
    import verl

    if not Path(verl.__file__).resolve().is_relative_to(DEFAULT_DESTINATION):
        raise RuntimeError("Imported veRL is not the pinned HarnessX dependency; check PYTHONPATH")
    check_versions()
    from recipe.coevolution.behavior import behavior_pool, resolve_behavior

    for behavior in behavior_pool():
        resolve_behavior(explicit=behavior)
    if os.environ.get("HARNESSX_HARNESS_CONFIG"):
        resolve_behavior(explicit=os.environ["HARNESSX_HARNESS_CONFIG"])
    from sglang.srt.entrypoints.http_server import _launch_subprocesses
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    from recipe.verl.agent_loop import HarnessXAgentLoop

    if not all(callable(interface) for interface in (_launch_subprocesses, SGLangHttpServer, HarnessXAgentLoop)):
        raise RuntimeError("Training backend interfaces are unavailable")
    for key in ("DATA_DIR", "MODEL_DIR"):
        if not Path(os.environ[key]).is_dir():
            raise RuntimeError(f"{key} must point to an existing local directory")


if __name__ == "__main__":
    main()
