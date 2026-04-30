"""OpenSandbox Harbor environment adapter.

Connects Harbor to a self-hosted OpenSandbox server (https://github.com/alibaba/OpenSandbox).
The server manages sandbox container lifecycle; the execd daemon inside each sandbox
handles command execution and file operations via port 44772.

Usage::

    from benchmarks.terminal_bench_2.opensandbox import OpenSandboxEnvironment

    env = OpenSandboxEnvironment(
        environment_dir=...,
        environment_name=...,
        session_id=...,
        trial_paths=...,
        task_env_config=...,
        server_url="http://10.x.x.x:12081",
    )
"""

from .environment import OpenSandboxEnvironment

__all__ = ["OpenSandboxEnvironment"]
