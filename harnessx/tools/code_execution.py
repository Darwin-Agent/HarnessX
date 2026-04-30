# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import gc
import os
import re
import subprocess
import tempfile
import uuid
from typing import Any

from harnessx.tools.base import Tool

# ---------------------------------------------------------------------------
# Safety patterns — always blocked regardless of allowed_modules.
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: list[re.Pattern] = [
    # ── File system / process access ──────────────────────────────────────
    re.compile(r"import\s+os", re.I),
    re.compile(r"import\s+sys", re.I),
    re.compile(r"import\s+subprocess", re.I),
    re.compile(r"import\s+shutil", re.I),
    re.compile(r"import\s+glob", re.I),
    re.compile(r"import\s+pathlib", re.I),
    # ── Dynamic code execution ────────────────────────────────────────────
    re.compile(r"__import__", re.I),
    re.compile(r"\beval\s*\(", re.I),
    re.compile(r"\bexec\s*\(", re.I),
    # ── File I/O & interactive input ──────────────────────────────────────
    re.compile(r"(?<!\w)open\s*\(", re.I),
    re.compile(r"\bfile\s*\(", re.I),
    re.compile(r"\binput\s*\(", re.I),
    # ── Sandbox escape vectors ────────────────────────────────────────────
    re.compile(r"__subclasses__"),
    re.compile(r"__builtins__"),
    re.compile(r"__globals__"),
]

# ---------------------------------------------------------------------------
# Code wrapper template — RLIMIT_AS + stdout/stderr capture
# ---------------------------------------------------------------------------

_WRAPPER = """\
import sys, traceback, resource
from io import StringIO

try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
except Exception:
    pass

old_stdout, old_stderr = sys.stdout, sys.stderr
stdout_cap = StringIO()
stderr_cap = StringIO()
sys.stdout = stdout_cap
sys.stderr = stderr_cap

try:
{indented_code}

    out = stdout_cap.getvalue()
    err = stderr_cap.getvalue()
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    result = ""
    if out:
        result += f"Output:\\n{{out}}"
    if err:
        result += f"\\nErrors:\\n{{err}}"
    print(result)

except Exception as e:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    print(f"Error: {{e}}\\nTraceback:\\n{{traceback.format_exc()}}")
"""

_TRUNCATION_NOTICE = "\n...[output truncated — printed too much, use fewer print() calls]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_safety(
    code: str,
    allowed_modules: frozenset[str],
) -> tuple[bool, str]:
    """Static safety check: regex blocklist + import allowlist."""
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(code):
            return False, f"Dangerous pattern detected: {pattern.pattern}"
    for imp in re.findall(r"^\s*import\s+(\w+)", code, re.MULTILINE) + re.findall(
        r"^\s*from\s+(\w+)", code, re.MULTILINE
    ):
        if imp not in allowed_modules:
            return False, f"Import of '{imp}' is not allowed"
    return True, "ok"


def _wrap_code(code: str) -> str:
    """Wrap user code with RLIMIT_AS + stdout/stderr capture."""
    indented = "\n".join("    " + line for line in code.splitlines())
    return _WRAPPER.format(indented_code=indented)


async def _execute_via_sandbox(
    sandbox: Any,
    wrapped_code: str,
    timeout: float,
) -> str:
    """Execute code through the active Sandbox (Local, Docker, etc.)."""
    script_path = f"/tmp/oh_code_{uuid.uuid4().hex[:8]}.py"
    try:
        await sandbox.write_file(script_path, wrapped_code)
        result = await sandbox.exec(
            f"python3 {script_path}",
            timeout=timeout,
        )
    except Exception as e:
        result = f"Error: {e}"
    finally:
        try:
            await sandbox.exec(f"rm -f {script_path}", timeout=5)
        except Exception:
            pass
    return result.strip()


def _execute_subprocess(
    wrapped_code: str,
    timeout: int,
    max_memory_mb: float,
) -> str:
    """Fallback: direct subprocess execution (no sandbox)."""
    try:
        import psutil

        if psutil.Process().memory_info().rss / 1024 / 1024 > max_memory_mb:
            for _ in range(3):
                gc.collect()
            return "Error: Memory usage too high, please try again"
    except ImportError:
        pass

    with tempfile.TemporaryDirectory(prefix="oh_code_") as tmpdir:
        script = os.path.join(tmpdir, "code.py")
        with open(script, "w") as f:
            f.write(wrapped_code)
        env = {**os.environ, "PYTHONPATH": tmpdir, "PYTHONUNBUFFERED": "1"}
        try:
            proc = subprocess.Popen(
                ["python3", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=tmpdir,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                result = (
                    stdout.strip()
                    if proc.returncode == 0
                    else (f"Error: Process exited with code {proc.returncode}\n{stderr}")
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                result = f"Error: Code execution timed out after {timeout} seconds"
        except Exception as exc:
            result = f"Error: Failed to execute code: {exc}"

    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_code_execution_tool(
    name: str = "code_interpreter",
    description: str = "Execute Python code in a safe sandbox environment.",
    allowed_modules: set[str] | None = None,
    timeout: int = 30,
    max_output_chars: int = 2000,
    concurrency: int = 32,
    max_memory_mb: float = 12288,
) -> Tool:
    """
    Build a sandbox-aware Python code execution Tool.

    When a Sandbox is active (inside Harness.run()), code is executed via
    ``sandbox.exec()`` — automatically routed to the configured sandbox
    provider (local subprocess, Docker, Harbor, etc.).

    Without an active sandbox, falls back to direct subprocess execution.

    Safety checks (regex blocklist + import allowlist) always run locally
    before code is sent to any execution environment.

    Args:
        name:             Tool name exposed to the model.
        description:      Tool description shown in the tool schema.
        allowed_modules:  Python module names that may be imported.
                          Any import not in this set is rejected.
                          None = deny all imports (builtins only).
        timeout:          Execution timeout in seconds.
        max_output_chars: Truncate output beyond this limit.
        concurrency:      Max simultaneous executions.
        max_memory_mb:    RSS threshold for fallback subprocess path.

    Returns:
        Tool instance ready for InMemoryToolRegistry.register().
    """
    _allowed = frozenset(allowed_modules or set())
    _semaphore = asyncio.Semaphore(concurrency)

    async def _execute(code: str) -> str:
        # 1. Safety check (always local, before any execution)
        safe, reason = _check_safety(code, _allowed)
        if not safe:
            return f"Error: {reason}"

        # 2. Wrap code
        wrapped = _wrap_code(code)

        # 3. Execute with concurrency control
        async with _semaphore:
            from harnessx.sandbox.base import get_current_sandbox

            sandbox = get_current_sandbox()

            if sandbox is not None:
                result = await _execute_via_sandbox(sandbox, wrapped, timeout)
            else:
                result = await asyncio.to_thread(
                    _execute_subprocess,
                    wrapped,
                    timeout,
                    max_memory_mb,
                )

        # 4. Truncate
        if len(result) > max_output_chars:
            result = result[:max_output_chars] + _TRUNCATION_NOTICE
        return result

    _schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            }
        },
        "required": ["code"],
    }

    return Tool(
        name=name,
        description=description,
        input_schema=_schema,
        fn=_execute,
    )
