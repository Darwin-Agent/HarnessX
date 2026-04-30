# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""
Code interpreter tool — execute Python code in a safe sandbox environment.

CodeInterpreter tool for verl_harnessX tool registry.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from typing import Any

from .base import tool

logger = logging.getLogger(__name__)

_success_count: int = 0
_fail_count: int = 0

# ---------------------------------------------------------------------------
# Safety patterns — always blocked regardless of allowed_modules.
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"import\s+os", re.I),
    re.compile(r"import\s+sys", re.I),
    re.compile(r"import\s+subprocess", re.I),
    re.compile(r"import\s+shutil", re.I),
    re.compile(r"import\s+glob", re.I),
    re.compile(r"import\s+pathlib", re.I),
    re.compile(r"__import__", re.I),
    re.compile(r"\beval\s*\(", re.I),
    re.compile(r"\bexec\s*\(", re.I),
    re.compile(r"(?<!\w)open\s*\(", re.I),
    re.compile(r"\bfile\s*\(", re.I),
    re.compile(r"\binput\s*\(", re.I),
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
# Configuration (override via env vars)
# ---------------------------------------------------------------------------

_ALLOWED_MODULES: frozenset[str] = frozenset(
    os.environ.get(
        "VERL_CODE_ALLOWED_MODULES",
        "math,json,re,collections,itertools,functools,string,datetime,decimal,fractions,statistics,random,hashlib,base64,copy,typing,dataclasses,enum,numbers,operator,textwrap,unicodedata,pprint,bisect,heapq,array,struct",
    ).split(",")
)
_TIMEOUT = int(os.environ.get("VERL_CODE_TIMEOUT", "60"))
_MAX_OUTPUT_CHARS = int(os.environ.get("VERL_CODE_MAX_OUTPUT", "2000"))
_CONCURRENCY = int(os.environ.get("VERL_CODE_CONCURRENCY", "32"))
_MAX_MEMORY_MB = float(os.environ.get("VERL_CODE_MAX_MEMORY_MB", "12288"))
_semaphore = asyncio.Semaphore(_CONCURRENCY)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_sandbox():
    try:
        from harnessx.sandbox.base import get_current_sandbox

        return get_current_sandbox()
    except ImportError:
        return None


def _check_safety(
    code: str,
    allowed_modules: frozenset[str],
) -> tuple[bool, str]:
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
    indented = "\n".join("    " + line for line in code.splitlines())
    return _WRAPPER.format(indented_code=indented)


async def _execute_via_sandbox(
    sandbox: Any,
    wrapped_code: str,
    timeout: float,
) -> str:
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
# Tool definition
# ---------------------------------------------------------------------------

_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "The Python code to execute.",
        }
    },
    "required": ["code"],
}


@tool(
    name="CodeInterpreter",
    description="Execute Python code in a safe sandbox environment. Supports math, json, re, collections, itertools, and other standard library modules.",
    input_schema=_SCHEMA,
)
async def code_tool(code: str) -> str:
    global _success_count, _fail_count
    t0 = time.monotonic()

    safe, reason = _check_safety(code, _ALLOWED_MODULES)
    if not safe:
        _fail_count += 1
        logger.warning(
            "CodeInterpreter BLOCKED: %s [total: %d ok, %d fail]",
            reason[:80],
            _success_count,
            _fail_count,
        )
        return f"Error: {reason}"

    wrapped = _wrap_code(code)

    async with _semaphore:
        sandbox = _get_sandbox()

        if sandbox is not None:
            result = await _execute_via_sandbox(sandbox, wrapped, _TIMEOUT)
        else:
            result = await asyncio.to_thread(
                _execute_subprocess,
                wrapped,
                _TIMEOUT,
                _MAX_MEMORY_MB,
            )

    elapsed = time.monotonic() - t0

    if len(result) > _MAX_OUTPUT_CHARS:
        result = result[:_MAX_OUTPUT_CHARS] + _TRUNCATION_NOTICE

    if result.startswith("Error:"):
        _fail_count += 1
        logger.warning(
            "CodeInterpreter FAILED (%.1fs): %s [total: %d ok, %d fail]",
            elapsed,
            result[:120],
            _success_count,
            _fail_count,
        )
    else:
        _success_count += 1
        if elapsed > 10:
            logger.warning(
                "CodeInterpreter OK (%.1fs, slow) [total: %d ok, %d fail]",
                elapsed,
                _success_count,
                _fail_count,
            )
        else:
            logger.warning(
                "CodeInterpreter OK (%.1fs) [total: %d ok, %d fail]",
                elapsed,
                _success_count,
                _fail_count,
            )

    return result
