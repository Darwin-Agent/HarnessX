# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ._utils import _read_dot_env, _parse_extra_headers, get_test_home  # noqa: F401
except ImportError:
    from _utils import (
        _read_dot_env,
        get_test_home,
    )  # when run as script


# ── CLI env builder ───────────────────────────────────────────────────────────


def _cli_env() -> dict:
    """Build subprocess env with model/API settings from .env / os.environ."""
    cfg = _read_dot_env()
    env = os.environ.copy()

    for key in (
        # Anthropic provider env vars
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_DEFAULT_MAIN_MODEL",
        "ANTHROPIC_API_BASE",
        "ANTHROPIC_BASE_URL",
        # OpenAI provider env vars
        "OPENAI_API_KEY",
        "OPENAI_DEFAULT_MAIN_MODEL",
        "OPENAI_API_BASE",
        # LiteLLM provider env vars
        "LITELLM_API_KEY",
        "LITELLM_DEFAULT_MAIN_MODEL",
        "LITELLM_API_BASE",
        # Common
        "EXTRA_HEADERS",
        "HARNESSX_MODEL",
    ):
        if key not in env and key in cfg:
            env[key] = cfg[key]

    # 30s per-request timeout for e2e tests — fail fast on stalled API calls
    env.setdefault("HARNESSX_REQUEST_TIMEOUT", "30")

    # Redirect agent_home to HXE2E_TEST_HOME (not the real ~/.harnessx/)
    env.setdefault("HARNESSX_HOME", str(get_test_home()))

    return env


def _cli_cmd(extra_args: list[str] | None = None) -> list[str]:
    """Base CLI command.

    Model and API settings are forwarded via environment variables in _cli_env(),
    not as CLI flags — the CLI has no -m/--model or -H flag; it reads
    ANTHROPIC_DEFAULT_MAIN_MODEL / OPENAI_DEFAULT_MAIN_MODEL / etc. from the env.
    """
    cmd = [sys.executable, "-m", "harnessx.cli"]
    if extra_args:
        cmd += extra_args
    return cmd


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_cli(args: list[str], stdin: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    env = _cli_env()
    return subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(PROJECT_ROOT),
    )


def _assert_time_in_output(stdout: str) -> str | None:
    """Return None if output looks like it contains a time/date, else an error string."""
    import re

    # Match common time patterns: HH:MM, AM/PM, digits that look like a date/time
    patterns = [
        r"\d{1,2}:\d{2}",  # HH:MM
        r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
        r"\d{10,}",  # epoch seconds
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"(january|february|march|april|june|july|august|september|october|november|december)",
    ]
    combined = stdout.lower()
    for pat in patterns:
        if re.search(pat, combined, re.IGNORECASE):
            return None
    return f"No recognisable date/time pattern found in output:\n{stdout[:500]}"


# ── Test: one-shot run mode ───────────────────────────────────────────────────


def test_cli_run_system_time():
    """
    harnessx -p "<task>"  — non-interactive mode, model must call Bash `date`.

    Expects:
    - exit code 0
    - stdout contains a recognisable date/time string
    - Bash tool was called (enforced by explicit prompt constraint)
    """
    task = (
        "Execute the Bash tool RIGHT NOW with the command `date`. "
        "Do not answer from memory or the system prompt — you MUST call the Bash tool first. "
        "After the tool returns, include its exact output in your reply."
    )
    cmd = _cli_cmd(["-p", task])

    proc = _run_cli(cmd)

    assert proc.returncode == 0, (
        f"CLI exited with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert proc.stdout.strip(), f"CLI produced no stdout\nSTDERR:\n{proc.stderr}"

    err = _assert_time_in_output(proc.stdout)
    assert err is None, err


# ── Test: chat mode ───────────────────────────────────────────────────────────


def test_cli_chat_system_time():
    """
    harnessx  (interactive mode, piping a single message via stdin)

    Expects:
    - exit code 0
    - stdout contains a recognisable date/time string
    - EOF on stdin terminates the chat loop gracefully
    """
    task = (
        "Execute the Bash tool RIGHT NOW with the command `date`. "
        "Do not answer from memory or the system prompt — you MUST call the Bash tool first. "
        "After the tool returns, include its exact output in your reply."
    )
    # Chat loop: send task, EOF terminates the loop
    stdin_input = f"{task}\n"

    cmd = _cli_cmd()
    proc = _run_cli(cmd, stdin=stdin_input)

    assert proc.returncode == 0, (
        f"CLI chat exited with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert proc.stdout.strip(), f"CLI chat produced no stdout\nSTDERR:\n{proc.stderr}"

    err = _assert_time_in_output(proc.stdout)
    assert err is None, err


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0

    tests = [
        ("CLI run — system time (one-shot)", test_cli_run_system_time),
        ("CLI chat — system time (chat mode)", test_cli_chat_system_time),
    ]

    print("HarnessX CLI E2E Tests")
    print("=" * 60)

    for name, fn in tests:
        print(f"\nRunning: {name} ...")
        try:
            fn()
            print("  PASS")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback

            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
