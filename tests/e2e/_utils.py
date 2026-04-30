# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_E2E_DIR = Path(__file__).parent


# ── .env file reader ──────────────────────────────────────────────────────────


def _read_dot_env() -> dict[str, str]:
    """Parse tests/e2e/.env → dict.  Strips surrounding quotes from values."""
    cfg: dict[str, str] = {}
    path = _E2E_DIR / ".env"
    if not path.exists():
        return cfg
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def _get(cfg: dict, *keys: str) -> str | None:
    """Return first non-empty value found across os.environ then .env dict."""
    for k in keys:
        v = os.environ.get(k) or cfg.get(k)
        if v:
            return v
    return None


# ── EXTRA_HEADERS parser ──────────────────────────────────────────────────────


def _parse_extra_headers(raw: str) -> dict[str, str]:
    """Parse 'Name: Value, Name2: Value2' or JSON → dict."""
    raw = raw.strip()
    if raw.startswith("{"):
        import json

        return json.loads(raw)
    headers: dict[str, str] = {}
    for part in raw.split(","):
        if ": " in part:
            name, _, value = part.strip().partition(": ")
            headers[name.strip()] = value.strip()
    return headers


# ── Test home helper ──────────────────────────────────────────────────────────


def get_test_home() -> Path:
    """Return the agent home for e2e tests.

    Reads HXE2E_TEST_HOME from tests/e2e/.env (or os.environ); falls back to
    PROJECT_ROOT / ".test_e2e" if the variable is not set.
    """
    cfg = _read_dot_env()
    raw = _get(cfg, "HXE2E_TEST_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return PROJECT_ROOT / ".test_e2e"


# ── Workspace helper ──────────────────────────────────────────────────────────


def make_test_workspace(test_name: str, mode: str = "shared"):
    """Return a Workspace auto-derived from HXE2E_TEST_HOME for the given test.

    Path layout: <HXE2E_TEST_HOME>/workspaces/<test_name>/default/
    """
    from harnessx.workspace.workspace import Workspace

    return Workspace(
        agent_id=test_name,
        home=get_test_home(),
        mode=mode,
    )


# ── Provider loader ───────────────────────────────────────────────────────────


def load_provider():
    """Build a model provider from env vars and tests/e2e/.env.

    Mirrors the CLI provider selection logic exactly (see module docstring).
    """
    cfg = _read_dot_env()

    anthropic_key = _get(cfg, "ANTHROPIC_API_KEY")
    anthropic_model = _get(cfg, "ANTHROPIC_DEFAULT_MAIN_MODEL")
    openai_key = _get(cfg, "OPENAI_API_KEY")
    openai_model = _get(cfg, "OPENAI_DEFAULT_MAIN_MODEL")
    litellm_key = _get(cfg, "LITELLM_API_KEY")
    litellm_model = _get(cfg, "LITELLM_DEFAULT_MAIN_MODEL")

    raw_headers = _get(cfg, "EXTRA_HEADERS")
    env_headers = _parse_extra_headers(raw_headers) if raw_headers else {}
    timeout_raw = _get(cfg, "HARNESSX_REQUEST_TIMEOUT")

    if anthropic_key or anthropic_model:
        from harnessx.providers.anthropic_provider import AnthropicProvider

        model = anthropic_model or "claude-sonnet-4-6"
        kw: dict = {}
        if anthropic_key:
            kw["api_key"] = anthropic_key
        base_url = _get(cfg, "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL")
        if base_url:
            kw["base_url"] = base_url
        if env_headers:
            kw["default_headers"] = env_headers
        if timeout_raw:
            kw["timeout"] = float(timeout_raw)
        return AnthropicProvider(model, **kw)

    if openai_key or openai_model:
        from harnessx.providers.litellm_provider import LiteLLMProvider

        model = openai_model or "gpt-4o"
        kw = {}
        if openai_key:
            kw["api_key"] = openai_key
        api_base = _get(cfg, "OPENAI_API_BASE")
        if api_base:
            kw["api_base"] = api_base
        if env_headers:
            kw["extra_headers"] = env_headers
        if timeout_raw:
            kw["request_timeout"] = int(timeout_raw)
        return LiteLLMProvider(model, **kw)

    if litellm_key or litellm_model:
        from harnessx.providers.litellm_provider import LiteLLMProvider

        model = litellm_model or "claude-sonnet-4-6"
        kw = {}
        if litellm_key:
            kw["api_key"] = litellm_key
        api_base = _get(cfg, "LITELLM_API_BASE")
        if api_base:
            kw["api_base"] = api_base
        if env_headers:
            kw["extra_headers"] = env_headers
        if timeout_raw:
            kw["request_timeout"] = int(timeout_raw)
        return LiteLLMProvider(model, **kw)

    # Fallback — no provider env vars set
    from harnessx.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider("claude-sonnet-4-6")
