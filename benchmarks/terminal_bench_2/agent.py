# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import re
import traceback
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .defaults import API_KEY_DEFAULT, MAX_STEPS, OUTPUT_LIMIT, REQUEST_TIMEOUT_SEC, TOKEN_BUDGET, WORKSPACE_PATH

_KNOWN_PREFIXES = ("openai/", "anthropic/", "azure/", "groq/", "ollama/", "mistral/")


def _parse_extra_headers(raw: str | dict | None) -> dict | None:
    """Accept either a dict or a 'Name: Value, Name2: Value2' string."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    headers: dict = {}
    for part in raw.split(","):
        part = part.strip()
        if ": " in part:
            name, _, value = part.partition(": ")
            headers[name.strip()] = value.strip()
    return headers or None


class HarnessXAgent(BaseAgent):
    """Terminal Bench 2.0 agent that runs the HarnessX RunLoop."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str = API_KEY_DEFAULT,
        extra_headers: str | dict | None = None,
        max_steps: int = MAX_STEPS,
        request_timeout_sec: int = REQUEST_TIMEOUT_SEC,
        max_tokens: int | None = None,
        harness_config_yaml: str | None = None,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._api_base = api_base
        self._api_key = api_key
        self._extra_headers = _parse_extra_headers(extra_headers)
        self._max_steps = int(max_steps)
        self._request_timeout_sec = int(request_timeout_sec)
        self._max_tokens = int(max_tokens) if max_tokens is not None else None
        # Optional path to an evolved HarnessConfig YAML.  When set, the
        # processor pipeline is loaded from this file; runtime slots are
        # injected at run time via .copy().
        # Pass via: --ak harness_config_yaml=/path/to/config.yaml
        # Or set env var TB2_HARNESS_CONFIG before launching eval.
        raw_cfg = harness_config_yaml or os.environ.get("TB2_HARNESS_CONFIG")
        self._harness_config_yaml = raw_cfg.strip() if isinstance(raw_cfg, str) and raw_cfg.strip() else None

    @staticmethod
    def name() -> str:
        return "harnessx"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:  # noqa: ARG002
        pass  # No setup required for HarnessX

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from harnessx import BaseTask
        from harnessx.core.harness import HarnessConfig
        from harnessx.core.model_config import ModelConfig
        from harnessx.tracing.journal import HarnessJournal
        from .harbor_sandbox import HarborSandboxProvider
        from .harness import make_tb2_harness_config

        # ── Build provider ────────────────────────────────────────────────────
        model = self.model_name or "openai/gpt-4o-mini"

        if (
            model.startswith("anthropic/")
            or model.startswith("claude-")
            or (self._api_base and "anthropic" in self._api_base)
        ):
            from harnessx.providers.anthropic_provider import AnthropicProvider

            anthropic_model = model[len("anthropic/") :]
            provider_kwargs: dict = {"timeout": float(self._request_timeout_sec)}
            if self._max_tokens is not None:
                provider_kwargs["max_tokens"] = self._max_tokens
            if self._api_key:
                provider_kwargs["api_key"] = self._api_key
            if self._api_base:
                # Anthropic SDK appends /v1/messages itself; strip trailing /v1
                # to avoid double-v1 (e.g. http://host/v1 → http://host)
                base = self._api_base.rstrip("/")
                provider_kwargs["base_url"] = base[:-3] if base.endswith("/v1") else base
            if self._extra_headers:
                provider_kwargs["default_headers"] = self._extra_headers
            provider = AnthropicProvider(model=anthropic_model, **provider_kwargs)
        else:
            from harnessx.providers.openai_provider import OpenAIProvider

            provider_kwargs = {}
            if self._max_tokens is not None:
                provider_kwargs["max_tokens"] = self._max_tokens
            if self._api_key:
                provider_kwargs["api_key"] = self._api_key
            if self._api_base:
                provider_kwargs["base_url"] = self._api_base
            if self._extra_headers:
                provider_kwargs["extra_headers"] = self._extra_headers
            provider = OpenAIProvider(model=model, **provider_kwargs)

        # ── Resolve task timeout ──────────────────────────────────────────────
        task_timeout = getattr(context, "timeout_sec", None)
        if not task_timeout:
            try:
                task_toml = environment.environment_dir.parent / "task.toml"
                m = re.search(r"timeout_sec\s*=\s*([0-9.]+)", task_toml.read_text())
                if m:
                    task_timeout = float(m.group(1))
            except Exception:
                pass
        task_timeout = task_timeout or 3600

        # ── Load harness config ───────────────────────────────────────────────
        # Load the processor pipeline from an evolved YAML when available,
        # otherwise build the default config.  Either way, inject the
        # runtime-only slots (sandbox_provider, tracer) via .copy() — these
        # are never serialised into the config YAML.
        if self._harness_config_yaml:
            base_config = HarnessConfig.from_yaml_file(self._harness_config_yaml)
        else:
            base_config = make_tb2_harness_config(timeout_seconds=task_timeout)

        harness_config = base_config.copy(
            sandbox_provider=HarborSandboxProvider(
                environment,
                workspace_path=WORKSPACE_PATH,
                output_limit=OUTPUT_LIMIT,
            ),
            tracer=HarnessJournal(
                base_dir=str(self.logs_dir / "oh_runs"),
                export_jsonl=True,
            ),
        )

        harness = ModelConfig(main=provider).agentic(harness_config)
        task = BaseTask(
            description=instruction,
            max_steps=self._max_steps,
            token_budget=TOKEN_BUDGET,
        )

        # ── Run ───────────────────────────────────────────────────────────────
        # Harbor enforces the wall-clock timeout (task.toml agent.timeout_sec)
        # externally; no internal asyncio.wait_for needed.
        try:
            result = await harness.run(task)
            context.n_input_tokens = result.total_tokens
            context.n_output_tokens = 0  # HarnessX tracks combined tokens

        except Exception:
            traceback.print_exc()
