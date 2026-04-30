# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any


def _is_anthropic_model_name(model: str) -> bool:
    m = model.strip().lower()
    return m.startswith("claude-") or m.startswith("anthropic/")


def _resolve_model_config(model_override: str | None = None) -> tuple[Any, str]:
    """Build ModelConfig from env vars, with optional --model override."""
    from harnessx.core.model_config import ModelConfig
    from harnessx.providers.anthropic_provider import AnthropicProvider
    from harnessx.providers.litellm_provider import LiteLLMProvider

    env = os.environ
    requested = (model_override or "").strip()

    if requested:
        if _is_anthropic_model_name(requested):
            kwargs: dict[str, str] = {}
            if env.get("ANTHROPIC_API_KEY"):
                kwargs["api_key"] = env["ANTHROPIC_API_KEY"]
            base_url = env.get("ANTHROPIC_API_BASE") or env.get("ANTHROPIC_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            provider = AnthropicProvider(requested, **kwargs)
            return ModelConfig(main=provider), f"AnthropicProvider/{provider.model}"

        kwargs = {}
        # Prefer OpenAI env when present; otherwise fall back to LiteLLM env.
        if env.get("OPENAI_API_KEY"):
            kwargs["api_key"] = env["OPENAI_API_KEY"]
            if env.get("OPENAI_API_BASE"):
                kwargs["api_base"] = env["OPENAI_API_BASE"]
        elif env.get("LITELLM_API_KEY"):
            kwargs["api_key"] = env["LITELLM_API_KEY"]
            if env.get("LITELLM_API_BASE"):
                kwargs["api_base"] = env["LITELLM_API_BASE"]
        provider = LiteLLMProvider(requested, **kwargs)
        return ModelConfig(main=provider), f"LiteLLMProvider/{provider.model}"

    # Priority aligns with CLI defaults: Anthropic -> OpenAI -> LiteLLM.
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_DEFAULT_MAIN_MODEL"):
        model = env.get("ANTHROPIC_DEFAULT_MAIN_MODEL", "claude-sonnet-4-6")
        kwargs = {}
        if env.get("ANTHROPIC_API_KEY"):
            kwargs["api_key"] = env["ANTHROPIC_API_KEY"]
        base_url = env.get("ANTHROPIC_API_BASE") or env.get("ANTHROPIC_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        provider = AnthropicProvider(model, **kwargs)
        return ModelConfig(main=provider), f"AnthropicProvider/{provider.model}"

    if env.get("OPENAI_API_KEY") or env.get("OPENAI_DEFAULT_MAIN_MODEL"):
        model = env.get("OPENAI_DEFAULT_MAIN_MODEL", "gpt-4o-mini")
        kwargs = {}
        if env.get("OPENAI_API_KEY"):
            kwargs["api_key"] = env["OPENAI_API_KEY"]
        if env.get("OPENAI_API_BASE"):
            kwargs["api_base"] = env["OPENAI_API_BASE"]
        provider = LiteLLMProvider(model, **kwargs)
        return ModelConfig(main=provider), f"LiteLLMProvider/{provider.model}"

    if env.get("LITELLM_API_KEY") or env.get("LITELLM_DEFAULT_MAIN_MODEL"):
        model = env.get("LITELLM_DEFAULT_MAIN_MODEL", "openai/gpt-4o-mini")
        kwargs = {}
        if env.get("LITELLM_API_KEY"):
            kwargs["api_key"] = env["LITELLM_API_KEY"]
        if env.get("LITELLM_API_BASE"):
            kwargs["api_base"] = env["LITELLM_API_BASE"]
        provider = LiteLLMProvider(model, **kwargs)
        return ModelConfig(main=provider), f"LiteLLMProvider/{provider.model}"

    raise RuntimeError(
        "No model credentials detected.\n"
        "Set one of these before running:\n"
        "  - ANTHROPIC_API_KEY (+ optional ANTHROPIC_DEFAULT_MAIN_MODEL)\n"
        "  - OPENAI_API_KEY (+ optional OPENAI_DEFAULT_MAIN_MODEL)\n"
        "  - LITELLM_API_KEY (+ optional LITELLM_DEFAULT_MAIN_MODEL)\n"
        "Or pass --model explicitly with matching provider credentials."
    )


def _seed_demo_workflow(workflow_dir: Path) -> Path:
    """Create a reusable sample workflow YAML for `flow_exec` demos."""
    workflow_dir.mkdir(parents=True, exist_ok=True)
    wf_path = workflow_dir / "git-health.yaml"
    if wf_path.exists():
        return wf_path

    wf_path.write_text(
        "\n".join(
            [
                "name: git-health",
                "description: Check git repo health (recent commits, branch, and dirty files).",
                "tags:",
                "  - git",
                "  - health",
                "trigger_patterns:",
                "  - check git health",
                "  - summarize git repo status",
                "params:",
                "  - name: max_log",
                "    description: Max number of recent commits to show.",
                '    default: "5"',
                "steps:",
                "  - id: log",
                "    description: Show recent commit history.",
                '    shell: git log --oneline -$max_log 2>/dev/null || echo "not a git repository"',
                "  - id: branch",
                "    shell: git branch --show-current 2>/dev/null || echo unknown",
                "  - id: dirty",
                "    shell: git status --short 2>/dev/null | head -20 || echo clean",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return wf_path


def _build_harness_config(workflow_dir: Path, *, internalize: bool) -> tuple[Any, list[str]]:
    """Build a runnable HarnessConfig with default tools + WorkflowPlugin."""
    from harnessx.bundles.context import make_context
    from harnessx.bundles.control import make_control
    from harnessx.bundles.execution import make_execution
    from harnessx.core.builder import HarnessBuilder
    from harnessx.processors.context.env_context_injector import (
        EnvironmentContextInjector,
    )
    from harnessx.tools.builtin import build_default_tools

    from extensions.plugins.workflow import WorkflowPlugin

    builder = HarnessBuilder().slot(tool_registry=build_default_tools())
    builder = (
        builder | make_context() | make_execution() | make_control(include_reliability=True, include_budget=False)
    ).add(EnvironmentContextInjector(working_dir=str(Path.cwd())))

    builder = builder.plugin(
        WorkflowPlugin(
            workflow_dir=str(workflow_dir),
            guidance=True,
            recall=True,
            internalize=internalize,
            # Keep internalization off by default in this demo.
            # Pass --enable-internalize only when extractor model is configured.
            judge_model=None,
            extractor_model=None,
            complexity_threshold=5,
        )
    )

    config = builder.build()
    tool_names = sorted(config.tool_registry.list_names()) if config.tool_registry else []
    return config, tool_names


async def _run_turn(
    harness: Any,
    text: str,
    *,
    session_id: str,
    resume_state: Any | None,
    max_steps: int,
) -> Any | None:
    from harnessx import BaseTask

    result = await harness.run(
        BaseTask(description=text, max_steps=max_steps),
        session_id=session_id,
        _resume_state=resume_state,
    )

    output = (result.final_output or "").strip()
    print("\nassistant:")
    print(output or "(empty response)")
    print(
        f"[steps={result.total_steps} "
        f"prompt={result.total_input_tokens} "
        f"complete={result.total_output_tokens} "
        f"exit={result.exit_reason}]"
    )
    return result.resume_state


async def _interactive_loop(harness: Any, *, max_steps: int, bootstrap: bool) -> None:
    session_id = "workflow-demo"
    resume_state: Any | None = None

    if bootstrap:
        print("\n--- bootstrap task ---")
        print("user:")
        bootstrap_prompt = (
            "Use the `flow_exec` tool with name='git-health' and params={'max_log': '3'}. "
            "Then summarize the repo status in three bullets."
        )
        print(bootstrap_prompt)
        resume_state = await _run_turn(
            harness,
            bootstrap_prompt,
            session_id=session_id,
            resume_state=resume_state,
            max_steps=max_steps,
        )

    print("\n--- interactive mode ---")
    print("Type your own task. Type 'exit' to quit.")
    print("Suggested prompts:")
    print("  1) Check git health again and use any matching stored workflow.")
    print("  2) Use flow to run a small multi-step shell pipeline for repo inspection.")
    print("  3) Create an approval-gated cleanup flow demo and wait for my confirmation.")

    while True:
        raw = (await asyncio.to_thread(input, "\nworkflow-demo> ")).strip()
        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            print("bye")
            return

        resume_state = await _run_turn(
            harness,
            raw,
            session_id=session_id,
            resume_state=resume_state,
            max_steps=max_steps,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorkflowPlugin runnable demo.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. Example: claude-sonnet-4-6 or gpt-4o-mini",
    )
    parser.add_argument(
        "--workflow-dir",
        default=None,
        help="Directory for workflow YAML files. Defaults to ~/.harnessx/workflow-demo",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps per user turn (default: 20).",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip the automatic first turn that demonstrates flow_exec.",
    )
    parser.add_argument(
        "--enable-internalize",
        action="store_true",
        help="Enable workflow internalization (requires extractor model setup).",
    )
    return parser.parse_args()


async def main() -> None:
    from harnessx.home import agent_home

    args = _parse_args()

    workflow_dir = (
        Path(args.workflow_dir).expanduser().resolve() if args.workflow_dir else (agent_home() / "workflow-demo")
    )
    seeded = _seed_demo_workflow(workflow_dir)

    harness_config, tool_names = _build_harness_config(
        workflow_dir,
        internalize=bool(args.enable_internalize),
    )

    try:
        model_config, model_label = _resolve_model_config(args.model)
    except RuntimeError as exc:
        print(str(exc))
        return

    harness = model_config.agentic(harness_config)

    print("=" * 70)
    print("WorkflowPlugin demo is ready")
    print("=" * 70)
    print(f"model:        {model_label}")
    print(f"workflow dir: {workflow_dir}")
    print(f"seeded file:  {seeded.name}")
    print(f"tool count:   {len(tool_names)}")
    print(
        "core tools:   "
        + ", ".join(
            name for name in ["flow", "flow_resume", "flow_exec", "Bash", "Read", "Write"] if name in tool_names
        )
    )
    print(f"internalize:  {'on' if args.enable_internalize else 'off'}")

    await _interactive_loop(
        harness,
        max_steps=max(1, int(args.max_steps)),
        bootstrap=not args.no_bootstrap,
    )


if __name__ == "__main__":
    asyncio.run(main())
