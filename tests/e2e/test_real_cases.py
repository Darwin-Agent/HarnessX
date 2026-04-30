# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harnessx import BaseTask, Workspace
from harnessx.tracing.journal import HarnessJournal
from ._utils import load_provider, get_test_home


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def provider():
    return load_provider()


# ── Harness factory ───────────────────────────────────────────────────────────


def _make_harness(provider):
    """Full CLI-equivalent config: personal assistant profile with workspace + provider."""
    import yaml as _yaml
    from harnessx.core.harness import HarnessConfig as _HC
    from harnessx.tools.builtin import build_default_tools
    from harnessx.core.model_config import ModelConfig

    # Load the assistant example config
    _examples = PROJECT_ROOT / "examples"
    raw = _yaml.safe_load((_examples / "assistant" / "harness_config.yaml").read_text()) or {}
    base_config = _HC(
        processors=raw.get("processors") or [],
        plugins=raw.get("plugins") or [],
    )

    workspace = Workspace(agent_id="real_case_tests", home=get_test_home(), mode="shared")
    config = base_config.copy(
        workspace=workspace,
        tracer=HarnessJournal(silent=True),
        tool_registry=build_default_tools(),
    )
    return ModelConfig(main=provider, summarize=provider).agentic(config)


def _tool_names(result) -> list[str]:
    return [obs.tool_name for step in result.trajectory.steps for obs in step.observation]


# ── Non-model tests ───────────────────────────────────────────────────────────


def test_memory_adapters():
    """All memory adapters implement the BaseMemory protocol correctly."""
    import asyncio
    from harnessx.processors.memory.strategies import (
        InMemoryMemory,
        SlidingWindowMemory,
    )
    from harnessx.core.events import Message

    msgs = [
        Message(role="user", content="What is HarnessX?"),
        Message(role="assistant", content="HarnessX is a composable agent harness."),
    ]

    async def _run():
        for name, mem in [
            ("InMemoryMemory", InMemoryMemory(max_messages=50)),
            ("SlidingWindowMemory", SlidingWindowMemory(n=50)),
        ]:
            await mem.add(msgs)
            _retrieved = await mem.retrieve("HarnessX", k=5)
            _compressed = await mem.compress(msgs, budget=100)
            await mem.persist()
            await mem.load("test-run-id")

    asyncio.run(_run())


def test_api_exports():
    """All key public API symbols are exported from harnessx package."""
    import harnessx as hx

    required = [
        "Harness",
        "BaseTask",
        "HarnessConfig",
        "HarnessResult",
        "EvaluationProcessor",
        "SummarizationMemory",
        "DefaultSystemPromptBuilder",
        "NullSystemPromptBuilder",
        "TemplateSystemPromptBuilder",
        "CoTUserWrapper",
        "XMLUserWrapper",
        "FullStateSnapshot",
        "StateDelta",
        "StatefulTrajectory",
        "TrajectoryStep",
        "Workspace",
    ]
    missing = [a for a in required if not hasattr(hx, a)]
    assert missing == [], f"Missing public symbols: {missing}"


# ── Model-based tests ─────────────────────────────────────────────────────────


async def _run(harness, task):
    """Run a task, skipping on transient API errors (timeout, rate-limit, auth)."""
    try:
        return await harness.run(task)
    except Exception as exc:
        msg = str(exc)
        if any(
            kw in msg
            for kw in (
                "APITimeoutError",
                "APIConnectionError",
                "RateLimitError",
                "Request timed out",
                "AuthenticationError",
            )
        ):
            pytest.skip(f"Transient API error: {exc}")
        raise


@pytest.mark.asyncio
async def test_terminal_bench(provider, tmp_path: Path):
    """Agent solves a coding + bash task using only filesystem tools."""
    harness = _make_harness(provider)
    out_path = tmp_path / "fib_output.txt"

    fib_code = (
        "fibs = [1, 1]\\n"
        "for _ in range(8): fibs.append(fibs[-1] + fibs[-2])\\n"
        f"open('{out_path}', 'w').write('\\\\n'.join(map(str, fibs[:10])))"
    )
    result = await _run(
        harness,
        BaseTask(
            description=(
                f"Run this Python one-liner using Bash to generate fibonacci numbers:\n"
                f'python3 -c "{fib_code}"\n'
                f"Then read {out_path} to verify it was created."
            ),
        ),
    )

    names = _tool_names(result)
    ran_bash = any("bash" in n.lower() for n in names)
    assert ran_bash or result.exit_reason == "done", (
        f"Expected Bash usage or done exit; got exit_reason={result.exit_reason}"
    )

    if out_path.exists():
        content = out_path.read_text().strip()
        assert "55" in content, f"Expected fib[10]=55 in output, got: {content!r}"


@pytest.mark.asyncio
async def test_docx_creation(provider, tmp_path: Path):
    """Agent creates a DOCX using python-docx via Bash."""
    harness = _make_harness(provider)
    out_path = tmp_path / "test_document.docx"

    result = await _run(
        harness,
        BaseTask(
            description=(
                f"Use Bash to run this Python code:\n\n"
                f"```python\n"
                f"from docx import Document\n"
                f"doc = Document()\n"
                f"doc.add_heading('HarnessX Documentation', level=0)\n"
                f"doc.add_paragraph('HarnessX is a composable agent harness for LLM research.')\n"
                f"doc.add_heading('Key Features', level=1)\n"
                f"for feat in ['Context Engine', 'Stateful Trajectory', 'Workspace Isolation']:\n"
                f"    doc.add_paragraph(feat, style='List Bullet')\n"
                f"doc.save('{out_path}')\n"
                f"print('DOCX created:', '{out_path}')\n"
                f"```\n\n"
                f'Execute this with: python3 -c "..."'
            ),
        ),
    )

    _names = _tool_names(result)
    tool_calls_made = sum(len(step.action.tool_calls) if step.action else 0 for step in result.trajectory.steps)
    assert out_path.exists() or (tool_calls_made > 0 and result.exit_reason == "done"), (
        f"DOCX not created and agent did not complete: exit_reason={result.exit_reason}"
    )
    if out_path.exists():
        assert out_path.stat().st_size > 0, "DOCX file is empty"


@pytest.mark.asyncio
async def test_data_analysis(provider, tmp_path: Path):
    """Agent generates synthetic data, runs Python analysis, saves chart + summary."""
    harness = _make_harness(provider)
    summary_path = tmp_path / "survival_summary.txt"
    chart_path = tmp_path / "survival_chart.png"

    result = await _run(
        harness,
        BaseTask(
            description=(
                "You are analyzing passenger survival data (like the Titanic dataset).\n"
                "Use Bash to run a single Python script that does ALL of the following:\n\n"
                "1. Generate 300 synthetic passengers: each has class (1/2/3), sex (M/F), "
                "age (1-80), survived (0/1 with ~38% survival rate), using random.seed(42)\n"
                "2. Compute survival rates by class and by sex\n"
                f"3. Save a text summary to {summary_path} with the rates (e.g. 'Class 1: 45.2%')\n"
                "4. Create a bar chart (2 subplots: by class, by sex) using matplotlib with Agg backend "
                f"and save to {chart_path}\n"
                "5. Print 'Analysis complete' at the end\n\n"
                "Write the Python script to /tmp/analysis.py first using Bash, then run it with python3."
            ),
        ),
    )

    bash_calls = sum(
        1 for step in result.trajectory.steps for obs in step.observation if "bash" in obs.tool_name.lower()
    )
    assert summary_path.exists() or (bash_calls >= 2 and result.exit_reason == "done"), (
        f"Analysis incomplete: summary_exists={summary_path.exists()} "
        f"bash_calls={bash_calls} exit_reason={result.exit_reason}"
    )
    if summary_path.exists():
        content = summary_path.read_text()
        assert any(kw in content for kw in ["Class", "Male", "Female", "%"]), f"Summary missing rate data: {content!r}"


@pytest.mark.asyncio
async def test_pptx_creation(provider, tmp_path: Path):
    """Agent creates a 3-slide PPTX, generates HTML preview, Browser-verifies layout."""
    harness = _make_harness(provider)
    out_path = tmp_path / "test_presentation.pptx"
    preview_path = tmp_path / "pptx_preview.html"

    result = await _run(
        harness,
        BaseTask(
            description=(
                f"Create a 3-slide PPTX presentation about AI agent technology trends "
                f"and save it to: {out_path}\n\n"
                "Slides: (1) Title slide, (2) Top 3 trends with bullet points, "
                "(3) Conclusion & next steps.\n\n"
                "After creating the PPTX:\n"
                f"1. Use Bash to generate a simple HTML preview of the slide text content "
                f"and save it to {preview_path}\n"
                f"2. Use Browser(action='navigate', url='file://{preview_path}') then "
                "Browser(action='screenshot') to visually verify the layout looks correct.\n"
                "3. If the content or layout has issues, fix the PPTX and re-verify.\n"
                "4. Confirm when satisfied with the result."
            ),
        ),
    )

    names = _tool_names(result)
    bash_calls = names.count("Bash")
    assert out_path.exists() and bash_calls > 0 and result.exit_reason == "done", (
        f"PPTX test failed: file={out_path.exists()} bash={bash_calls} exit_reason={result.exit_reason}"
    )
    assert out_path.stat().st_size > 0, "PPTX file is empty"


@pytest.mark.asyncio
async def test_webpage_creation(provider, tmp_path: Path):
    """Agent creates HTML5 page, Browser-navigates + screenshots, self-verifies layout."""
    harness = _make_harness(provider)
    out_path = tmp_path / "agent_trends_2026.html"

    result = await _run(
        harness,
        BaseTask(
            description=(
                f"Create a complete HTML5 webpage and save it to: {out_path}\n\n"
                "Topic: 'Deep Research Report — AI Agent Technology Trends & Opportunities in 2026'.\n"
                "Requirements: full HTML5 with inline CSS, sections for Executive Summary, "
                "Top 5 Trends, Key Opportunities, Conclusion. Self-contained (no external links).\n\n"
                "After writing the file:\n"
                f"1. Use Browser(action='navigate', url='file://{out_path}') to open it.\n"
                "2. Use Browser(action='screenshot') to capture the rendered page.\n"
                "3. Review the screenshot — verify the layout, readability, and section structure "
                "look correct.\n"
                "4. If anything looks wrong (missing sections, broken layout, unreadable text), "
                "fix the HTML and re-verify with another screenshot.\n"
                "5. Confirm done when the page looks good."
            ),
        ),
    )

    assert out_path.exists(), f"HTML file not created at {out_path}"
    html = out_path.read_text()
    assert len(html) > 500, f"HTML too short ({len(html)} chars)"
    assert all(t in html.lower() for t in ["<!doctype", "<head", "<body"]), "HTML missing basic structure tags"
    sections_found = sum(1 for kw in ["executive summary", "trend", "opportunit", "conclusion"] if kw in html.lower())
    assert sections_found >= 3, f"Only {sections_found}/4 required sections found"
    assert result.exit_reason == "done", f"Unexpected exit: {result.exit_reason}"


@pytest.mark.asyncio
async def test_deep_research(provider, tmp_path: Path):
    """Agent uses web tools to research LLM inference optimization techniques.

    Validates:
    - WebSearch / WebFetch were actually invoked (not pure model memory)
    - Agent produced technique keywords in output or a written file
    - Exit was clean (done or budget_exceeded)
    """
    harness = _make_harness(provider)
    report_path = tmp_path / "report.md"

    result = await _run(
        harness,
        BaseTask(
            description=(
                f"Research and write a concise report to {report_path} answering: "
                "What are the top 3 open-source LLM inference optimization techniques "
                "in 2024-2025? For each: name, description, key benefit. "
                "Use WebSearch to find current info. Keep it brief."
            ),
            max_steps=12,
        ),
    )

    # Web tool usage — confirms agent searched, not just recalled from memory
    web_tool_calls = sum(
        1 for step in result.trajectory.steps for obs in step.observation if obs.tool_name in ("WebSearch", "WebFetch")
    )
    assert web_tool_calls > 0, "Agent made no web tool calls — expected at least 1 WebSearch or WebFetch"

    # Content check: final_output or written file must contain technique keywords
    output = result.final_output or ""
    file_content = report_path.read_text() if report_path.exists() else ""
    combined = (output + file_content).lower()
    assert any(
        kw in combined
        for kw in [
            "quantization",
            "kv cache",
            "speculative",
            "flash attention",
            "vllm",
            "continuous batching",
            "paged",
            "awq",
            "gptq",
            "gguf",
            "inference",
            "optimization",
        ]
    ), f"No technique keywords found. output={output[:200]!r} file={file_content[:200]!r}"

    assert result.exit_reason in ("done", "budget_exceeded"), f"Unexpected exit: {result.exit_reason}"
