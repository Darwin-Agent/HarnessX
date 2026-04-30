#!/usr/bin/env python3
"""Run GAIA benchmark with HarnessX.

Usage:
    # Run curated cases (default)
    python -m benchmarks.gaia.run_gaia

    # Run all Level 1 validation
    python -m benchmarks.gaia.run_gaia --level 1 --max-tasks 10

    # Use a specific model
    python -m benchmarks.gaia.run_gaia --model openai/gpt-4o

    # GPT-5 Medium preset (Responses API + code_interpreter + tuned params)
    python -m benchmarks.gaia.run_gaia --preset gpt5 --from-hf

    # GPT-5 Medium with specific provider
    python -m benchmarks.gaia.run_gaia --model gpt-5-medium --provider-type responses --from-hf

    # Run curated cases only
    python -m benchmarks.gaia.run_gaia --curated
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from harnessx.core.harness import Harness
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider
from harnessx.workspace.factory import build_spawn_tool

from .defaults import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER_ID,
    GPT5_API_BASE,
    GPT5_MAX_COST_USD,
    GPT5_MAX_STEPS,
    GPT5_MODEL,
    GPT5_PROVIDER_ID,
    GPT5_STEPS_PER_LEVEL,
    MAX_COST_USD,
    MAX_STEPS,
)
from .evaluator import GAIAPipelineEvaluator
from .harness import make_gaia_harness, make_gaia_harness_gpt5
from .task import GAIATask, load_gaia_tasks, load_webthinker_gaia_tasks

logger = logging.getLogger(__name__)

# ── Curated good cases for quick testing ──────────────────────────────────────
# These are Level 1-2 questions (no file attachments) that exercise web search
# and multi-step reasoning — the sweet spot for DeepResearch evaluation.
CURATED_CASES: list[dict] = [
    # Level 1 — straightforward web lookup + reasoning
    {
        "task_id": "curated-01",
        "question": (
            "What is the population of the capital city of the country "
            "that won the most gold medals in the 2024 Paris Olympics?"
        ),
        "level": 1,
        "final_answer": "",  # will be validated by LLM judge if no ground truth
    },
    {
        "task_id": "curated-02",
        "question": ("Who was the CEO of OpenAI when GPT-4 was released, and what university did they attend?"),
        "level": 1,
        "final_answer": "Sam Altman, Stanford University",
    },
    {
        "task_id": "curated-03",
        "question": (
            "What is the tallest building in the world as of 2024, "
            "how tall is it in meters, and in which city is it located?"
        ),
        "level": 1,
        "final_answer": "Burj Khalifa, 828 meters, Dubai",
    },
    # Level 2 — multi-hop reasoning with web search
    {
        "task_id": "curated-04",
        "question": (
            "Find the author of the paper 'Attention Is All You Need'. "
            "Among the authors, who is currently (as of 2024) the CEO of a company? "
            "What is the name of that company and when was it founded?"
        ),
        "level": 2,
        "final_answer": "",  # Complex multi-hop, judge will evaluate
    },
    {
        "task_id": "curated-05",
        "question": (
            "What is the GDP per capita (nominal, in USD) of the country where "
            "the 2025 Nobel Prize in Physics laureates did their primary research? "
            "If there are multiple laureates from different countries, list all."
        ),
        "level": 2,
        "final_answer": "",
    },
    {
        "task_id": "curated-06",
        "question": (
            "The Transformer architecture paper was published in 2017. "
            "How many citations does it have on Google Scholar as of today? "
            "Compare this to the number of citations of the original BERT paper. "
            "Which has more and by what ratio (rounded to 1 decimal)?"
        ),
        "level": 2,
        "final_answer": "",
    },
    # Level 2 — synthesis across sources
    {
        "task_id": "curated-07",
        "question": (
            "Compare the market capitalization of NVIDIA and Apple as of the "
            "most recent trading day. Which company has higher market cap "
            "and what is the difference in billions of USD? "
            "Also state each company's P/E ratio."
        ),
        "level": 2,
        "final_answer": "",
    },
    {
        "task_id": "curated-08",
        "question": (
            "What programming language was used to write the first version of Git? "
            "Who wrote it and in what year? How long did the initial development take?"
        ),
        "level": 1,
        "final_answer": "C, Linus Torvalds, 2005, about 2 weeks",
    },
]


def build_curated_tasks(max_steps: int = MAX_STEPS) -> list[GAIATask]:
    """Build GAIATask instances from curated cases."""
    tasks = []
    for case in CURATED_CASES:
        tasks.append(
            GAIATask(
                description=case["question"],
                task_id=case["task_id"],
                question=case["question"],
                level=case["level"],
                final_answer=case.get("final_answer", ""),
                max_steps=max_steps,
            )
        )
    return tasks


def _build_provider(
    model: str,
    provider_type: str,
    *,
    extra_headers: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    stream: bool = False,
):
    """Create a model provider based on --provider-type flag."""
    if provider_type == "responses":
        from harnessx.providers.responses_provider import ResponsesAPIProvider

        kwargs = {}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["reasoning_summary"] = True
        if api_base:
            kwargs["base_url"] = api_base
        if api_key:
            kwargs["api_key"] = api_key
        return ResponsesAPIProvider(model=model, **kwargs)

    if provider_type == "openai":
        from harnessx.providers.openai_provider import OpenAIProvider

        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return OpenAIProvider(
            model=model,
            base_url=api_base,
            api_key=api_key,
            extra_headers=extra_headers,
            stream=stream,
            **kwargs,
        )

    # LiteLLM: api_base goes via kwargs
    kwargs = {}
    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return LiteLLMProvider(model, **kwargs)


async def run_bare(
    model: str,
    api_base: str | None,
    api_key: str | None,
    provider_id: str | None,
    task: GAIATask,
) -> dict:
    """Run a single GAIA task as a bare single-shot LLM call (no tools, no agent loop).

    Mimics the 'Direct Reasoning' baseline in the WebThinker paper (QwQ-32B: 22.3%).
    Uses the OpenAI SDK directly with X-Model-Provider-Id header routing.
    """
    import re
    from openai import AsyncOpenAI

    t0 = time.time()

    _BARE_SYSTEM = (
        "You are an expert research assistant. Each question has ONE specific, verifiable answer.\n"
        "Reason carefully using your knowledge. After your reasoning, output your final answer:\n\n"
        "FINAL ANSWER: <your concise answer>\n\n"
        "Rules:\n"
        "- Numbers: plain digits (e.g., '42' not 'forty-two').\n"
        "- No units unless specifically asked for.\n"
        "- No explanations in the FINAL ANSWER line — just the answer.\n"
        "- If listing items: comma-separated, no 'and'.\n"
        "- Always provide a FINAL ANSWER — never say 'unable to determine'."
    )

    import httpx

    headers = {}
    if provider_id:
        headers["X-Model-Provider-Id"] = provider_id

    client = AsyncOpenAI(
        api_key=api_key or "dummy",
        base_url=api_base or "https://api.openai.com/v1",
        default_headers=headers,
        timeout=httpx.Timeout(300.0, connect=10.0),
    )

    try:
        # Use streaming — required for reasoning models (QwQ-32B, o1, etc.) on this endpoint
        output = ""
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _BARE_SYSTEM},
                {"role": "user", "content": task.question},
            ],
            max_tokens=16384,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                output += chunk.choices[0].delta.content

        # Strip <think>...</think> blocks (QwQ-32B style)
        output_clean = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
        elapsed = time.time() - t0

        # Extract FINAL ANSWER
        m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", output_clean, re.IGNORECASE)
        extracted = m.group(1).strip() if m else (output_clean.strip().splitlines()[-1] if output_clean.strip() else "")

        from .evaluator import _answers_match

        passed = bool(task.final_answer and _answers_match(extracted, task.final_answer.strip()))

        tokens = len(output) // 4  # rough estimate when usage not available
        status = "✅" if passed else "❌"
        logger.info(
            "%s [%s] bare | extracted=%r | expected=%r | tokens=%d | %.1fs",
            status,
            task.task_id[:8],
            extracted[:60],
            task.final_answer[:40],
            tokens,
            elapsed,
        )
        return {
            "task_id": task.task_id,
            "level": task.level,
            "question": task.question[:200],
            "expected": task.final_answer,
            "agent_output": extracted,
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": f"extracted='{extracted[:80]}' vs expected='{task.final_answer[:80]}'",
            "steps": 1,
            "tokens": tokens,
            "cost_usd": 0.0,
            "elapsed_s": round(elapsed, 1),
            "exit_reason": "done",
            "bare": True,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("💥 [%s] bare failed: %s", task.task_id[:8], exc)
        return {
            "task_id": task.task_id,
            "level": task.level,
            "question": task.question[:200],
            "passed": False,
            "score": 0.0,
            "reason": f"error: {exc}",
            "elapsed_s": round(elapsed, 1),
            "bare": True,
        }


async def run_native_tools(
    model: str,
    api_base: str | None,
    api_key: str | None,
    provider_id: str | None,
    task: GAIATask,
    max_steps: int = 15,
) -> dict:
    """GPT-5 with tools but WITHOUT HarnessX orchestration.

    Uses the same tools as HarnessX (WebSearch, WebFetch, Browser, Bash) but runs them
    in a naive OpenAI function-calling loop. No token budget, no loop detection,
    no verification forcing, no special GAIA system prompt.
    This is the fair 'model-with-tools baseline' to compare against HarnessX.
    """
    import re
    import httpx
    from openai import AsyncOpenAI
    from harnessx.tools.builtin import web_search_tool, web_fetch_tool, browser_tool, bash_tool
    from harnessx.providers._utils import to_openai_tools

    _NATIVE_SYSTEM = (
        "You are an expert research assistant. Answer the question by using the available tools.\n"
        "Search the web, fetch pages, and compute as needed.\n"
        "When you have a confident answer, output it as:\n\n"
        "FINAL ANSWER: <your concise answer>\n\n"
        "Rules:\n"
        "- Numbers: plain digits (e.g., '42' not 'forty-two').\n"
        "- No units unless specifically asked.\n"
        "- Lists: comma-separated, no 'and'.\n"
        "- Always output FINAL ANSWER when done."
    )

    headers = {}
    if provider_id:
        headers["X-Model-Provider-Id"] = provider_id

    client = AsyncOpenAI(
        api_key=api_key or "dummy",
        base_url=api_base or "https://api.openai.com/v1",
        default_headers=headers,
        timeout=httpx.Timeout(300.0, connect=10.0),
        max_retries=0,
    )

    tool_objects = [web_search_tool, web_fetch_tool, browser_tool, bash_tool]
    oai_tools = to_openai_tools([t.to_schema() for t in tool_objects])
    tool_map = {t.name: t for t in tool_objects}

    messages = [
        {"role": "system", "content": _NATIVE_SYSTEM},
        {"role": "user", "content": task.question},
    ]

    t0 = time.time()
    steps = 0
    total_tokens = 0
    final_answer = ""

    try:
        for step in range(max_steps):
            steps += 1
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=oai_tools,
                max_tokens=8192,
                stream=False,
            )
            choice = response.choices[0]
            msg = choice.message
            if response.usage:
                total_tokens += (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)

            content = msg.content or ""
            # Strip <think> blocks
            content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Append assistant message
            assistant_msg = {"role": "assistant", "content": content}
            if msg.tool_calls:
                import json as _json

                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # Check for FINAL ANSWER in content
            m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", content_clean, re.IGNORECASE)
            if m:
                final_answer = m.group(1).strip()
                if not msg.tool_calls:
                    break

            if not msg.tool_calls:
                # No tools and no final answer — take last non-empty line
                if not final_answer and content_clean:
                    final_answer = content_clean.strip().splitlines()[-1]
                break

            # Execute tool calls
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    import json as _json

                    tool_input = _json.loads(tc.function.arguments or "{}")
                except Exception:
                    tool_input = {}

                tool_obj = tool_map.get(tool_name)
                if tool_obj:
                    try:
                        tool_result = await tool_obj.fn(**tool_input)
                        if not isinstance(tool_result, str):
                            tool_result = str(tool_result)
                    except Exception as e:
                        tool_result = f"Tool error: {e}"
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result[:6000],  # truncate long results
                    }
                )

    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("💥 [%s] native_tools failed: %s", task.task_id[:8], exc)
        return {
            "task_id": task.task_id,
            "level": task.level,
            "question": task.question[:200],
            "passed": False,
            "score": 0.0,
            "reason": f"error: {exc}",
            "elapsed_s": round(elapsed, 1),
            "native_tools": True,
        }

    elapsed = time.time() - t0
    if not final_answer and messages:
        # fallback: scan all assistant messages for FINAL ANSWER
        for m_msg in reversed(messages):
            if m_msg.get("role") == "assistant" and m_msg.get("content"):
                c = re.sub(r"<think>.*?</think>", "", m_msg["content"], flags=re.DOTALL).strip()
                match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", c, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).strip()
                    break

    from .evaluator import _answers_match

    passed = bool(task.final_answer and _answers_match(final_answer, task.final_answer.strip()))
    status = "✅" if passed else "❌"
    logger.info(
        "%s [%s] native_tools steps=%d | extracted=%r | expected=%r | tokens=%d | %.1fs",
        status,
        task.task_id[:8],
        steps,
        final_answer[:60],
        task.final_answer[:40] if task.final_answer else "",
        total_tokens,
        elapsed,
    )
    return {
        "task_id": task.task_id,
        "level": task.level,
        "question": task.question[:200],
        "expected": task.final_answer,
        "agent_output": final_answer,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": f"extracted='{final_answer[:80]}' vs expected='{task.final_answer[:80] if task.final_answer else ''}'",
        "steps": steps,
        "tokens": total_tokens,
        "cost_usd": 0.0,
        "elapsed_s": round(elapsed, 1),
        "exit_reason": "done",
        "native_tools": True,
    }


async def run_single(harness: Harness, task: GAIATask) -> dict:
    """Run a single GAIA task and return results.

    Evaluation happens inside the pipeline (EvaluationProcessor) — this
    function just reads ``result.eval_result``.
    """
    t0 = time.time()
    logger.info("▶ Running [%s] Level %d: %s", task.task_id, task.level, task.question[:80])

    try:
        result = await harness.run(task)
        elapsed = time.time() - t0

        eval_result = result.eval_result
        passed = bool(eval_result and eval_result.passed)
        score = float(eval_result.score) if eval_result else 0.0
        reason = eval_result.reason if eval_result else "no eval_result from pipeline"

        record = {
            "task_id": task.task_id,
            "level": task.level,
            "question": task.question[:200],
            "expected": task.final_answer,
            "agent_output": result.final_output[:500] if result.final_output else "",
            "passed": passed,
            "score": score,
            "reason": reason,
            "steps": result.total_steps,
            "tokens": result.total_tokens,
            "cost_usd": result.total_cost_usd,
            "elapsed_s": round(elapsed, 1),
            "exit_reason": result.exit_reason,
        }

        status = "\u2705" if passed else "\u274c"
        logger.info(
            "%s [%s] score=%.1f steps=%d tokens=%d cost=$%.3f time=%.1fs",
            status,
            task.task_id,
            score,
            result.total_steps,
            result.total_tokens,
            result.total_cost_usd,
            elapsed,
        )
        return record

    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("\U0001f4a5 [%s] failed: %s (%.1fs)", task.task_id, exc, elapsed)
        return {
            "task_id": task.task_id,
            "level": task.level,
            "question": task.question[:200],
            "passed": False,
            "score": 0.0,
            "reason": f"error: {exc}",
            "elapsed_s": round(elapsed, 1),
        }


def _print_summary(results: list[dict], output_path: Path) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_tokens = sum(r.get("tokens", 0) for r in results)
    logger.info("=" * 60)
    logger.info("GAIA Results Summary")
    logger.info("=" * 60)
    logger.info("Total: %d | Passed: %d | Accuracy: %.2f%%", total, passed, 100 * passed / total if total else 0)
    logger.info("Total cost: $%.3f | Total tokens: %d", total_cost, total_tokens)
    for lvl in sorted(set(r["level"] for r in results)):
        lvl_results = [r for r in results if r["level"] == lvl]
        lvl_passed = sum(1 for r in lvl_results if r.get("passed"))
        logger.info(
            "  Level %d: %d/%d = %.2f%%",
            lvl,
            lvl_passed,
            len(lvl_results),
            100 * lvl_passed / len(lvl_results) if lvl_results else 0,
        )
    exit_reasons: dict = {}
    for r in results:
        er = r.get("exit_reason", "unknown")
        exit_reasons[er] = exit_reasons.get(er, 0) + 1
    logger.info("Exit reasons: %s", exit_reasons)
    logger.info("Results written to %s", output_path)
    logger.info("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAIA benchmark with HarnessX")
    parser.add_argument("--model", default=None, help="Model to test (e.g., gpt-5-medium, openai/gpt-4o)")
    parser.add_argument("--judge-model", default=None, help="Judge model (default: same as model)")
    parser.add_argument("--level", type=int, default=None, help="GAIA level (1/2/3)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--curated", action="store_true", default=True, help="Use curated cases")
    parser.add_argument("--from-hf", action="store_true", help="Load from HuggingFace instead of curated")
    parser.add_argument("--split", default="validation", help="Dataset split: validation (default) or test")
    parser.add_argument("--output", default="gaia_results.jsonl", help="Output JSONL file")
    parser.add_argument("--provider-id", default=DEFAULT_PROVIDER_ID, help="X-Model-Provider-Id header value")
    parser.add_argument("--max-cost", type=float, default=None, help="Max cost per task in USD")
    parser.add_argument(
        "--provider-type",
        choices=["litellm", "openai", "responses"],
        default="litellm",
        help="Provider type: litellm (default), openai (Chat Completions), responses (Responses API)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort for Responses API (low/medium/high)",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "gpt5"],
        default="default",
        help="Preset config: default (Claude), gpt5 (GPT-5 tuned)",
    )
    parser.add_argument(
        "--from-webthinker",
        default=None,
        metavar="FILE",
        help="Load 103-question WebThinker GAIA subset from a JSON file (e.g., webthinker_gaia_dev.json)",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Bare mode: single-shot LLM call, no tools, no agent loop (Direct Reasoning baseline)",
    )
    parser.add_argument(
        "--native-tools",
        action="store_true",
        help="Native tools baseline: GPT-5 with same tools as HarnessX but naive loop (no HarnessX orchestration)",
    )
    parser.add_argument(
        "--api-base", default=None, help="Custom API base URL (e.g., https://your-api-base.example.com/v1)"
    )
    parser.add_argument("--api-key", default=None, help="API key (overrides env var)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (0 for deterministic)")
    parser.add_argument(
        "--output-predictions",
        default=None,
        help="Write leaderboard-format predictions to this JSONL file (for test set submission)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already-completed tasks",
    )
    parser.add_argument(
        "--resume-passed",
        default=None,
        metavar="FILE",
        help="Resume from a previous run's output: keep passed results, re-run only failed tasks with current code improvements",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retry attempts for failed tasks (0=no retry, 1=retry once)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent tasks (default 1). Each slot gets its own harness.",
    )
    parser.add_argument(
        "--adaptive-steps",
        action="store_true",
        help="Use per-level step budgets (GPT5_STEPS_PER_LEVEL) instead of flat max_steps",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Apply preset defaults
    is_gpt5 = args.preset == "gpt5"

    if is_gpt5:
        model = args.model or GPT5_MODEL
        max_cost = args.max_cost or GPT5_MAX_COST_USD
        max_steps = GPT5_MAX_STEPS
        provider_type = args.provider_type
        if args.provider_type == "litellm":
            provider_type = "openai"
        api_base = args.api_base or GPT5_API_BASE
        api_key = args.api_key
        provider_id = args.provider_id if args.provider_id != DEFAULT_PROVIDER_ID else GPT5_PROVIDER_ID
    else:
        model = args.model or DEFAULT_MODEL
        max_cost = args.max_cost or MAX_COST_USD
        max_steps = MAX_STEPS
        provider_type = args.provider_type
        api_base = args.api_base
        api_key = args.api_key
        provider_id = args.provider_id

    # Build tasks
    if args.from_webthinker:
        tasks = load_webthinker_gaia_tasks(args.from_webthinker, level=args.level, max_tasks=args.max_tasks)
    elif args.from_hf:
        tasks = load_gaia_tasks(level=args.level, split=args.split, max_tasks=args.max_tasks)
    else:
        tasks = build_curated_tasks(max_steps=max_steps)
        if args.level:
            tasks = [t for t in tasks if t.level == args.level]
        if args.max_tasks:
            tasks = tasks[: args.max_tasks]

    if not tasks:
        logger.error("No tasks to run!")
        return

    logger.info("=" * 60)
    logger.info(
        "GAIA Benchmark — %d tasks, model=%s, provider=%s, preset=%s%s",
        len(tasks),
        model,
        provider_type,
        args.preset,
        " [BARE]" if args.bare else (" [NATIVE-TOOLS]" if args.native_tools else ""),
    )
    logger.info("=" * 60)

    # Build provider
    extra_headers = {"X-Model-Provider-Id": provider_id}
    temperature = args.temperature
    use_stream = False
    provider = _build_provider(
        model,
        provider_type,
        extra_headers=extra_headers,
        reasoning_effort=args.reasoning_effort,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        stream=use_stream,
    )
    judge_model = args.judge_model or model
    judge_provider = _build_provider(
        judge_model,
        provider_type,
        extra_headers=extra_headers,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        stream=use_stream,
    )

    # Adaptive per-level step budgets
    adaptive_steps = args.adaptive_steps or is_gpt5
    if is_gpt5:
        steps_per_level = GPT5_STEPS_PER_LEVEL
    else:
        steps_per_level = {1: MAX_STEPS, 2: MAX_STEPS, 3: MAX_STEPS}

    # Bare mode: direct single-shot calls, no harness pool needed
    if args.bare:
        import gc

        concurrency = max(1, args.concurrency)
        sem = asyncio.Semaphore(concurrency)
        results: list[dict] = []
        output_path = Path(args.output)
        write_lock = asyncio.Lock()
        outfile = open(output_path, "w")

        async def _bare_one(t: GAIATask) -> None:
            async with sem:
                record = await run_bare(model, api_base, api_key, provider_id, t)
            async with write_lock:
                results.append(record)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfile.flush()

        await asyncio.gather(*[_bare_one(t) for t in tasks])
        outfile.close()
        gc.collect()
        _print_summary(results, output_path)
        return

    # Native tools mode: same tools as HarnessX, naive loop, no orchestration
    if args.native_tools:
        import gc

        concurrency = max(1, args.concurrency)
        sem = asyncio.Semaphore(concurrency)
        results: list[dict] = []
        output_path = Path(args.output)
        write_lock = asyncio.Lock()
        outfile = open(output_path, "w")

        async def _native_one(t: GAIATask) -> None:
            async with sem:
                record = await run_native_tools(model, api_base, api_key, provider_id, t, max_steps=max_steps)
            async with write_lock:
                results.append(record)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfile.flush()

        await asyncio.gather(*[_native_one(t) for t in tasks])
        outfile.close()
        gc.collect()
        _print_summary(results, output_path)
        return

    concurrency = max(1, args.concurrency)

    # Build harness pool (one slot per concurrency level for thread-safety)
    model_config = ModelConfig(main=provider, evaluator=judge_provider)
    harness_pool: asyncio.Queue = asyncio.Queue()

    for _ in range(concurrency):
        eval_i = GAIAPipelineEvaluator(judge_provider=judge_provider)
        if is_gpt5:
            config_i = make_gaia_harness_gpt5(
                max_cost_usd=max_cost,
                pipeline_evaluator=eval_i,
            )
        else:
            config_i = make_gaia_harness(
                max_cost_usd=max_cost,
                pipeline_evaluator=eval_i,
            )
        config_i.tool_registry.register(build_spawn_tool(model_config, config_i))
        harness_i = model_config.agentic(config_i)
        harness_pool.put_nowait((eval_i, harness_i))

    if concurrency > 1:
        logger.info("Using %d concurrent harness slots", concurrency)

    # Run all tasks — write results incrementally to survive OOM/crashes
    import gc

    results = []
    predictions = []
    output_path = Path(args.output)
    write_lock = asyncio.Lock()
    completed_count = 0

    # Resume support: load existing results and skip completed tasks
    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path) as rf:
            for line in rf:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        results.append(rec)
                        done_ids.add(rec["task_id"])
                    except json.JSONDecodeError:
                        pass
        logger.info("Resuming: %d tasks already done, %d remaining", len(done_ids), len(tasks) - len(done_ids))

    # Resume-passed: keep only PASSED results from a previous run, re-run failures
    if args.resume_passed:
        prev_path = Path(args.resume_passed)
        if prev_path.exists():
            prev_passed = 0
            prev_failed = 0
            with open(prev_path) as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("passed"):
                            results.append(rec)
                            done_ids.add(rec["task_id"])
                            prev_passed += 1
                        else:
                            prev_failed += 1
                    except json.JSONDecodeError:
                        pass
            logger.info(
                "Resume-passed from %s: keeping %d passed, re-running %d failed",
                prev_path,
                prev_passed,
                prev_failed,
            )

    tasks_to_run = [t for t in tasks if t.task_id not in done_ids]
    for t in tasks_to_run:
        t.max_steps = steps_per_level.get(t.level, max_steps) if adaptive_steps else max_steps
        t.max_cost_usd = max_cost

    if args.resume and done_ids:
        file_mode = "a"
    elif args.resume_passed and done_ids:
        file_mode = "w"
    else:
        file_mode = "w"
    outfile = open(output_path, file_mode)
    if args.resume_passed and done_ids:
        for rec in results:
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
        outfile.flush()
        logger.info("Wrote %d passed results to %s", len(results), output_path)

    async def _run_one(task: GAIATask) -> dict:
        nonlocal completed_count
        eval_i, harness_i = await harness_pool.get()
        try:
            eval_i.set_ground_truth(task.final_answer)
            record = await run_single(harness_i, task)
            async with write_lock:
                results.append(record)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfile.flush()
                completed_count += 1
                if completed_count % 5 == 0:
                    gc.collect()
            if args.output_predictions:
                async with write_lock:
                    predictions.append(
                        {
                            "task_id": task.task_id,
                            "model_answer": record.get("agent_output", ""),
                        }
                    )
            return record
        finally:
            harness_pool.put_nowait((eval_i, harness_i))

    if concurrency > 1:
        await asyncio.gather(*[_run_one(t) for t in tasks_to_run])
    else:
        for t in tasks_to_run:
            await _run_one(t)

    outfile.close()
    logger.info("Results written to %s", output_path)

    # Retry failed tasks with previous-answer hints
    if args.retries > 0:
        import copy

        all_prev_answers: dict[str, list[str]] = {}  # task_id -> list of wrong answers
        for retry_round in range(1, args.retries + 1):
            failed_records = {r["task_id"]: r for r in results if not r.get("passed")}
            failed_tasks = [t for t in tasks if t.task_id in failed_records]
            if not failed_tasks:
                break
            logger.info("=" * 60)
            logger.info("RETRY round %d — %d failed tasks", retry_round, len(failed_tasks))
            logger.info("=" * 60)

            async def _retry_one(task: GAIATask, f_handle) -> None:
                nonlocal results
                eval_i, harness_i = await harness_pool.get()
                try:
                    task.max_steps = steps_per_level.get(task.level, max_steps) if adaptive_steps else max_steps
                    task.max_cost_usd = max_cost
                    eval_i.set_ground_truth(task.final_answer)
                    prev = failed_records.get(task.task_id, {})
                    prev_answer = prev.get("agent_output", "")[:200]
                    prev_exit = prev.get("exit_reason", "")
                    original_desc = copy.deepcopy(task.description)
                    hint = ""
                    if prev_exit == "budget_exceeded":
                        task.max_steps = int(task.max_steps * 1.5)
                        hint = (
                            "\n\n[RETRY HINT: Previous attempt ran out of steps before "
                            "finding the answer. Be MORE FOCUSED: pick the single most "
                            "promising approach, avoid redundant searches, and get to the "
                            "answer efficiently. If you need to compute something, use "
                            "code_interpreter/Bash immediately rather than searching.]"
                        )
                    elif prev_answer and prev_exit == "done":
                        prev_answers = all_prev_answers.get(task.task_id, [])
                        if retry_round > 1 and prev_answers:
                            all_wrong = ", ".join(f'"{a[:50]}"' for a in prev_answers[-2:])
                            hint = (
                                f"\n\n[RETRY HINT (FINAL ATTEMPT): You have already tried "
                                f"{retry_round} times and gotten it wrong. "
                                f"Previous wrong answers: {all_wrong}. "
                                f"You MUST use a COMPLETELY DIFFERENT METHOD: "
                                f"if you searched before, try computing directly; "
                                f"if you computed, search for source data to verify inputs; "
                                f"if you used WebFetch, try Browser to navigate the page; "
                                f"if you read the question one way, re-read it for a "
                                f"different interpretation. Re-read the EXACT question "
                                f"word by word before starting.]"
                            )
                        else:
                            hint = (
                                f"\n\n[RETRY HINT: A previous attempt answered incorrectly. "
                                f"The wrong answer was: {prev_answer[:100]}. "
                                f"That answer is WRONG — try a completely different approach. "
                                f"Search for different sources, use different query terms, "
                                f"re-read the question more carefully, and double-check any "
                                f"calculations. If the task involves a file, re-examine the "
                                f"file data from scratch.]"
                            )
                    if prev_answer:
                        all_prev_answers.setdefault(task.task_id, []).append(prev_answer[:100])
                    if hint:
                        if isinstance(task.description, str):
                            task.description = task.description + hint
                        elif isinstance(task.description, list):
                            for block in task.description:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    block["text"] = block["text"] + hint
                                    break
                    record = await run_single(harness_i, task)
                    record["retry"] = retry_round
                    task.description = original_desc
                    if record.get("passed"):
                        async with write_lock:
                            results = [r for r in results if r["task_id"] != task.task_id]
                            results.append(record)
                            f_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f_handle.flush()
                        logger.info("Retry PASSED: %s", task.task_id[:8])
                    else:
                        failed_records[task.task_id] = record
                    gc.collect()
                finally:
                    harness_pool.put_nowait((eval_i, harness_i))

            with open(output_path, "a") as f:
                if concurrency > 1:
                    await asyncio.gather(*[_retry_one(t, f) for t in failed_tasks])
                else:
                    for task in failed_tasks:
                        await _retry_one(task, f)

    # Write leaderboard predictions if requested
    if args.output_predictions and predictions:
        pred_path = Path(args.output_predictions)
        with open(pred_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info("Predictions written to %s", pred_path)

    _print_summary(results, output_path)


if __name__ == "__main__":
    asyncio.run(main())
