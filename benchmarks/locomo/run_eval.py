"""run_eval.py — batch evaluation script for the LoCoMo benchmark.

Usage
-----
# Verbatim compressor, all categories, first 50 conversations:
python benchmarks/locomo/run_eval.py --compressor verbatim --max-samples 50

# Summary compressor, temporal reasoning only:
python benchmarks/locomo/run_eval.py --compressor summary --categories temporal_reasoning

# Fact compressor with a different model:
python benchmarks/locomo/run_eval.py --compressor facts --model gpt-4o-mini

# Light-memory (rule-based ingestion) with Sonnet 4.5 + extended thinking:
python benchmarks/locomo/run_eval.py \\
    --compressor light-memory \\
    --model YOUR_PROVIDER/claude-sonnet-4-5 \\
    --extended-thinking --thinking-budget 62976 \\
    --data-path benchmarks/locomo/locomo10.json \\
    --output reports/locomo_lm.jsonl --resume

Environment variables
---------------------
ANTHROPIC_API_KEY / OPENAI_API_KEY  — model credentials
ANTHROPIC_BASE_URL                  — use AnthropicProvider (proxy / ET)
MODEL                               — override --model flag

Optimisation note
-----------------
Tasks are grouped by sample_id so each conversation's sessions are ingested
only once, regardless of how many QA pairs that conversation contains.
EvalReadOnlyPolicy prevents QA answers from being written back to memory,
keeping the ingested baseline clean across all tasks in the same group.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.locomo.task import LoCoMoTask, LoCoMoEvaluator
from benchmarks.locomo.ingester import (
    SessionIngester,
    VerbatimCompressor,
    SummaryCompressor,
    FactCompressor,
    LightMemorySessionCompressor,
    LightMemoryLLMCompressor,
)
from benchmarks.locomo.harness import make_locomo_harness, LightMemoryBackend, _LOCOMO_SYSTEM
from harnessx.processors.memory.strategies.policy import EvalReadOnlyPolicy


COMPRESSOR_MAP = {
    "verbatim": lambda model: VerbatimCompressor(),
    "summary": lambda model: SummaryCompressor(model=model),
    "facts": lambda model: FactCompressor(model=model),
}


def _make_memory():
    from harnessx.processors.memory.strategies.custom import InMemoryMemory

    return InMemoryMemory(max_messages=5000)


def _make_call_llm(model: str, extra_headers: dict[str, str] | None = None):
    """Return an async LLMCallFn for light-memory LLM paths.

    Routes through the Anthropic proxy when ANTHROPIC_BASE_URL is set
    (matching QA-model routing), otherwise falls back to LiteLLM.
    """
    if os.environ.get("ANTHROPIC_BASE_URL"):
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic()

        async def call_llm(prompt: str) -> str | None:
            try:
                response = await _client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                return response.content[0].text.strip()
            except Exception:  # noqa: BLE001
                return None

        return call_llm

    async def call_llm(prompt: str) -> str | None:
        import litellm

        headers = {"X-Model-Provider-Id": "YOUR_PROVIDER_ID"}
        if extra_headers:
            headers.update(extra_headers)
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                extra_headers=headers,
            )
            return response.choices[0].message.content.strip()
        except Exception:  # noqa: BLE001
            return None

    return call_llm


def _make_qa_caller(model: str, extra_headers: dict[str, str] | None = None):
    """Return an async caller for QA that takes (system, user) messages — v7f style."""
    if os.environ.get("ANTHROPIC_BASE_URL"):
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic()

        async def qa_call(system: str, user: str) -> str:
            response = await _client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=4096,
            )
            return response.content[0].text.strip()

        return qa_call

    async def qa_call(system: str, user: str) -> str:
        import litellm

        headers = {"X-Model-Provider-Id": "YOUR_PROVIDER_ID"}
        if extra_headers:
            headers.update(extra_headers)
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
            extra_headers=headers,
        )
        return response.choices[0].message.content.strip()

    return qa_call


async def evaluate_sample(
    sample_id: str,
    tasks: list[LoCoMoTask],
    ingester: "SessionIngester | None",
    evaluator: LoCoMoEvaluator,
    model: str,
    comp_name: str,
    *,
    persist_dir: str = ".locomo_lm",
    extended_thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    top_k: int = 40,
    half_life_days: int = 365,
    call_llm=None,
    organization_enabled: bool = False,
) -> list[dict]:
    """Ingest once, then run all QA tasks for one conversation."""
    is_light_memory = comp_name == "light-memory"

    if is_light_memory:
        from datetime import datetime as _dt
        from harnessx.plugins.dimensions.light_memory._core.backend import ensure_memory_repo
        from harnessx.plugins.dimensions.light_memory._core.types import PluginConfig

        memory_root = os.path.join(persist_dir, f"lm_{sample_id}")
        cfg = PluginConfig(
            memory_root=memory_root,
            user_id="eval-user",
            top_k=top_k,
            access_half_life_days=half_life_days,
            organization_enabled=organization_enabled,
            organization_timeout_ms=60_000,
            decay_enabled=half_life_days < 9999,
            git_mode="disabled",
            auto_commit=False,
        )
        ensure_memory_repo(cfg)

        # Resume: skip ingestion if sessions already written (≥80% complete)
        sessions_dir = os.path.join(memory_root, "sessions", "user")
        expected_sessions = len(tasks[0].sessions)
        already_ingested = False
        if os.path.isdir(sessions_dir):
            ingested_ids: set[int] = set()
            for _root, _dirs, files in os.walk(sessions_dir):
                for f in files:
                    m = re.search(r"session-(\d+)-", f)
                    if m:
                        ingested_ids.add(int(m.group(1)))
            if len(ingested_ids) >= expected_sessions * 0.8:
                already_ingested = True
            elif ingested_ids:
                import shutil

                print(f"    WARNING: partial ingestion ({len(ingested_ids)}/{expected_sessions}), re-ingesting...")
                shutil.rmtree(memory_root, ignore_errors=True)
                ensure_memory_repo(cfg)

        if not already_ingested:
            if call_llm is not None:
                compressor = LightMemoryLLMCompressor(cfg, call_llm)
            else:
                compressor = LightMemorySessionCompressor(cfg)

            for sess in tasks[0].sessions:
                lm_ingester = SessionIngester(compressor=compressor, concurrency=1)
                await lm_ingester.ingest([sess], _NullMemory())

                if organization_enabled and call_llm is not None:
                    from datetime import timezone as _tz

                    from harnessx.plugins.dimensions.light_memory._core.lifecycle import (
                        perform_organization,
                    )

                    try:
                        sess_now = _dt.fromisoformat(sess.date + "T12:00:00+00:00")
                    except (ValueError, TypeError):
                        sess_now = _dt.now(_tz.utc)
                    try:
                        await perform_organization(cfg, call_llm, sess_now)
                    except Exception:  # noqa: BLE001
                        pass

        # Set eval_now to last session date for temporal accuracy
        memory = LightMemoryBackend(cfg, call_llm=call_llm)
        last_session = tasks[0].sessions[-1]
        try:
            memory.set_eval_now(_dt.fromisoformat(last_session.date + "T12:00:00+00:00"))
        except (ValueError, TypeError):
            pass
    else:
        memory = _make_memory()
        await ingester.ingest(tasks[0].sessions, memory)

    # For light-memory + LLM: bypass Harness, use v7f single-message format
    qa_caller = None
    if is_light_memory and call_llm is not None:
        qa_caller = _make_qa_caller(model)

    results: list[dict] = []
    for task in tasks:
        if qa_caller is not None:
            try:
                from harnessx.plugins.dimensions.light_memory._core.lifecycle import (
                    read_recalled_memories_with_llm,
                )

                recalled = await read_recalled_memories_with_llm(cfg, task.question, call_llm, now=memory._eval_now)
                if recalled and recalled.strip():
                    memory_context = f"Retrieved Memories:\n{recalled}"
                else:
                    memory_context = "(No relevant memories found)"
                user_message = f"{memory_context}\n\nQuestion: {task.question}\n\nAnswer:"
                prediction = await qa_caller(_LOCOMO_SYSTEM, user_message)
            except Exception as exc:  # noqa: BLE001
                prediction = f"ERROR: {exc}"
        else:
            policy = EvalReadOnlyPolicy()
            harness, _ = make_locomo_harness(
                model=model,
                memory=memory,
                memory_policy=policy,
                extended_thinking=extended_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
            )
            try:
                result = await harness.run(task)
                prediction = result.task_end.final_output or ""
            except Exception as exc:  # noqa: BLE001
                prediction = f"ERROR: {exc}"

        eval_result = evaluator.evaluate(prediction, task)
        results.append(
            {
                "sample_id": task.sample_id,
                "category": task.category,
                "question": task.question,
                "gold": task.gold_answer,
                "prediction": prediction,
                "score": eval_result.score,
                "f1": eval_result.f1,
                "exact_match": eval_result.exact_match,
                "rouge_l": eval_result.rouge_l,
                "n_sessions": len(task.sessions),
            }
        )

    return results


class _NullMemory:
    """No-op memory backend used when LightMemorySessionCompressor manages storage."""

    async def add(self, messages, **_kwargs):
        pass

    async def retrieve(self, query, k=10):
        return []

    async def compress(self, messages, budget):
        return messages

    async def persist(self):
        pass

    async def load(self, run_id):
        return []


def _load_completed(output_path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not output_path.exists():
        return completed
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if "sample_id" in rec and "question" in rec and "score" in rec:
                completed.add((rec["sample_id"], rec["question"]))
        except (json.JSONDecodeError, ValueError):
            continue
    return completed


def _append_jsonl(output_path: Path, records: list[dict]) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def run_eval(args: argparse.Namespace) -> None:
    model = os.environ.get("MODEL", args.model)
    categories = args.categories.split(",") if args.categories else None
    comp_names = (
        list({**COMPRESSOR_MAP, "light-memory": None}.keys()) if args.compressor == "all" else [args.compressor]
    )

    print(f"Loading LoCoMo dataset (split={args.split}, max_samples={args.max_samples})...")
    if args.data_path:
        tasks = LoCoMoTask.from_json_file(
            path=args.data_path,
            max_samples=args.max_samples,
            categories=categories,
        )
    else:
        tasks = LoCoMoTask.from_dataset(
            split=args.split,
            max_samples=args.max_samples,
            categories=categories,
        )
    print(f"  {len(tasks)} QA tasks from dataset")

    by_sample: dict[str, list[LoCoMoTask]] = defaultdict(list)
    for t in tasks:
        by_sample[t.sample_id].append(t)
    n_samples = len(by_sample)
    n_tasks = len(tasks)
    print(f"  {n_samples} unique conversations → ingest once per conversation")
    print(f"  Ingest calls: {n_samples} ({n_tasks // max(n_samples, 1)}× speedup for LLM compressors)\n")

    evaluator = LoCoMoEvaluator()
    all_results: dict[str, list[dict]] = {}

    memory_model = getattr(args, "memory_model", None) or model
    call_llm = _make_call_llm(memory_model) if getattr(args, "llm_memory", False) else None
    top_k = getattr(args, "top_k", 40)
    half_life_days = getattr(args, "half_life", 365)
    workers = getattr(args, "workers", 1)

    for comp_name in comp_names:
        lm_tag = (
            f" llm-memory={args.llm_memory} top_k={top_k} half_life={half_life_days}"
            if comp_name == "light-memory"
            else ""
        )
        print(f"=== Compressor: {comp_name} | Model: {model} | ET: {args.extended_thinking}{lm_tag} ===")

        is_light_memory = comp_name == "light-memory"
        if not is_light_memory:
            compressor = COMPRESSOR_MAP[comp_name](model)
            ingester = SessionIngester(compressor=compressor, concurrency=args.concurrency)
        else:
            ingester = None  # per-sample ingester created inside evaluate_sample

        # Resume support (JSONL only)
        completed_keys: set[tuple[str, str]] = set()
        use_jsonl = args.output and args.output.endswith(".jsonl")
        if args.resume and use_jsonl:
            completed_keys = _load_completed(Path(args.output))
            if completed_keys:
                print(f"  Resume: {len(completed_keys)} questions already done, skipping\n")

        results: list[dict] = []
        counter = {"done": 0}
        sem = asyncio.Semaphore(workers)
        file_lock = asyncio.Lock()

        async def run_one(sample_idx: int, sample_id: str, sample_tasks: list) -> None:
            remaining = (
                [t for t in sample_tasks if (t.sample_id, t.question) not in completed_keys]
                if completed_keys
                else sample_tasks
            )
            if not remaining:
                counter["done"] += len(sample_tasks)
                print(f"  [{sample_idx:3d}/{n_samples}] sample={sample_id[:8]} SKIPPED (all done)")
                return

            async with sem:
                try:
                    org_enabled = not getattr(args, "no_org", False) and call_llm is not None
                    sample_results = await evaluate_sample(
                        sample_id=sample_id,
                        tasks=remaining,
                        ingester=ingester,
                        evaluator=evaluator,
                        model=model,
                        comp_name=comp_name,
                        persist_dir=args.persist_dir,
                        extended_thinking=args.extended_thinking,
                        thinking_budget_tokens=args.thinking_budget,
                        top_k=top_k,
                        half_life_days=half_life_days,
                        call_llm=call_llm,
                        organization_enabled=org_enabled,
                    )

                    if getattr(args, "judge", False):
                        from benchmarks.locomo.judge import judge_accuracy

                        judge_model = getattr(args, "judge_model", "gpt-4o-mini")
                        judge_runs = getattr(args, "judge_runs", 3)
                        for rec in sample_results:
                            if "score" not in rec:
                                continue
                            try:
                                rec["llm_judge"] = await judge_accuracy(
                                    rec["question"],
                                    rec["gold"],
                                    rec["prediction"],
                                    model=judge_model,
                                    num_runs=judge_runs,
                                )
                            except Exception:  # noqa: BLE001
                                pass

                    counter["done"] += len(sample_tasks)
                    valid = [r for r in sample_results if "score" in r]
                    avg_score = sum(r["score"] for r in valid) / len(valid) if valid else 0.0
                    print(
                        f"  [{sample_idx:3d}/{n_samples}] sample={sample_id[:8]} "
                        f"qa={len(remaining):2d} avg_score={avg_score:.3f} "
                        f"[{counter['done']}/{n_tasks}]"
                    )
                    async with file_lock:
                        results.extend(sample_results)
                        if use_jsonl:
                            _append_jsonl(Path(args.output), sample_results)

                except Exception as exc:  # noqa: BLE001
                    error_records = [{"error": str(exc), "sample_id": t.sample_id} for t in remaining]
                    counter["done"] += len(sample_tasks)
                    print(f"  [{sample_idx:3d}/{n_samples}] sample={sample_id[:8]} ERROR: {exc}")
                    async with file_lock:
                        results.extend(error_records)
                        if use_jsonl:
                            _append_jsonl(Path(args.output), error_records)

        await asyncio.gather(*[run_one(idx, sid, stasks) for idx, (sid, stasks) in enumerate(by_sample.items(), 1)])

        valid = [r for r in results if "score" in r]
        agg = evaluator.aggregate([type("R", (), {"score": r["score"], "category": r["category"]})() for r in valid])
        print(f"\n  Aggregated scores ({comp_name}) — {len(valid)}/{n_tasks} valid:")
        for cat, score in agg.items():
            print(f"    {cat:<25} {score:.4f}")

        if getattr(args, "judge", False):
            from benchmarks.locomo.judge import compute_aligned_accuracy

            judged = [r for r in valid if r.get("llm_judge") is not None]
            if judged:
                aligned = compute_aligned_accuracy(judged, exclude_adversarial=True)
                print(f"\n  EverMemOS-aligned Accuracy (excluding adversarial, {len(judged)} judged):")
                for cat, acc in aligned.items():
                    print(f"    {cat:<25} {acc:.2f}%")
        print()

        all_results[comp_name] = results

    if args.output and not use_jsonl:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"Results written to {out_path}")
    elif use_jsonl:
        print(f"Results in {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark evaluation")
    parser.add_argument(
        "--compressor",
        choices=["verbatim", "summary", "facts", "light-memory", "all"],
        default="verbatim",
        help="Session compression strategy (default: verbatim)",
    )
    parser.add_argument(
        "--model",
        default="openai/pa/claude-haiku-4-5-20251001",
        help="Model for QA (and for LLM-based compressors)",
    )
    parser.add_argument("--split", default="test", help="Dataset split (HuggingFace only)")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to local JSON file; if set, skips HuggingFace download",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max conversations to evaluate")
    parser.add_argument("--categories", default=None, help="Comma-separated question categories")
    parser.add_argument("--concurrency", type=int, default=4, help="Max parallel LLM calls for compressors")
    parser.add_argument("--output", default=None, help="Results path (.jsonl for incremental/resume, .json for batch)")
    parser.add_argument(
        "--persist-dir", default=".locomo_lm", help="Directory for light-memory files per sample (default: .locomo_lm)"
    )
    # Extended Thinking
    parser.add_argument(
        "--extended-thinking", action="store_true", help="Enable extended thinking (AnthropicProvider only)"
    )
    parser.add_argument("--thinking-budget", type=int, default=8000, help="Thinking budget tokens (default: 8000)")
    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from existing .jsonl output")
    # Sample-level concurrency
    parser.add_argument("--workers", type=int, default=1, help="Parallel sample workers (default: 1)")
    # Light-memory tuning
    parser.add_argument(
        "--llm-memory", action="store_true", help="Use LLM for ingestion and retrieval (light-memory only)"
    )
    parser.add_argument(
        "--memory-model", default=None, help="Model for light-memory LLM paths (default: same as --model)"
    )
    parser.add_argument("--top-k", type=int, default=40, help="light-memory retrieval top_k (default: 40)")
    parser.add_argument(
        "--half-life",
        type=int,
        default=365,
        help="light-memory decay half-life in days (default: 365, use 9999 to disable)",
    )
    parser.add_argument(
        "--no-org",
        action="store_true",
        help="Disable organization pass (default: enabled when --llm-memory is set)",
    )
    # LLM Judge
    parser.add_argument("--judge", action="store_true", help="Enable EverMemOS-aligned LLM-as-Judge accuracy scoring")
    parser.add_argument(
        "--judge-model", default="azure_openai/gpt-4o-mini", help="Judge model (default: azure_openai/gpt-4o-mini)"
    )
    parser.add_argument("--judge-runs", type=int, default=3, help="Independent judge runs per question (default: 3)")
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
