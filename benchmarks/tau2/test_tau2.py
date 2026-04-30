#!/usr/bin/env python
"""
tau2-bench + HarnessX test script

Usage:
    # Run inside tau2-bench's venv (after installing harnessx):
    #   cd ~/tau2-bench && uv run python /path/to/harnessx/benchmarks/tau2/test_tau2.py
    #
    # Or directly (if tau2 and harnessx are both in the current Python environment):
    #   python benchmarks/tau2/test_tau2.py

    # Customization:
    #   python benchmarks/tau2/test_tau2.py --domain airline --num-tasks 5
    #   python benchmarks/tau2/test_tau2.py --agent-model openai/gpt-4.1 --api-base https://api.openai.com/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any

# ─── Output utilities ────────────────────────────────────────────────────────

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

_passed = 0
_failed = 0
_skipped = 0


def info(msg: str) -> None:
    print(f"{GREEN}[INFO]{NC} {msg}")


def step_pass(name: str, detail: str = "") -> None:
    global _passed
    _passed += 1
    suffix = f" ({detail})" if detail else ""
    print(f"  {GREEN}PASS{NC} {name}{suffix}")


def step_fail(name: str, detail: str = "") -> None:
    global _failed
    _failed += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  {RED}FAIL{NC} {name}{suffix}")


def step_skip(name: str, reason: str = "") -> None:
    global _skipped
    _skipped += 1
    suffix = f" — {reason}" if reason else ""
    print(f"  {YELLOW}SKIP{NC} {name}{suffix}")


# ─── Test steps ──────────────────────────────────────────────────────────────


def test_imports() -> bool:
    """Step 1: full-stack import test"""
    info("=== Step 1: import test ===")
    try:
        from tau2.data_model.message import (  # noqa: F401
            AssistantMessage,
            UserMessage,
            ToolMessage,
            SystemMessage,
            ToolCall,
            MultiToolMessage,
        )
        from tau2.agent.base_agent import HalfDuplexAgent  # noqa: F401
        from tau2.runner.helpers import get_tasks  # noqa: F401
        from tau2.data_model.simulation import TextRunConfig  # noqa: F401
        from tau2.evaluator.evaluator import EvaluationType  # noqa: F401

        from harnessx.core.events import (  # noqa: F401
            Message,
            ToolSchema,
            EvalResult,
            ModelResponseEvent,
        )
        from harnessx.providers.anthropic_provider import AnthropicProvider  # noqa: F401

        from benchmarks.tau2 import (  # noqa: F401
            HarnessXAgent,
            Tau2Task,
            Tau2Evaluator,
            create_harnessx_agent,
            make_tau2_harness,
        )
        from harnessx.core.builder import HarnessBuilder  # noqa: F401
        from harnessx.core.harness import HarnessConfig  # noqa: F401
        from harnessx.processors.context.system_prompt import SystemPromptProcessor  # noqa: F401
        from harnessx.processors.control.token_budget import TokenBudgetProcessor  # noqa: F401

        step_pass("tau2 + harnessx + recipe full-stack import (including HarnessBuilder)")
        return True
    except Exception as e:
        step_fail("import", str(e))
        return False


def test_message_conversion() -> bool:
    """Step 2: message format conversion (tau2 → OH)"""
    info("=== Step 2: message conversion test ===")
    try:
        from benchmarks.tau2.agent import _tau2_messages_to_oh
        from tau2.data_model.message import (
            SystemMessage,
            UserMessage,
            AssistantMessage,
            ToolMessage,
            MultiToolMessage,
            ToolCall,
        )
        from harnessx.core.events import ToolCall as OHToolCall

        # tau2 -> OH: basic types
        oh = _tau2_messages_to_oh(
            [
                SystemMessage(role="system", content="Hi"),
                UserMessage.text(content="Q"),
                AssistantMessage.text(
                    content=None,
                    tool_calls=[
                        ToolCall(id="c1", name="fn", arguments={"a": 1}),
                    ],
                ),
                ToolMessage(id="c1", role="tool", content="ok"),
                AssistantMessage.text(content="Done"),
            ]
        )
        assert len(oh) == 5, f"expected 5, got {len(oh)}"
        assert oh[0].role == "system"
        assert oh[2].tool_calls[0].input == {"a": 1}
        assert oh[3].tool_call_id == "c1"

        # MultiToolMessage expansion
        multi_oh = _tau2_messages_to_oh(
            [
                MultiToolMessage(
                    role="tool",
                    tool_messages=[
                        ToolMessage(id="c2", role="tool", content="a"),
                        ToolMessage(id="c3", role="tool", content="b"),
                    ],
                )
            ]
        )
        assert len(multi_oh) == 2
        assert multi_oh[0].tool_call_id == "c2"
        assert multi_oh[1].tool_call_id == "c3"

        # content=None handling
        none_oh = _tau2_messages_to_oh([UserMessage.text(content=None)])
        assert none_oh[0].content == ""

        # OH state → tau2 AssistantMessage (new arch: extract last assistant message from oh_state.messages)
        from harnessx.core.events import Message
        from harnessx.core.state import State
        from harnessx.core.events import make_run_id
        from tau2.data_model.message import ToolCall as Tau2TC

        oh_state = State(run_id=make_run_id())
        oh_state.add_message(Message(role="user", content="help"))
        # Simulate run_loop adding an assistant message with tool calls
        oh_state.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=(OHToolCall(id="x", name="fn", input={"a": 1}),),
            )
        )
        last_ast = next(m for m in reversed(oh_state.messages) if m.role == "assistant")
        tau2_tc = [Tau2TC(id=tc.id, name=tc.name, arguments=tc.input) for tc in last_ast.tool_calls]
        assert len(tau2_tc) == 1
        assert tau2_tc[0].arguments == {"a": 1}
        # content=None when tool_calls present (tau2 convention)
        assistant_msg = AssistantMessage.text(content=None, tool_calls=tau2_tc)
        assert assistant_msg.content is None
        assert len(assistant_msg.tool_calls) == 1

        step_pass("tau2→OH conversion (5 types + MultiTool expansion + None) + OH state→tau2 extraction")
        return True
    except Exception as e:
        step_fail("message conversion", str(e))
        return False


def test_task_loading() -> bool:
    """Step 3: task loading across domains"""
    info("=== Step 3: task loading test ===")
    try:
        from benchmarks.tau2 import Tau2Task

        counts = {}
        for domain in ["mock", "airline", "retail", "telecom"]:
            ids = Tau2Task.list_tasks(domain)
            counts[domain] = len(ids)
            print(f"    {domain}: {len(ids)} tasks")

        task = Tau2Task.from_domain("mock", task_id="create_task_1")
        assert task.tau2_task_id == "create_task_1"
        assert task.domain == "mock"

        detail = ", ".join(f"{d}={n}" for d, n in counts.items())
        step_pass("task loading per domain", detail)
        return True
    except Exception as e:
        step_fail("task loading", str(e))
        return False


def test_agent_registration() -> bool:
    """Step 4: Agent registration & initialization"""
    info("=== Step 4: Agent registration & initialization ===")
    try:
        from tau2.registry import registry
        from tau2.runner.build import build_environment, build_agent
        from tau2.runner.helpers import get_tasks
        from tau2.agent.base_agent import HalfDuplexAgent
        from benchmarks.tau2.agent import create_harnessx_agent, HarnessXAgent

        registry.register_agent_factory(create_harnessx_agent, "harnessx")

        env = build_environment("mock")
        tasks = get_tasks("mock", num_tasks=1)
        agent = build_agent(
            "harnessx",
            env,
            llm="test-model",
            llm_args={},
            task=tasks[0],
        )

        assert isinstance(agent, HarnessXAgent), "not HarnessXAgent"
        assert issubclass(HarnessXAgent, HalfDuplexAgent), "not subclass"

        state = agent.get_init_state()
        assert len(state.system_messages) == 1
        assert "policy" in state.system_messages[0].content.lower()

        step_pass("registration + build_agent + isinstance + state init")
        return True
    except Exception as e:
        step_fail("agent test", str(e))
        return False


def test_harness_pipeline() -> bool:
    """Step 5: HarnessBuilder + Harness.run() pipeline"""
    info("=== Step 5: Harness pipeline test ===")
    try:
        from benchmarks.tau2.harness import make_tau2_harness, make_tau2_provider
        from harnessx.core.harness import HarnessConfig, Harness
        from harnessx.processors.context.system_prompt import SystemPromptProcessor
        from harnessx.processors.control.token_budget import TokenBudgetProcessor
        from harnessx.providers.anthropic_provider import AnthropicProvider
        from harnessx.tracing.journal import HarnessJournal

        # Build HarnessConfig via HarnessBuilder
        config = make_tau2_harness(
            model="test-model",
            api_base="http://localhost:1234",
        )

        # Verify provider via make_tau2_provider
        from harnessx.providers.litellm_provider import LiteLLMProvider

        provider = make_tau2_provider(model="test-model", api_base="http://localhost:1234")
        assert isinstance(provider, (AnthropicProvider, LiteLLMProvider)), (
            f"provider is unexpected type: {type(provider)}"
        )

        # Verify config structure (no model_provider — moved to ModelConfig)
        assert isinstance(config, HarnessConfig), "not HarnessConfig"
        assert isinstance(config.tracer, HarnessJournal), "tracer is not HarnessJournal"
        assert config.processors, "no processors registered"

        # Verify SystemPromptProcessor + TokenBudgetProcessor are in the pipeline
        all_procs = list(getattr(config, "_rt_procs", None) or [])
        assert any(isinstance(p, SystemPromptProcessor) for p in all_procs), (
            "SystemPromptProcessor not found in pipeline"
        )
        assert any(isinstance(p, TokenBudgetProcessor) for p in all_procs), "TokenBudgetProcessor not found in pipeline"

        # Verify agent holds a Harness instance (not just raw provider)
        from tau2.runner.build import build_environment, build_agent
        from tau2.runner.helpers import get_tasks
        from benchmarks.tau2.agent import create_harnessx_agent, HarnessXAgent
        from tau2.registry import registry

        try:
            registry.register_agent_factory(create_harnessx_agent, "harnessx")
        except (ValueError, KeyError):
            pass

        env = build_environment("mock")
        tasks = get_tasks("mock", num_tasks=1)
        agent = build_agent(
            "harnessx",
            env,
            llm="test-model",
            llm_args={"api_base": "http://localhost:1234"},
            task=tasks[0],
        )

        assert isinstance(agent, HarnessXAgent), "not HarnessXAgent"
        assert hasattr(agent, "_harness"), "agent has no _harness"
        assert isinstance(agent._harness, Harness), "agent._harness is not Harness"
        assert hasattr(agent, "_config"), "agent has no _config (HarnessConfig)"
        assert isinstance(agent._config, HarnessConfig), "agent._config is not HarnessConfig"
        assert agent._tool_names, "agent._tool_names is empty (interrupt_on needs tool names)"

        # Verify interrupt_on would be populated correctly
        state = agent.get_init_state()
        assert state.oh_state is not None, "AgentState.oh_state is None (OH State not created)"
        # System message should be pre-loaded
        sys_msgs = [m for m in state.oh_state.messages if m.role == "system"]
        assert sys_msgs, "no system message in OH State"
        assert "policy" in sys_msgs[0].content.lower(), "system message missing policy"

        step_pass("HarnessBuilder + Harness.run() + interrupt_on + AgentState.oh_state")
        return True
    except Exception as e:
        step_fail("Harness pipeline", str(e))
        return False


def test_llm_api(model: str, api_base: str) -> bool:
    """Step 6: LLM API connectivity"""
    info("=== Step 6: LLM API connectivity ===")
    try:
        import litellm

        extra_kw: dict = {}
        resp = litellm.completion(
            model=model,
            api_base=api_base,
            messages=[{"role": "user", "content": "Say OK in one word"}],
            max_tokens=10,
            **extra_kw,
        )
        content = (resp.choices[0].message.content or "").strip()[:30]
        step_pass("LLM API reachable", f"response: {content}")
        return True
    except Exception as e:
        step_fail("LLM API", str(e)[:100])
        return False


def _build_task_record(task_id: str, result: Any, elapsed: float) -> dict:
    """Serialize one tau2 simulation result into a JSON-serialisable dict."""
    ri = result.reward_info
    record: dict = {
        "task_id": task_id,
        "reward": ri.reward if ri else 0.0,
        "passed": (ri.reward == 1.0) if ri else False,
        "termination_reason": result.termination_reason.value if result.termination_reason else None,
        "elapsed_s": round(elapsed, 2),
        "num_messages": len(result.messages) if result.messages else 0,
    }
    if ri:
        if ri.db_check is not None:
            record["db_check"] = {
                "passed": ri.db_check.db_match,
                "reward": ri.db_check.db_reward,
            }
        if ri.action_checks:
            record["action_checks"] = [
                {
                    "tool": ac.action.name if hasattr(ac.action, "name") else str(ac.action),
                    "passed": ac.action_match,
                    "reward": ac.action_reward,
                    "tool_type": ac.tool_type.value if ac.tool_type else None,
                }
                for ac in ri.action_checks
            ]
        if ri.communicate_checks:
            record["communicate_checks"] = [
                {
                    "info": cc.info,
                    "passed": cc.met,
                    "justification": cc.justification,
                }
                for cc in ri.communicate_checks
            ]
        if ri.nl_assertions:
            record["nl_assertions"] = [
                {
                    "assertion": nc.nl_assertion,
                    "passed": nc.met,
                    "justification": nc.justification,
                }
                for nc in ri.nl_assertions
            ]
        if ri.reward_breakdown:
            record["reward_breakdown"] = {k.value: v for k, v in ri.reward_breakdown.items()}
    return record


def test_simulation(
    domain: str,
    num_tasks: int,
    agent_model: str,
    user_model: str,
    api_base: str,
    agent_api_base: str = "",
    judge_api_base: str = "",
    judge_model: str = "",
    report_path: str = "",
    policy_hints: bool = False,
    harness_config: str = "harness_config.yaml",
    task_ids: list[str] | None = None,
    num_trials: int = 1,
    extended_thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    max_concurrency: int = 1,
    stop_guard: bool = False,
    auto_resume: bool = False,
    logs_dir: str = "runs",
) -> bool:
    """Step 7: end-to-end simulation"""
    info(f"=== Step 7: end-to-end simulation ({domain} x {num_tasks}) ===")
    try:
        from tau2.registry import registry
        from benchmarks.tau2.agent import create_harnessx_agent
        from tau2.data_model.simulation import TextRunConfig
        from tau2.runner.batch import run_single_task, run_tasks
        from tau2.runner.helpers import get_tasks
        from tau2.evaluator.evaluator import EvaluationType

        # ensure registration (idempotent)
        try:
            registry.register_agent_factory(create_harnessx_agent, "harnessx")
        except (ValueError, KeyError):
            pass

        # register stop-guard user simulator (strips premature ###STOP### from
        # "Yes, please proceed.###STOP###" messages before the orchestrator sees them)
        user_simulator_name = "user_simulator"
        if stop_guard:
            from benchmarks.tau2.stop_guard import StopGuardUserSimulator

            try:
                registry.register_user(StopGuardUserSimulator, "harnessx_stop_guard")
            except (ValueError, KeyError):
                pass
            user_simulator_name = "harnessx_stop_guard"

        # Point the NL assertions judge at the agent's gpt-4.1 endpoint;
        # otherwise tau2 defaults to gpt-4.1-2025-04-14 (requires an OpenAI key).
        import tau2.evaluator.evaluator_nl_assertions as _nl_eval

        _nl_eval.DEFAULT_LLM_NL_ASSERTIONS = judge_model or user_model
        _nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
            "temperature": 0.0,
            "api_base": judge_api_base or api_base,
        }

        # agent may use a different api_base (e.g. Anthropic endpoint for extended thinking)
        # user simulator always uses api_base (OpenAI-compatible endpoint)
        effective_agent_api_base = agent_api_base or api_base
        agent_llm_args: dict = {"api_base": effective_agent_api_base}
        user_llm_args: dict = {"api_base": api_base}
        if policy_hints:
            agent_llm_args["policy_hints"] = True
        if harness_config != "harness_config.yaml":
            agent_llm_args["harness_config"] = harness_config
        if extended_thinking:
            agent_llm_args["extended_thinking"] = True
            agent_llm_args["thinking_budget_tokens"] = thinking_budget_tokens
        if logs_dir != "runs":
            agent_llm_args["logs_dir"] = logs_dir

        if task_ids:
            tasks = get_tasks(domain, task_ids=task_ids)
        else:
            tasks = get_tasks(domain, num_tasks=num_tasks)
        config = TextRunConfig(
            domain=domain,
            agent="harnessx",
            user=user_simulator_name,
            llm_agent=agent_model,
            llm_args_agent=agent_llm_args,
            llm_user=user_model,
            llm_args_user=user_llm_args,
            num_trials=num_trials,
            max_steps=200,
            max_concurrency=max_concurrency,
            auto_resume=auto_resume,
        )

        total_reward = 0.0
        task_records: list[dict] = []

        if max_concurrency > 1:
            # ── parallel path via run_tasks ───────────────────────────────────
            import pathlib

            # When not resuming, remove any existing report so run_tasks
            # does not prompt interactively "Do you want to resume?" in a
            # non-tty environment.
            if not auto_resume and report_path:
                _existing = pathlib.Path(report_path)
                if _existing.exists():
                    _existing.unlink()

            batch_results = run_tasks(
                config,
                tasks,
                save_path=pathlib.Path(report_path) if report_path else None,
                evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
                console_display=True,
            )
            # group simulations by task_id; aggregate across trials
            from collections import defaultdict

            sims_by_task: dict = defaultdict(list)
            for sim in batch_results.simulations:
                sims_by_task[sim.task_id].append(sim)

            for task in tasks:
                sims = sims_by_task.get(task.id, [])
                if not sims:
                    task_records.append(
                        {
                            "task_id": task.id,
                            "reward": 0.0,
                            "passed": False,
                            "termination_reason": "error",
                            "elapsed_s": 0.0,
                            "num_messages": 0,
                            "error": "no simulation result",
                        }
                    )
                    continue

                rewards = [s.reward_info.reward if s.reward_info else 0.0 for s in sims]
                avg_reward = sum(rewards) / len(rewards)
                passed = all(r == 1.0 for r in rewards)
                total_duration = sum(s.duration for s in sims)
                num_msgs = sum(len(s.messages) for s in sims if s.messages)
                last_sim = sims[-1]
                term = last_sim.termination_reason.value if last_sim.termination_reason else None

                record: dict = {
                    "task_id": task.id,
                    "reward": round(avg_reward, 4),
                    "passed": passed,
                    "termination_reason": term,
                    "elapsed_s": round(total_duration, 2),
                    "num_messages": num_msgs,
                    "num_trials": len(sims),
                    "trial_rewards": rewards,
                }
                task_records.append(record)
                total_reward += avg_reward

                status = f"{GREEN}PASS{NC}" if passed else f"{RED}FAIL{NC}"
                parts = [
                    f"reward={avg_reward:.3f}",
                    f"trials={len(sims)}",
                    f"term={term}",
                    f"msgs={num_msgs}",
                    f"{total_duration:.1f}s",
                ]
                print(f"    [{status}] {task.id}: {', '.join(parts)}")
        else:
            # ── serial path ───────────────────────────────────────────────────
            for task in tasks:
                t0 = time.time()
                try:
                    result = run_single_task(
                        config,
                        task,
                        seed=42,
                        evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
                    )
                except Exception as exc:
                    elapsed = time.time() - t0
                    print(
                        f"    [{RED}ERROR{NC}] {task.id}: evaluation crashed ({exc.__class__.__name__}: {exc}), skipping"
                    )
                    task_records.append(
                        {
                            "task_id": task.id,
                            "reward": 0.0,
                            "passed": False,
                            "termination_reason": "error",
                            "elapsed_s": elapsed,
                            "num_messages": 0,
                            "error": str(exc),
                        }
                    )
                    continue
                elapsed = time.time() - t0

                reward = result.reward_info.reward if result.reward_info else 0.0
                total_reward += reward

                record = _build_task_record(task.id, result, elapsed)
                task_records.append(record)

                # ── per-task inline summary ───────────────────────────────────────
                parts = [f"reward={reward:.3f}"]
                status = f"{GREEN}PASS{NC}" if record["passed"] else f"{RED}FAIL{NC}"
                parts.append(f"term={record['termination_reason']}")
                if "db_check" in record:
                    parts.append(f"db={'pass' if record['db_check']['passed'] else 'fail'}")
                if "action_checks" in record:
                    p = sum(1 for c in record["action_checks"] if c["passed"])
                    parts.append(f"actions={p}/{len(record['action_checks'])}")
                parts.append(f"msgs={record['num_messages']}")
                parts.append(f"{elapsed:.1f}s")
                print(f"    [{status}] {task.id}: {', '.join(parts)}")

                # append to report incrementally (useful for long-running evaluations)
                if report_path:
                    _flush_report(
                        report_path,
                        domain,
                        agent_model,
                        task_records,
                        policy_hints=policy_hints,
                        num_trials=num_trials,
                        extended_thinking=extended_thinking,
                    )

        avg = total_reward / len(tasks) if tasks else 0
        n_passed = sum(1 for r in task_records if r["passed"])
        n_total = len(task_records)

        if report_path and max_concurrency == 1:
            _flush_report(
                report_path,
                domain,
                agent_model,
                task_records,
                final=True,
                policy_hints=policy_hints,
                num_trials=num_trials,
                extended_thinking=extended_thinking,
            )
            print(f"  report saved: {report_path}")
        elif report_path:
            print(f"  report saved: {report_path}")

        step_pass(
            f"{n_total} {domain} tasks",
            f"passed={n_passed}/{n_total}  avg_reward={avg:.3f}",
        )
        return True
    except Exception as e:
        step_fail("simulation", str(e)[:200])
        traceback.print_exc()
        return False


def _flush_report(
    path: str,
    domain: str,
    agent_model: str,
    records: list[dict],
    final: bool = False,
    policy_hints: bool = False,
    num_trials: int = 1,
    extended_thinking: bool = False,
) -> None:
    """Write/overwrite the JSON report file."""
    n_passed = sum(1 for r in records if r["passed"])
    n_total = len(records)
    avg_reward = sum(r["reward"] for r in records) / n_total if n_total else 0.0

    report = {
        "meta": {
            "domain": domain,
            "agent_model": agent_model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "complete" if final else "in_progress",
            "policy_hints": policy_hints,
            "num_trials": num_trials,
            "extended_thinking": extended_thinking,
        },
        "summary": {
            "total": n_total,
            "passed": n_passed,
            "failed": n_total - n_passed,
            "pass_rate": round(n_passed / n_total, 4) if n_total else 0.0,
            "avg_reward": round(avg_reward, 4),
        },
        "tasks": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="tau2-bench recipe test")
    parser.add_argument("--domain", default=os.environ.get("TAU2_DOMAIN", "mock"))
    parser.add_argument("--num-tasks", type=int, default=int(os.environ.get("TAU2_NUM_TASKS", "2")))
    parser.add_argument(
        "--agent-model",
        default=os.environ.get("TAU2_AGENT_MODEL", "anthropic/claude-haiku-4-5-20251001"),
    )
    parser.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL", "openai/gpt-4.1"))
    parser.add_argument(
        "--api-base",
        default=os.environ.get("TAU2_API_BASE", os.environ.get("OPENAI_API_BASE", "")),
    )
    parser.add_argument(
        "--agent-api-base",
        default=os.environ.get("TAU2_AGENT_API_BASE", ""),
        help="Dedicated API endpoint for the agent model (e.g. Anthropic endpoint for extended thinking; defaults to --api-base)",
    )
    parser.add_argument(
        "--judge-api-base",
        default=os.environ.get("TAU2_JUDGE_API_BASE", ""),
        help="API endpoint for the NL assertions judge (defaults to --api-base)",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("TAU2_JUDGE_MODEL", ""),
        help="LLM model for NL assertions judge (defaults to --user-model)",
    )
    parser.add_argument(
        "--harness-config",
        default=os.environ.get("TAU2_HARNESS_CONFIG", "harness_config.yaml"),
        help="YAML pipeline config filename inside benchmarks/tau2/ (default: harness_config.yaml)",
    )
    parser.add_argument("--skip-llm", action="store_true", help="skip LLM-related tests (Steps 6-7)")
    parser.add_argument(
        "--policy-hints",
        action="store_true",
        help="enable PolicyHintProcessor: scan tool results each step and append reminders to system prompt for unresolved policy requirements",
    )
    parser.add_argument(
        "--extended-thinking",
        action="store_true",
        help="enable Extended Thinking (AnthropicProvider / anthropic/ prefix models only)",
    )
    parser.add_argument(
        "--thinking-budget-tokens",
        type=int,
        default=8000,
        help="Extended Thinking token budget (default: 8000)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="number of times to repeat each task; reward is averaged (pass^k needs k>=4; default: 1)",
    )
    parser.add_argument(
        "--report",
        default="",
        metavar="PATH",
        help="save evaluation report to a JSON file (e.g. reports/airline.json)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=[],
        metavar="TASK_ID",
        help="explicit list of task IDs to run (overrides --num-tasks)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("TAU2_WORKERS", "1")),
        help="number of concurrent workers for parallel simulation (default: 1 = serial)",
    )
    parser.add_argument(
        "--stop-guard",
        action="store_true",
        default=os.environ.get("TAU2_STOP_GUARD", "").lower() in ("1", "true", "yes"),
        help="strip premature ###STOP### from user confirmation messages (fixes ~80%% of retail/airline false negatives caused by GPT user simulator stopping before agent executes the write tool)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=os.environ.get("TAU2_RESUME", "").lower() in ("1", "true", "yes"),
        help="resume from previous run checkpoint (tau2 auto_resume); default is False (always start fresh)",
    )
    parser.add_argument(
        "--logs-dir",
        default=os.environ.get("TAU2_LOGS_DIR", "runs"),
        metavar="DIR",
        help="directory for HarnessJournal JSONL trajectory logs (default: runs, env: TAU2_LOGS_DIR)",
    )
    args = parser.parse_args()

    # ── suppress litellm retry noise ──────────────────────────────────────────
    # tau2's user simulator strips the "anthropic/" prefix when calling the LLM,
    # causing litellm to raise BadRequestError before tau2 retries successfully —
    # this is tau2 internal behavior and does not affect evaluation results.
    # Suppress three sources:
    #   1. litellm direct print() to stderr: "Provider List: ..." messages
    #   2. litellm/Python logging ERROR-level logs
    #   3. tau2/utils/llm_utils loguru ERROR-level logs
    import litellm as _litellm
    import logging as _logging

    _litellm.suppress_debug_info = True
    _litellm.set_verbose = False
    _logging.getLogger("LiteLLM").setLevel(_logging.CRITICAL)
    _logging.getLogger("litellm").setLevel(_logging.CRITICAL)
    try:
        from loguru import logger as _loguru

        _loguru.disable("tau2.utils.llm_utils")  # suppress tau2 internal LLM retry ERROR logs
    except ImportError:
        pass
    import warnings as _warnings

    _warnings.filterwarnings("ignore", category=RuntimeWarning, module="litellm")
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n  agent_model:       {args.agent_model}")
    print(f"  user_model:        {args.user_model}")
    print(f"  api_base:          {args.api_base}")
    if args.agent_api_base:
        print(f"  agent_api_base:    {args.agent_api_base}")
    print(f"  domain:            {args.domain} ({args.num_tasks} tasks)")
    print(f"  policy_hints:      {args.policy_hints}")
    print(f"  extended_thinking: {args.extended_thinking}")
    print(f"  num_trials:        {args.num_trials}")
    print(f"  workers:           {args.workers}")
    print(f"  stop_guard:        {args.stop_guard}")
    if args.report:
        print(f"  report:            {args.report}")
    print(f"  logs_dir:          {args.logs_dir}")
    print()

    # Steps 1-5: no LLM required
    test_imports()
    test_message_conversion()
    test_task_loading()
    test_agent_registration()
    test_harness_pipeline()

    # Steps 6-7: require LLM
    if args.skip_llm:
        step_skip("LLM API", "--skip-llm")
        step_skip("end-to-end simulation", "--skip-llm")
    else:
        llm_ok = test_llm_api(args.agent_model, args.agent_api_base or args.api_base)
        if llm_ok:
            test_simulation(
                domain=args.domain,
                num_tasks=args.num_tasks,
                agent_model=args.agent_model,
                user_model=args.user_model,
                api_base=args.api_base,
                agent_api_base=args.agent_api_base,
                judge_api_base=args.judge_api_base,
                judge_model=args.judge_model,
                report_path=args.report,
                policy_hints=args.policy_hints,
                harness_config=args.harness_config,
                task_ids=args.task_ids or None,
                num_trials=args.num_trials,
                extended_thinking=args.extended_thinking,
                thinking_budget_tokens=args.thinking_budget_tokens,
                max_concurrency=args.workers,
                stop_guard=args.stop_guard,
                auto_resume=args.resume,
                logs_dir=args.logs_dir,
            )
        else:
            step_skip("end-to-end simulation", "LLM API unavailable")

    # summary
    total = _passed + _failed + _skipped
    print(f"\n{'=' * 45}")
    print(
        f"  results: {GREEN}{_passed} passed{NC}, {RED}{_failed} failed{NC}, {YELLOW}{_skipped} skipped{NC}  (total {total})"
    )
    print(f"{'=' * 45}\n")

    if _failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
