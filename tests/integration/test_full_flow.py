# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import asyncio
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fixtures.mock_provider import MockProvider
from fixtures.mock_tools import add_tool, echo_tool, make_registry

from harnessx import (
    Harness,
    BaseTask,
    HarnessConfig,
    on_step_end,
    SegmentBoundaryEvent,
    ModelConfig,
)
from harnessx.core.events import Message, StepEndEvent
from harnessx.core.state import State
from harnessx.processors.control.compaction import CompactionProcessor
from harnessx.tracing.journal import HarnessJournal
from harnessx.tracing.null_tracer import NullTracer

MinimalConfig = HarnessConfig(processors=[])


async def _module_level_custom_tool_fn(x: str) -> str:
    """Module-level fn used by test_custom_tools_recorded_in_config — needs a stable
    importable qualname so it can round-trip via tools.custom in harness_config.yaml.
    """
    return x


def make_config(responses=None, tools=None, processors=None):
    return HarnessConfig(
        tool_registry=make_registry(*(tools or [])),
        tracer=NullTracer(),
        processors=processors or [],
    )


def make_harness(responses=None, tools=None, processors=None, extra_processors=None):
    mc = ModelConfig(main=MockProvider(responses=responses or ["Answer: 4"]))
    config = make_config(responses, tools, processors)
    if extra_processors:
        return Harness(mc, config, extra_processors=extra_processors)
    return mc.agentic(config)


@pytest.mark.asyncio
async def test_basic_runloop_completes():
    """RunLoop should complete and return TaskEndEvent."""
    harness = make_harness(responses=["The answer is 4."])
    result = await harness.run(BaseTask(description="What is 2+2?"))
    assert result is not None
    assert result.final_output == "The answer is 4."
    assert result.exit_reason == "done"


@pytest.mark.asyncio
async def test_tool_call_executed():
    """Tool calls in model response should be executed."""
    responses = [
        {
            "content": "Let me add those.",
            "tool_calls": [{"id": "c1", "name": "add", "input": {"a": 2, "b": 2}}],
        },
        "The answer is 4.",
    ]
    harness = make_harness(responses=responses, tools=[add_tool])
    result = await harness.run(BaseTask(description="What is 2+2?", max_steps=10))
    assert result.exit_reason in ("done", "budget_exceeded")


@pytest.mark.asyncio
async def test_custom_hook_injected():
    """Custom @on_step_end hook should be called."""
    steps_recorded = []

    @on_step_end
    async def record_step(event: StepEndEvent):
        steps_recorded.append(event.step_id)
        yield event

    harness = make_harness(
        responses=["Done."], extra_processors={"step_end": [record_step]}
    )  # extra_processors uses hook-keyed dict (runtime API)
    _result = await harness.run(BaseTask(description="test"))
    assert len(steps_recorded) > 0


@pytest.mark.asyncio
async def test_processor_removal_does_not_crash():
    """Removing LoopDetectionProcessor should not affect correctness."""
    harness = make_harness(
        responses=["Done."],
        processors=[],  # No processors at all
    )
    result = await harness.run(BaseTask(description="test"))
    assert result.exit_reason == "done"


@pytest.mark.asyncio
async def test_multiple_processors_in_chain():
    """Multiple step_end processors should all run."""
    log = []

    @on_step_end
    async def proc1(event: StepEndEvent):
        log.append("proc1")
        yield event

    @on_step_end
    async def proc2(event: StepEndEvent):
        log.append("proc2")
        yield event

    harness = make_harness(
        responses=["Done."],
        processors=[proc1, proc2],
    )
    await harness.run(BaseTask(description="test"))
    assert "proc1" in log
    assert "proc2" in log


@pytest.mark.asyncio
async def test_trace_jsonl_produced(tmp_path):
    """After run, {run_id}_trace.jsonl and {run_id}.jsonl should be produced under sessions/{session_id}/."""
    sessions_dir = tmp_path / "sessions"
    session_id = "trace-test-session"
    config = HarnessConfig(
        tool_registry=make_registry(),
        tracer=HarnessJournal(base_dir=str(sessions_dir), export_jsonl=True, session_id=session_id),
        processors=[],
    )
    harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
    result = await harness.run(BaseTask(description="test"))

    # Layout: sessions/{session_id}/{run_id}_trace.jsonl
    session_dir = sessions_dir / session_id
    trace_path = session_dir / f"{result.run_id}_trace.jsonl"
    session_path = session_dir / f"{result.run_id}.jsonl"
    assert trace_path.exists(), f"Expected {trace_path}"
    assert session_path.exists(), f"Expected {session_path}"

    # trace jsonl: execution metadata, no message content
    with open(trace_path) as f:
        trace_lines = [json.loads(line) for line in f if line.strip()]
    assert len(trace_lines) > 0
    event_types = {r["event_type"] for r in trace_lines}
    assert "task_start" in event_types
    assert "task_end" in event_types
    for r in trace_lines:
        assert "session_id" in r
        assert "step" in r
        assert "messages" not in r  # no message content in trace

    # session.jsonl: conversation, all lines have session_id + uuid
    with open(session_path) as f:
        session_lines = [json.loads(line) for line in f if line.strip()]
    assert len(session_lines) > 0
    types = {r["type"] for r in session_lines}
    assert "session_start" in types
    assert "episode_end" in types
    for r in session_lines:
        assert "session_id" in r
        assert "uuid" in r
        assert "parent_uuid" in r


@pytest.mark.asyncio
async def test_harness_config_copy():
    """HarnessConfig.copy() should allow single-slot override."""
    from harnessx.core.config_schema import NullTracerConfig, TracerConfig

    original = MinimalConfig
    custom = original.copy(tracer=NullTracerConfig())
    assert isinstance(custom.tracer, TracerConfig)
    assert custom.tracer._target_ == "harnessx.tracing.null_tracer.NullTracer"
    assert custom.processors == original.processors


@pytest.mark.asyncio
async def test_budget_exceeded_exit():
    """RunLoop should exit with budget_exceeded when cost limit hit."""
    # Use max_cost_usd=0 to immediately trigger
    harness = make_harness(responses=["Done."])
    result = await harness.run(BaseTask(description="test", max_cost_usd=0.0))
    assert result.exit_reason == "budget_exceeded"


@pytest.mark.asyncio
async def test_full_trajectory_inspection():
    """
    Full-chain: task → RunLoop → tool call → tool result → final answer.
    Verifies the complete trajectory structure: steps, action, observation,
    state snapshot, state delta, and HarnessResult fields.
    """
    responses = [
        {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "add", "input": {"a": 10, "b": 32}}],
        },
        "The answer is 42.",
    ]
    harness = make_harness(responses=responses, tools=[add_tool])
    result = await harness.run(BaseTask(description="What is 10+32?", max_steps=10))

    # ── HarnessResult ─────────────────────────────────────────────────────────
    assert result.exit_reason in ("done", "budget_exceeded")
    assert result.final_output == "The answer is 42."
    assert result.total_steps >= 1
    assert result.run_id

    # ── Trajectory shape ──────────────────────────────────────────────────────
    traj = result.trajectory
    assert traj is not None
    assert traj.run_id == result.run_id
    assert len(traj.steps) >= 1

    # ── Step 0: tool call ─────────────────────────────────────────────────────
    step0 = traj.steps[0]
    assert step0.step_id == 0
    assert step0.action is not None
    assert len(step0.action.tool_calls) == 1
    tc = step0.action.tool_calls[0]
    assert tc.name == "add"
    assert tc.input == {"a": 10, "b": 32}

    # Tool result observed
    assert len(step0.observation) == 1
    obs = step0.observation[0]
    assert obs.result == "42"
    assert obs.error is None

    # ── State snapshot on each step ───────────────────────────────────────────
    for step in traj.steps:
        snap = step.state_snapshot
        assert snap is not None
        assert snap.step_id == step.step_id
        assert len(snap.messages) > 0  # conversation present
        assert snap.cumulative_tokens >= 0

        delta = step.state_delta
        assert delta is not None
        assert delta.step_id == step.step_id

    # ── Final step: model answered, no pending tool calls ─────────────────────
    last = traj.steps[-1]
    assert last.action is not None
    assert last.action.finish_reason in ("end_turn", "done", "stop", "")
    assert len(last.observation) == 0  # no tool calls on final answer


# ── HarnessJournal recovery tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_final_state_written(tmp_path):
    """After run, sessions/{session_id}/{run_id}_state.json must exist with schema_version==2."""
    sessions_dir = tmp_path / "sessions"
    session_id = "state-test-session"
    config = HarnessConfig(
        tool_registry=make_registry(),
        tracer=HarnessJournal(base_dir=str(sessions_dir), export_jsonl=True, session_id=session_id),
        processors=[],
    )
    harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
    result = await harness.run(BaseTask(description="test"))

    # Layout: sessions/{session_id}/{run_id}_state.json
    state_path = sessions_dir / session_id / f"{result.run_id}_state.json"
    assert state_path.exists(), f"segment state missing at {state_path}"
    with open(state_path) as f:
        data = json.load(f)
    assert data["schema_version"] == 2
    assert data["run_id"] == result.run_id
    assert "slots" in data
    # Current schema: messages are NOT in the snapshot — JSONL is the single source.


@pytest.mark.asyncio
async def test_harness_config_written(tmp_path):
    """harness_config.yaml must be written to workspace.root after a run."""
    from harnessx.workspace.workspace import Workspace

    ws_root = tmp_path / "ws"
    workspace = Workspace(root=ws_root, agent_id="test", mode="shared")
    config = HarnessConfig(
        tool_registry=make_registry(),
        workspace=workspace,
        tracer=HarnessJournal(export_jsonl=True),
        processors=[],
        init_workspace=False,
    )
    harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
    _result = await harness.run(BaseTask(description="test"))

    # Runtime snapshot is written to workspace.root (not the agent-shared default config).
    config_path = ws_root / "harness_config.yaml"
    assert config_path.exists(), f"harness_config.yaml missing at {config_path}"


@pytest.mark.asyncio
async def test_wake_from_disk(tmp_path):
    """Run → wake() → second run with resumed state has combined message history."""
    from harnessx.workspace.workspace import Workspace

    session_id = "test-wake-session"
    workspace_root = tmp_path / "ws"
    workspace = Workspace(root=workspace_root, agent_id="test", mode="shared")

    def _make_cfg(responses):
        config = HarnessConfig(
            tool_registry=make_registry(),
            workspace=workspace,
            tracer=HarnessJournal(export_jsonl=True, session_id=session_id),
            processors=[],
            init_workspace=False,
        )
        return ModelConfig(main=MockProvider(responses=responses)).agentic(config)

    # First run
    harness1 = _make_cfg(["First answer."])
    result1 = await harness1.run(BaseTask(description="Hello"), session_id=session_id)
    assert result1.final_output

    # Verify wake() works manually
    resumed_state = HarnessJournal.wake(session_id, str(workspace_root))
    assert len(resumed_state.messages) > 0

    # Second run: pass session_id only — harness auto-resumes from disk.
    harness2 = _make_cfg(["Second answer."])
    result2 = await harness2.run(
        BaseTask(description="Follow-up"),
        session_id=session_id,
    )
    assert result2.final_output

    # Session index: same run_id is reused across turns (new run_id only on compaction).
    index_path = workspace_root / "sessions" / f"{session_id}.json"
    with open(index_path) as f:
        idx = json.load(f)
    # Both turns share the same run_id — the index deduplicates it.
    assert result1.run_id == result2.run_id, "Same session should reuse run_id across turns"
    assert len(idx["run_ids"]) == 1
    assert idx["latest_run_id"] == result2.run_id


@pytest.mark.asyncio
async def test_resume_state_keeps_effective_messages_for_next_turn():
    """Resume must preserve restored effective messages, not overwrite from raw track."""

    class _CaptureProvider(MockProvider):
        def __init__(self):
            super().__init__(responses=["Second answer."])
            self.seen_messages = None

        async def complete(self, messages, tools, stream_callback=None, **kwargs):
            self.seen_messages = list(messages)
            return await super().complete(
                messages=messages,
                tools=tools,
                stream_callback=stream_callback,
                **kwargs,
            )

    provider = _CaptureProvider()
    harness = ModelConfig(main=provider).agentic(
        HarnessConfig(
            tool_registry=make_registry(),
            tracer=NullTracer(),
            processors=[],
        )
    )

    resume_state = State(run_id="run-resume-ctx", max_steps=8)
    resume_state.raw_messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="first answer"),
    ]
    resume_state.messages = [
        Message(role="system", content="S: once"),
        Message(role="user", content="<wrapped>hello</wrapped>"),
        Message(role="assistant", content="first answer"),
    ]

    result = await harness.run(
        BaseTask(description="follow-up", max_steps=1),
        _resume_state=resume_state,
    )
    assert result.final_output == "Second answer."
    assert provider.seen_messages is not None

    seen = provider.seen_messages
    assert [m.role for m in seen[:4]] == ["system", "user", "assistant", "user"]
    assert seen[0].content == "S: once"
    assert "wrapped" in str(seen[1].content)
    assert seen[3].content == "follow-up"


@pytest.mark.asyncio
async def test_cancelled_run_injects_user_interrupted_assistant_message():
    """Ctrl-C style cancellation should append a terminal assistant interruption message."""

    class _SlowProvider(MockProvider):
        async def complete(self, messages, tools, stream_callback=None, **kwargs):
            await asyncio.sleep(10.0)
            return await super().complete(messages, tools, stream_callback=stream_callback, **kwargs)

    harness = ModelConfig(main=_SlowProvider(responses=["Should not arrive."])).agentic(
        HarnessConfig(
            tool_registry=make_registry(),
            tracer=NullTracer(),
            processors=[],
        )
    )

    task = asyncio.create_task(harness.run(BaseTask(description="hello", max_steps=5)))
    await asyncio.sleep(0.05)
    task.cancel()
    result = await task

    assert result.exit_reason == "interrupted"
    assert result.final_output == "user actively interrupted execution"
    assert result.resume_state.raw_messages[-1].role == "assistant"
    assert result.resume_state.raw_messages[-1].content == "user actively interrupted execution"
    assert result.resume_state.messages[-1].role == "assistant"
    assert result.resume_state.messages[-1].content == "user actively interrupted execution"


@pytest.mark.asyncio
async def test_cancelled_run_persists_interrupt_message_in_wake(tmp_path):
    """Interrupted runs should persist the terminal assistant message for wake() recovery."""
    from harnessx.workspace.workspace import Workspace

    class _SlowProvider(MockProvider):
        async def complete(self, messages, tools, stream_callback=None, **kwargs):
            await asyncio.sleep(10.0)
            return await super().complete(messages, tools, stream_callback=stream_callback, **kwargs)

    session_id = "cancel-wake-session"
    ws_root = tmp_path / "ws"
    workspace = Workspace(root=ws_root, agent_id="test", mode="shared")

    def _make_harness(provider):
        cfg = HarnessConfig(
            tool_registry=make_registry(),
            workspace=workspace,
            tracer=HarnessJournal(
                base_dir=str(ws_root / "sessions"),
                export_jsonl=True,
                session_id=session_id,
            ),
            processors=[],
            init_workspace=False,
        )
        return ModelConfig(main=provider).agentic(cfg)

    # Build non-zero step history first, then interrupt the next turn.
    harness_ok = _make_harness(MockProvider(responses=["First answer."]))
    first = await harness_ok.run(BaseTask(description="first", max_steps=5), session_id=session_id)
    assert first.exit_reason == "done"

    harness = _make_harness(_SlowProvider(responses=["Should not arrive."]))

    task = asyncio.create_task(harness.run(BaseTask(description="cancel me", max_steps=5), session_id=session_id))
    await asyncio.sleep(0.05)
    task.cancel()
    result = await task
    assert result.exit_reason == "interrupted"

    resumed = HarnessJournal.wake(session_id, str(ws_root))
    assert resumed.raw_messages[-1].role == "assistant"
    assert resumed.raw_messages[-1].content == "user actively interrupted execution"
    assert resumed.messages[-1].role == "assistant"
    assert resumed.messages[-1].content == "user actively interrupted execution"


@pytest.mark.asyncio
async def test_sessions_index_updated(tmp_path):
    """Multiple turns with same session_id share a run_id; index is created and valid."""
    sessions_dir = tmp_path / "sessions"
    session_id = "multi-run-session"

    run_ids = []
    for i, resp in enumerate(["Run 1.", "Run 2.", "Run 3."]):
        config = HarnessConfig(
            tool_registry=make_registry(),
            tracer=HarnessJournal(
                base_dir=str(sessions_dir),
                export_jsonl=True,
                session_id=session_id,
            ),
            processors=[],
        )
        harness = ModelConfig(main=MockProvider(responses=[resp])).agentic(config)
        result = await harness.run(BaseTask(description=f"Task {i}"), session_id=session_id)
        run_ids.append(result.run_id)

    # All three turns share the same run_id (new run_id only on compaction/boundary).
    assert run_ids[0] == run_ids[1] == run_ids[2], (
        "Same session without compaction should reuse the same run_id across turns"
    )

    index_path = sessions_dir / f"{session_id}.json"
    assert index_path.exists()
    with open(index_path) as f:
        idx = json.load(f)

    assert idx["schema_version"] == 1
    assert idx["session_id"] == session_id
    # run_ids is deduplicated — one entry for the shared run_id
    assert idx["run_ids"] == [run_ids[0]]
    assert idx["latest_run_id"] == run_ids[0]
    # No duplicate run_ids in the index
    assert len(idx["run_ids"]) == len(set(idx["run_ids"]))


# ── Content-addressed config + config_hash tests ──────────────────────────────


@pytest.mark.asyncio
async def test_config_hash_in_final_state(tmp_path):
    """Segment state file must contain config_hash when workspace is set."""
    from harnessx.workspace.workspace import Workspace

    ws_root = tmp_path / "ws"
    session_id = "config-hash-session"
    workspace = Workspace(root=ws_root, agent_id="test", mode="shared")
    config = HarnessConfig(
        tool_registry=make_registry(),
        workspace=workspace,
        tracer=HarnessJournal(base_dir=str(ws_root / "sessions"), export_jsonl=True, session_id=session_id),
        processors=[],
        init_workspace=False,
    )
    harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
    result = await harness.run(BaseTask(description="test"))

    # Layout: sessions/{session_id}/{run_id}_state.json
    state_path = ws_root / "sessions" / session_id / f"{result.run_id}_state.json"
    with open(state_path) as f:
        data = json.load(f)
    assert "config_hash" in data, "segment state must contain config_hash"
    assert len(data["config_hash"]) == 64, "config_hash must be a sha256 hex digest"


@pytest.mark.asyncio
async def test_config_deduplication(tmp_path):
    """Same Harness config run twice must produce only one new file in agent configs/."""
    from harnessx.workspace.workspace import Workspace
    from harnessx.home import agent_configs_dir

    ws_root = tmp_path / "ws"
    workspace = Workspace(root=ws_root, agent_id="test", mode="shared")

    def _make_cfg():
        config = HarnessConfig(
            tool_registry=make_registry(),
            workspace=workspace,
            tracer=HarnessJournal(export_jsonl=True),
            processors=[],
            init_workspace=False,
        )
        return ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)

    configs_dir = agent_configs_dir()
    before = set(configs_dir.glob("*.yaml")) if configs_dir.exists() else set()

    harness1 = _make_cfg()
    harness2 = _make_cfg()
    await harness1.run(BaseTask(description="first task"))
    await harness2.run(BaseTask(description="second task"))

    after = set(configs_dir.glob("*.yaml"))
    new_files = after - before
    assert len(new_files) == 1, (
        f"Same config run twice should produce exactly 1 new file in agent configs/, got {len(new_files)}"
    )


@pytest.mark.asyncio
async def test_wake_config_from_run(tmp_path):
    """wake_config(from_run=run_id) returns config of that specific run."""
    from harnessx.workspace.workspace import Workspace

    ws_root = tmp_path / "ws"
    session_id = "repro-session"
    workspace = Workspace(root=ws_root, agent_id="test", mode="shared")

    config = HarnessConfig(
        tool_registry=make_registry(),
        workspace=workspace,
        tracer=HarnessJournal(export_jsonl=True, session_id=session_id),
        processors=[],
        init_workspace=False,
    )
    harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
    result = await harness.run(BaseTask(description="benchmark task"), session_id=session_id)

    # wake_config with from_run should return the exact harness config dict
    # (model config is separate — not included in harness_config.yaml)
    cfg = HarnessJournal.wake_config(session_id, str(ws_root), from_run=result.run_id)
    assert isinstance(cfg, dict)
    assert len(cfg) > 0, "Config dict should not be empty"
    # Model is intentionally excluded; workspace and/or sandbox are always present
    assert "model" not in cfg, "harness_config.yaml must not contain model section"
    assert "workspace" in cfg or "sandbox" in cfg or "harness" in cfg or "tools" in cfg


@pytest.mark.asyncio
async def test_custom_tools_recorded_in_config(tmp_path):
    """Custom tool functions should appear under tools.custom in harness_config.yaml."""
    from harnessx.workspace.workspace import Workspace
    from harnessx.tools.base import Tool

    ws_root = tmp_path / "ws"
    workspace = Workspace(root=ws_root, agent_id="test", mode="shared")

    custom_tool = Tool(
        name="my_custom_tool",
        description="A custom tool",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        fn=_module_level_custom_tool_fn,
        tags=[],
        execution_target="local",
    )
    registry = make_registry()
    registry.register(custom_tool)

    config = HarnessConfig(
        tool_registry=registry,
        workspace=workspace,
        tracer=HarnessJournal(export_jsonl=True),
        processors=[],
        init_workspace=False,
    )
    harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
    await harness.run(BaseTask(description="test"))

    config_path = ws_root / "harness_config.yaml"
    content = config_path.read_text()
    # Tools emitted under tools.custom store their fn's module::qualname path
    # (not the Tool.name attribute) — the loader re-imports the fn from there.
    assert "_module_level_custom_tool_fn" in content, (
        "Custom tool fn path should appear under tools.custom in harness_config.yaml"
    )


# ─── Compaction + SegmentBoundaryEvent ───────────────────────────────────────


@pytest.mark.asyncio
async def test_compaction_emits_segment_boundary_event(tmp_path):
    """CompactionProcessor emits SegmentBoundaryEvent when message_threshold is hit."""
    collected: list[SegmentBoundaryEvent] = []

    @on_step_end
    async def _capture_boundary(event):
        # Capture via a side-channel — we monkey-patch the tracer below
        yield event

    # Use a recording tracer that captures SegmentBoundaryEvent
    class _RecordingTracer(NullTracer):
        async def on_event(self, event):
            if isinstance(event, SegmentBoundaryEvent):
                collected.append(event)

    # threshold=2: fires when there are > 2 messages accumulated
    # Use tool calls to force multiple steps: step 0 → tool call, step 1 → tool call,
    # step 2 → final answer.  Each tool call+result pair adds 2 messages.
    compaction = CompactionProcessor(
        token_threshold=999_999,  # disable token trigger
        message_threshold=3,  # fire after 3 messages (user + assistant + tool = 3)
        retention_window=1,
    )

    responses = [
        {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "a"}}],
        },
        {
            "content": "",
            "tool_calls": [{"id": "c2", "name": "echo", "input": {"message": "b"}}],
        },
        "Done.",
    ]

    config = HarnessConfig(
        tool_registry=make_registry(echo_tool),
        tracer=_RecordingTracer(),
        processors=[compaction],
    )
    harness = ModelConfig(main=MockProvider(responses=responses)).agentic(config)
    result = await harness.run(BaseTask(description="Run a multi-step task.", max_steps=5))

    assert result.exit_reason == "done"
    assert len(collected) >= 1, "At least one SegmentBoundaryEvent must be emitted when message_threshold is exceeded"
    for ev in collected:
        assert ev.reason == "compaction"
        assert ev.new_run_id != ev.run_id
        assert ev.new_run_id != ""


@pytest.mark.asyncio
async def test_compaction_rotates_journal_files(tmp_path):
    """After compaction, HarnessJournal creates new segment files under sessions/{session_id}/."""
    session_id = "test-compaction-session"
    ws_root = tmp_path / "workspace"

    compaction = CompactionProcessor(
        token_threshold=999_999,
        message_threshold=3,  # fire after user+assistant+tool = 3 messages
        retention_window=1,
    )

    journal = HarnessJournal(
        base_dir=str(ws_root / "sessions"),
        export_jsonl=True,
        silent=True,
        session_id=session_id,
    )

    responses = [
        {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "a"}}],
        },
        {
            "content": "",
            "tool_calls": [{"id": "c2", "name": "echo", "input": {"message": "b"}}],
        },
        "Done.",
    ]

    config = HarnessConfig(
        tool_registry=make_registry(echo_tool),
        tracer=journal,
        processors=[compaction],
        init_workspace=False,
    )
    harness = ModelConfig(main=MockProvider(responses=responses)).agentic(config)
    result = await harness.run(BaseTask(description="Compact me.", max_steps=5))

    assert result.exit_reason == "done"

    session_dir = ws_root / "sessions" / session_id
    assert session_dir.is_dir(), f"Session directory must exist: {session_dir}"

    # At least two .jsonl segment files should exist (one before compaction, one after)
    segment_files = sorted(session_dir.glob("*.jsonl"))
    assert len(segment_files) >= 2, (
        f"Expected ≥2 segment .jsonl files after compaction, found: {[f.name for f in segment_files]}"
    )

    # Segment state checkpoint must exist for the first segment
    state_files = sorted(session_dir.glob("*_state.json"))
    assert len(state_files) >= 1, f"Expected ≥1 _state.json checkpoint, found: {[f.name for f in state_files]}"
    for sf in state_files:
        with open(sf) as f:
            data = json.load(f)
        assert data.get("schema_version") == 2
        assert "segment_end_reason" in data

    # Session index must list multiple run_ids
    index_path = ws_root / "sessions" / f"{session_id}.json"  # index at sessions/{session_id}.json
    assert index_path.exists(), "Session index must exist"
    with open(index_path) as f:
        idx = json.load(f)
    assert len(idx["run_ids"]) >= 2, f"Session index must list ≥2 run_ids after compaction, got: {idx['run_ids']}"


@pytest.mark.asyncio
async def test_wake_after_compaction(tmp_path):
    """wake() correctly restores state from the latest segment after compaction."""
    session_id = "wake-after-compact"
    ws_root = tmp_path / "workspace"

    compaction = CompactionProcessor(
        token_threshold=999_999,
        message_threshold=3,
        retention_window=1,
    )

    journal = HarnessJournal(
        base_dir=str(ws_root / "sessions"),
        export_jsonl=True,
        silent=True,
        session_id=session_id,
    )

    responses = [
        {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "a"}}],
        },
        {
            "content": "",
            "tool_calls": [{"id": "c2", "name": "echo", "input": {"message": "b"}}],
        },
        "Done.",
    ]

    config = HarnessConfig(
        tool_registry=make_registry(echo_tool),
        tracer=journal,
        processors=[compaction],
        init_workspace=False,
    )
    harness = ModelConfig(main=MockProvider(responses=responses)).agentic(config)
    result = await harness.run(BaseTask(description="Hello.", max_steps=5))
    assert result.exit_reason == "done"

    # Restore via wake() — must not raise
    from harnessx.core.state import State

    state = HarnessJournal.wake(session_id, str(ws_root))
    assert isinstance(state, State)
    # State must reflect completed steps
    assert state.step >= 1
