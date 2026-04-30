# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import pytest
from harnessx.core.events import (
    BeforeModelEvent,
    StepStartEvent,
    Message,
    StepEndEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from harnessx.core.runloop import BudgetExceededError, LoopDetectedError
from harnessx.processors import (
    CostGuardProcessor,
    LoopDetectionProcessor,
    RepeatedFileEditDetector,
    ToolWhitelistProcessor,
    TokenBudgetProcessor,
    CheckpointProcessor,
)
from harnessx.core.processor import pipe


async def _tool_pair(processor, tool_name: str, tool_input: dict, *, run_id="r1", step_id=0, call_id="c1"):
    """Simulate a single tool call + result through the processor chain.

    Returns the final ToolResultEvent (possibly modified by the processor).
    """
    tc = ToolCallEvent(
        run_id=run_id,
        step_id=step_id,
        tool_name=tool_name,
        tool_call_id=call_id,
        tool_input=tool_input,
        approved=True,
    )
    await pipe(tc, [processor])
    tr = ToolResultEvent(
        run_id=run_id,
        step_id=step_id,
        tool_name=tool_name,
        tool_call_id=call_id,
        result="ok",
    )
    return await pipe(tr, [processor])


async def _edit_file(processor, path, call_id):
    """Helper: simulate one Write tool call through before_tool + after_tool."""
    call_event = ToolCallEvent(
        run_id="r1",
        step_id=0,
        tool_name="Write",
        tool_input={"file_path": path},
        tool_call_id=call_id,
    )
    await pipe(call_event, [processor])
    result_event = ToolResultEvent(
        run_id="r1",
        step_id=0,
        tool_name="Write",
        tool_call_id=call_id,
        result="ok",
    )
    return await pipe(result_event, [processor])


class TestProcessors:
    @pytest.mark.asyncio
    async def test_cost_guard_passes_under_limit(self):
        processor = CostGuardProcessor(max_usd=1.0)
        event = BeforeModelEvent(run_id="r1", step_id=0, cumulative_cost_usd=0.5)
        result = await pipe(event, [processor])
        assert result is not None

    @pytest.mark.asyncio
    async def test_cost_guard_raises_at_limit(self):
        processor = CostGuardProcessor(max_usd=1.0)
        event = BeforeModelEvent(run_id="r1", step_id=0, cumulative_cost_usd=1.5)
        with pytest.raises(BudgetExceededError):
            await pipe(event, [processor])

    @pytest.mark.asyncio
    async def test_loop_detection_no_loop(self):
        processor = LoopDetectionProcessor(threshold=3)
        for i in range(5):
            # Each call has a unique input — no loop
            result = await _tool_pair(processor, "Bash", {"cmd": f"ls {i}"}, step_id=i, call_id=f"c{i}")
            assert result is not None
            assert "[LoopDetection]" not in (result.result or "")

    @pytest.mark.asyncio
    async def test_loop_detection_detects_loop(self):
        processor = LoopDetectionProcessor(threshold=3, warn_threshold=2, window_size=10)
        # First call: no warning
        r1 = await _tool_pair(processor, "Bash", {"cmd": "ls"}, step_id=0, call_id="c0")
        assert "[LoopDetection]" not in (r1.result or "")
        # Second: warning injected (warn_threshold=2)
        r2 = await _tool_pair(processor, "Bash", {"cmd": "ls"}, step_id=1, call_id="c1")
        assert "[LoopDetection]" in (r2.result or "")
        # Third: raises LoopDetectedError
        tc = ToolCallEvent(
            run_id="r1",
            step_id=2,
            tool_name="Bash",
            tool_call_id="c2",
            tool_input={"cmd": "ls"},
            approved=True,
        )
        await pipe(tc, [processor])
        tr = ToolResultEvent(run_id="r1", step_id=2, tool_name="Bash", tool_call_id="c2", result="ok")
        with pytest.raises(LoopDetectedError):
            await pipe(tr, [processor])

    @pytest.mark.asyncio
    async def test_loop_detection_interleaved_no_raise(self):
        """Non-consecutive repeats (ls → cat → ls → cat → ls) must NOT raise or warn."""
        processor = LoopDetectionProcessor(threshold=3, warn_threshold=2, window_size=10)
        for i in range(3):
            r_ls = await _tool_pair(processor, "Bash", {"cmd": "ls"}, step_id=i * 2, call_id=f"ls{i}")
            r_cat = await _tool_pair(processor, "Bash", {"cmd": "cat"}, step_id=i * 2 + 1, call_id=f"cat{i}")
            # ls and cat interleave — consecutive run never reaches warn_threshold
            assert "[LoopDetection]" not in (r_ls.result or "")
            assert "[LoopDetection]" not in (r_cat.result or "")

    @pytest.mark.asyncio
    async def test_tool_whitelist_allows_listed(self):
        processor = ToolWhitelistProcessor(allowed_tools=["bash"], dangerous_tools=[])
        event = ToolCallEvent(run_id="r1", step_id=0, tool_name="bash", tool_input={}, approved=True)
        result = await pipe(event, [processor])
        assert result is not None
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_tool_whitelist_blocks_unlisted(self):
        processor = ToolWhitelistProcessor(allowed_tools=["bash"], dangerous_tools=[], allow_all=False)
        event = ToolCallEvent(
            run_id="r1",
            step_id=0,
            tool_name="unknown_tool",
            tool_input={},
            approved=True,
        )
        result = await pipe(event, [processor])
        assert result is not None
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_tool_whitelist_allow_all(self):
        processor = ToolWhitelistProcessor(allow_all=True, dangerous_tools=[])
        event = ToolCallEvent(run_id="r1", step_id=0, tool_name="any_tool", tool_input={}, approved=True)
        result = await pipe(event, [processor])
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_token_budget_no_memory_passes_through(self):
        """TokenBudgetProcessor should pass through when under budget."""
        processor = TokenBudgetProcessor()
        event = StepStartEvent(
            run_id="r1",
            step_id=0,
            messages=(Message(role="user", content="hello"),),
            token_count=100,
        )
        result = await pipe(event, [processor])
        assert result is not None
        assert result.token_count == 100

    @pytest.mark.asyncio
    async def test_checkpoint_saves(self, tmp_path):
        db_path = str(tmp_path / "ckpt.db")
        processor = CheckpointProcessor(every_n=5, db_path=db_path)

        snapshot = {"run_id": "r1", "step": 5, "messages": []}
        event = StepEndEvent(run_id="r1", step_id=5, state_snapshot=snapshot)
        await pipe(event, [processor])

        # Verify checkpoint was saved
        loaded = CheckpointProcessor.load_checkpoint("r1", db_path=db_path)
        assert loaded is not None
        assert loaded["run_id"] == "r1"

    @pytest.mark.asyncio
    async def test_repeated_edit_no_hint_below_soft_threshold(self):
        processor = RepeatedFileEditDetector(soft_threshold=7, hard_threshold=12)
        for i in range(6):
            result = await _edit_file(processor, "/app/foo.py", f"id-{i}")
        assert "RepeatedFileEditDetector" not in result.result

    @pytest.mark.asyncio
    async def test_repeated_edit_soft_hint_at_soft_threshold(self):
        processor = RepeatedFileEditDetector(soft_threshold=7, hard_threshold=12)
        result = None
        for i in range(7):
            result = await _edit_file(processor, "/app/foo.py", f"id-{i}")
        assert "RepeatedFileEditDetector" in result.result
        assert "/app/foo.py" in result.result
        # Soft hint does NOT reset counter
        result = await _edit_file(processor, "/app/foo.py", "id-7")
        assert "RepeatedFileEditDetector" in result.result  # still >= soft threshold

    @pytest.mark.asyncio
    async def test_repeated_edit_hard_hint_resets_counter(self):
        processor = RepeatedFileEditDetector(soft_threshold=7, hard_threshold=12)
        # Trigger hard hint on 12th edit
        for i in range(12):
            await _edit_file(processor, "/app/foo.py", f"id-{i}")
        # Counter reset after hard hint — next 5 edits (count 1-5) should not trigger
        for i in range(12, 17):
            result = await _edit_file(processor, "/app/foo.py", f"id-{i}")
        assert "RepeatedFileEditDetector" not in result.result

    @pytest.mark.asyncio
    async def test_repeated_edit_tracks_files_independently(self):
        processor = RepeatedFileEditDetector(soft_threshold=7, hard_threshold=12)
        for i in range(6):
            await _edit_file(processor, "/app/a.py", f"a-{i}")
        # Different file — should not trigger
        result = await _edit_file(processor, "/app/b.py", "b-0")
        assert "RepeatedFileEditDetector" not in result.result
        # a.py hits soft threshold on 7th edit
        result = await _edit_file(processor, "/app/a.py", "a-6")
        assert "RepeatedFileEditDetector" in result.result
