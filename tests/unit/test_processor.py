# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import pytest
from harnessx.core.events import (
    BeforeModelEvent,
    Event,
    ModelResponseEvent,
    SpawnSubAgentEvent,
    StepEndEvent,
    TaskEndEvent,
)
from harnessx.core.processor import (
    MultiHookProcessor,
    ProcessorChain,
    on,
    on_step_end,
    pipe,
    pipe_all,
)


class PassThroughProcessor:
    async def process(self, event: Event):
        yield event


class BlockAllProcessor:
    async def process(self, event: Event):
        return
        yield  # never yields


class DoubleProcessor:
    """Yields event twice."""

    async def process(self, event: Event):
        yield event
        yield event


class TestProcessor:
    @pytest.mark.asyncio
    async def test_pass_through(self):
        event = StepEndEvent(run_id="r1", step_id=0)
        result = await pipe(event, [PassThroughProcessor()])
        assert result is event

    @pytest.mark.asyncio
    async def test_block_all(self):
        event = StepEndEvent(run_id="r1", step_id=0)
        result = await pipe(event, [BlockAllProcessor()])
        assert result is None

    @pytest.mark.asyncio
    async def test_processor_chain_ordering(self):
        """Processors should run in order."""
        order = []

        class RecordProcessor:
            def __init__(self, name):
                self.name = name

            async def process(self, event: Event):
                order.append(self.name)
                yield event

        event = StepEndEvent(run_id="r1", step_id=0)
        chain = ProcessorChain(RecordProcessor("A"), RecordProcessor("B"), RecordProcessor("C"))
        results = []
        async for ev in chain.process(event):
            results.append(ev)

        assert order == ["A", "B", "C"]
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_on_step_end_hook(self):
        """on_step_end decorator should wrap function as processor."""
        called_with = []

        @on_step_end
        async def my_hook(event: StepEndEvent):
            called_with.append(event.step_id)
            yield event

        event = StepEndEvent(run_id="r1", step_id=7)
        result = await pipe(event, [my_hook])
        assert result is not None
        assert 7 in called_with

    @pytest.mark.asyncio
    async def test_hook_only_fires_on_matching_event_type(self):
        """Hook should pass through events of other types unchanged."""
        called = []

        @on_step_end
        async def my_hook(event: StepEndEvent):
            called.append(True)
            yield event

        # Pass a non-StepEndEvent
        event = TaskEndEvent(run_id="r1", step_id=0)
        result = await pipe(event, [my_hook])
        assert result is not None
        assert called == []  # Hook should NOT be called

    @pytest.mark.asyncio
    async def test_pipe_empty_processors(self):
        """pipe with empty processor list returns original event."""
        event = StepEndEvent(run_id="r1", step_id=0)
        result = await pipe(event, [])
        assert result is event

    # ─── pipe_all ─────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_pipe_all_empty_returns_single_event(self):
        event = StepEndEvent(run_id="r1", step_id=0)
        results = await pipe_all(event, [])
        assert results == [event]

    @pytest.mark.asyncio
    async def test_pipe_all_pass_through(self):
        event = StepEndEvent(run_id="r1", step_id=0)
        results = await pipe_all(event, [PassThroughProcessor()])
        assert results == [event]

    @pytest.mark.asyncio
    async def test_pipe_all_collects_multi_yield(self):
        """pipe_all must return every yielded event, not just the last."""
        event = StepEndEvent(run_id="r1", step_id=0)
        results = await pipe_all(event, [DoubleProcessor()])
        assert len(results) == 2
        assert all(r is event for r in results)

    @pytest.mark.asyncio
    async def test_pipe_all_intercept_returns_empty(self):
        event = StepEndEvent(run_id="r1", step_id=0)
        results = await pipe_all(event, [BlockAllProcessor()])
        assert results == []

    @pytest.mark.asyncio
    async def test_pipe_all_heterogeneous_event_types(self):
        """Processor may yield events of different types — all are returned."""
        model_event = ModelResponseEvent(run_id="r1", step_id=0)
        spawn_event = SpawnSubAgentEvent(run_id="r1", step_id=0)

        class ForkProcessor:
            async def process(self, event: Event):
                yield event
                yield spawn_event

        results = await pipe_all(model_event, [ForkProcessor()])
        assert len(results) == 2
        assert isinstance(results[0], ModelResponseEvent)
        assert isinstance(results[1], SpawnSubAgentEvent)

    @pytest.mark.asyncio
    async def test_pipe_still_returns_last_event(self):
        """pipe() backward-compat: still returns the last yielded event."""
        event = StepEndEvent(run_id="r1", step_id=0)
        result = await pipe(event, [DoubleProcessor()])
        assert result is event  # last yield, not None

    # ─── @on() decorator ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_on_decorator_dispatches_by_type(self):
        """@on(EventType) registers handler in MultiHookProcessor._DISPATCH."""
        called_with = []

        class MyProc(MultiHookProcessor):
            @on(BeforeModelEvent)
            async def guard(self, event: BeforeModelEvent):
                called_with.append(event)
                yield event

        proc = MyProc()
        ev = BeforeModelEvent(run_id="r1", step_id=0)
        result = await pipe(ev, [proc])
        assert result is ev
        assert called_with == [ev]

    @pytest.mark.asyncio
    async def test_on_decorator_ignores_other_types(self):
        """@on handler must NOT fire for unrelated event types."""
        called = []

        class MyProc(MultiHookProcessor):
            @on(BeforeModelEvent)
            async def guard(self, event: BeforeModelEvent):
                called.append(True)
                yield event

        proc = MyProc()
        ev = StepEndEvent(run_id="r1", step_id=0)
        result = await pipe(ev, [proc])
        assert result is ev
        assert called == []

    @pytest.mark.asyncio
    async def test_on_decorator_multiple_handlers(self):
        """One class can use @on on multiple methods."""
        log = []

        class DualProc(MultiHookProcessor):
            @on(BeforeModelEvent)
            async def before(self, event: BeforeModelEvent):
                log.append("before_model")
                yield event

            @on(StepEndEvent)
            async def after(self, event: StepEndEvent):
                log.append("step_end")
                yield event

        proc = DualProc()
        await pipe(BeforeModelEvent(run_id="r1", step_id=0), [proc])
        await pipe(StepEndEvent(run_id="r1", step_id=0), [proc])
        assert log == ["before_model", "step_end"]

    @pytest.mark.asyncio
    async def test_on_decorator_inherits_parent_dispatch(self):
        """Subclass @on entries merge with parent _DISPATCH without clobbering it."""
        log = []

        class Base(MultiHookProcessor):
            async def on_step_end(self, event: StepEndEvent):
                log.append("base_step_end")
                yield event

        class Child(Base):
            @on(BeforeModelEvent)
            async def extra(self, event: BeforeModelEvent):
                log.append("child_before_model")
                yield event

        proc = Child()
        await pipe(StepEndEvent(run_id="r1", step_id=0), [proc])
        await pipe(BeforeModelEvent(run_id="r1", step_id=0), [proc])
        assert log == ["base_step_end", "child_before_model"]

    @pytest.mark.asyncio
    async def test_on_decorator_overrides_parent_method(self):
        """@on(EventType) in a subclass overrides the parent's on_* method for that type."""
        log = []

        class Parent(MultiHookProcessor):
            async def on_before_model(self, event: BeforeModelEvent):
                log.append("parent")
                yield event

        class Child(Parent):
            @on(BeforeModelEvent)
            async def my_check(self, event: BeforeModelEvent):
                log.append("child")
                yield event

        proc = Child()
        await pipe(BeforeModelEvent(run_id="r1", step_id=0), [proc])
        assert log == ["child"]  # parent's on_before_model is superseded

    # ─── MultiHookProcessor override style (Direction B) ─────────────────────────

    @pytest.mark.asyncio
    async def test_multi_hook_processor_method_override(self):
        """Classic Direction B: override on_* method without @on decorator."""
        called = []

        class AuditProc(MultiHookProcessor):
            async def on_step_end(self, event: StepEndEvent):
                called.append(event.step_id)
                yield event

        proc = AuditProc()
        ev = StepEndEvent(run_id="r1", step_id=42)
        result = await pipe(ev, [proc])
        assert result is ev
        assert called == [42]

    @pytest.mark.asyncio
    async def test_multi_hook_processor_pass_through_unhandled(self):
        """MultiHookProcessor must pass through event types it doesn't handle."""

        class StepOnlyProc(MultiHookProcessor):
            async def on_step_end(self, event: StepEndEvent):
                yield event

        proc = StepOnlyProc()
        ev = TaskEndEvent(run_id="r1", step_id=0)
        result = await pipe(ev, [proc])
        assert result is ev  # pass-through, not intercepted
