# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from types import SimpleNamespace

import pytest

from harnessx.core.events import (
    ModelResponseEvent,
    ProcessorTriggerEvent,
    TaskEndEvent,
    TaskStartEvent,
    Usage,
)
from harnessx.core.harness import BaseTask, HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.core.processor import MultiHookProcessor, pipe_all
from harnessx.core.state import State
from harnessx.processors.multi_model.model_router import ModelRouterProcessor
from harnessx.tracing.null_tracer import NullTracer


class _FakeSubHarness:
    def __init__(self, output: str = "", exc: Exception | None = None) -> None:
        self.output = output
        self.exc = exc
        self.calls: list[tuple[BaseTask, str | None]] = []

    async def run(self, task: BaseTask, *, parent_run_id: str | None = None):
        self.calls.append((task, parent_run_id))
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(final_output=self.output)


class _FakeProvider:
    def __init__(self, model: str, reply: str) -> None:
        self.model = model
        self.reply = reply
        self.calls = 0
        self.context_window = 32_000

    async def complete(self, messages, tools, stream_callback=None):
        self.calls += 1
        return ModelResponseEvent(
            run_id="provider",
            step_id=0,
            content=self.reply,
            finish_reason="stop",
            usage=Usage(input_tokens=3, output_tokens=2),
            model=self.model,
        )

    def count_tokens(self, messages):
        return 0

    def annotate_trajectory(self, trajectory) -> None:
        return None


class _RouterAndAnswerProvider:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0
        self.context_window = 32_000

    async def complete(self, messages, tools, stream_callback=None):
        self.calls += 1
        text = ""
        if messages:
            last = messages[-1]
            if isinstance(last.content, str):
                text = last.content
        if "[MODEL_ROUTER_CLASSIFY]" in text:
            content = '{"complexity":"simple","confidence":0.95,"reason":"easy"}'
        else:
            content = f"answer-{self.model}"
        return ModelResponseEvent(
            run_id="provider",
            step_id=0,
            content=content,
            finish_reason="stop",
            usage=Usage(input_tokens=3, output_tokens=2),
            model=self.model,
        )

    def count_tokens(self, messages):
        return 0

    def annotate_trajectory(self, trajectory) -> None:
        return None


class _RouteSlotProcessor(MultiHookProcessor):
    _singleton_group = "test_route_slot"
    _order = 1

    def __init__(self, selected_key: str) -> None:
        self.selected_key = selected_key

    async def on_task_start(self, event: TaskStartEvent):
        if event.state is not None:
            event.state.set_slot(
                "model.route",
                "model_route",
                {"selected_key": self.selected_key, "source": "test"},
            )
        yield event


def _task_start_event(state: State, description: str = "hello") -> TaskStartEvent:
    return TaskStartEvent(
        run_id=state.run_id,
        step_id=0,
        task_description=description,
        state=state,
    )


def _triggers(events: list) -> list:
    return [e for e in events if isinstance(e, ProcessorTriggerEvent)]


class TestParseClassifierOutput:
    def _parse(self, text: str):
        return ModelRouterProcessor()._parse_classifier_output(text)

    def test_parse_json_with_label_key(self):
        result = self._parse('{"label":"complex","confidence":0.8}')
        assert result == ("complex", 0.8, "")

    def test_parse_json_with_route_key(self):
        result = self._parse('{"route":"simple","confidence":0.85}')
        assert result == ("simple", 0.85, "")

    def test_parse_json_embedded_in_text(self):
        result = self._parse('Here is my answer: {"complexity":"simple","confidence":0.9,"reason":"trivial"} done')
        assert result is not None
        label, conf, reason = result
        assert label == "simple"
        assert conf == 0.9
        assert reason == "trivial"

    def test_parse_regex_fallback_simple(self):
        result = self._parse("I think this is a simple task")
        assert result is not None
        label, conf, reason = result
        assert label == "simple"
        assert conf == 0.6  # default when no confidence number found

    def test_parse_regex_fallback_complex_with_confidence(self):
        result = self._parse("This is complex, confidence 0.85")
        assert result is not None
        label, conf, reason = result
        assert label == "complex"
        assert conf == 0.85

    def test_parse_empty_string(self):
        assert self._parse("") is None

    def test_parse_unrecognizable(self):
        assert self._parse("I have no idea what to do") is None

    def test_parse_invalid_json_no_keywords(self):
        assert self._parse('{"foo":"bar"}') is None

    def test_parse_uppercase_label(self):
        """Mixed-case labels like 'SIMPLE' should be normalized via .lower()."""
        result = self._parse('{"complexity":"SIMPLE","confidence":0.9}')
        assert result is not None
        assert result[0] == "simple"

    def test_parse_non_numeric_confidence_defaults_to_one(self):
        """Non-numeric confidence string (e.g. 'high') falls back to 1.0."""
        result = self._parse('{"complexity":"simple","confidence":"high"}')
        assert result is not None
        label, conf, _ = result
        assert label == "simple"
        assert conf == 1.0

    def test_parse_markdown_fenced_json(self):
        """JSON wrapped in markdown code fence should be extracted."""
        text = '```json\n{"complexity":"complex","confidence":0.8}\n```'
        result = self._parse(text)
        assert result is not None
        assert result[0] == "complex"
        assert result[1] == 0.8

    def test_parse_both_simple_and_complex_in_text_simple_wins(self):
        """When both 'simple' and 'complex' appear, 'simple' wins (if-elif order)."""
        result = self._parse("This is not simple, it is actually complex")
        assert result is not None
        assert result[0] == "simple"  # "simple" checked first in if/elif

    def test_parse_regex_fallback_confidence_out_of_regex_range(self):
        """Confidence value >1 in text doesn't match regex [01], defaults to 0.6."""
        result = self._parse("This is complex, confidence 2.5")
        assert result is not None
        label, conf, _ = result
        assert label == "complex"
        assert conf == 0.6  # regex doesn't match "2.5", falls back to default


class TestModelRouter:
    @pytest.mark.asyncio
    async def test_model_router_routes_simple_high_confidence_to_small(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.95,"reason":"easy"}')
        proc._bind_sub_harnesses({"small": sub})

        _events = await pipe_all(_task_start_event(state, "translate this sentence"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "small"
        assert slot.content["label"] == "simple"
        assert slot.content["source"] == "router_llm"
        assert sub.calls and sub.calls[0][1] == "r1"
        assert "MODEL_ROUTER_CLASSIFY" in sub.calls[0][0].description

    @pytest.mark.asyncio
    async def test_model_router_low_confidence_falls_back_to_complex_key(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor(confidence_threshold=0.7, complex_key="main")
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.3,"reason":"uncertain"}')
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "maybe hard maybe easy"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "main"
        assert slot.content["label"] == "simple"
        assert slot.content["source"] == "router_low_confidence"

    @pytest.mark.asyncio
    async def test_model_router_missing_router_harness_falls_back_to_complex_key(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor(router_key="small", complex_key="main")
        proc._bind_sub_harnesses({})

        _events = await pipe_all(_task_start_event(state, "any query"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "main"
        assert slot.content["source"] == "router_missing"

    @pytest.mark.asyncio
    async def test_model_router_reroutes_even_when_existing_slot_present(self):
        state = State(run_id="r1")
        state.set_slot(
            "model.route",
            "model_route",
            {"selected_key": "small", "source": "preset"},
        )
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"complex","confidence":0.99}')
        proc._bind_sub_harnesses({"small": sub})

        _events = await pipe_all(_task_start_event(state, "new query"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "main"
        assert slot.content["source"] == "router_llm"
        assert len(sub.calls) == 1

    @pytest.mark.asyncio
    async def test_model_router_clears_route_slot_on_task_end(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.95,"reason":"easy"}')
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "simple query"), [proc])
        assert state.get_slot("model.route") is not None

        await pipe_all(TaskEndEvent(run_id="r1", step_id=0, final_output="ok"), [proc])
        assert state.get_slot("model.route") is None

    @pytest.mark.asyncio
    async def test_harness_uses_model_route_slot_to_pick_small_provider(self):
        main = _FakeProvider(model="main-model", reply="from-main")
        small = _FakeProvider(model="small-model", reply="from-small")

        model = ModelConfig(main=main, small=small)
        config = HarnessConfig(
            tracer=NullTracer(),
            processors=[_RouteSlotProcessor("small")],
        )
        harness = model.agentic(config)

        result = await harness.run(BaseTask(description="route me", max_steps=1))

        assert result.final_output == "from-small"
        assert small.calls == 1
        assert main.calls == 0
        assert result.resume_state is not None
        assert result.resume_state.get_slot("model.route") is None

    @pytest.mark.asyncio
    async def test_harness_route_slot_unknown_key_falls_back_to_main_provider(self):
        main = _FakeProvider(model="main-model", reply="from-main")
        small = _FakeProvider(model="small-model", reply="from-small")

        model = ModelConfig(main=main, small=small)
        config = HarnessConfig(
            tracer=NullTracer(),
            processors=[_RouteSlotProcessor("unknown-role")],
        )
        harness = model.agentic(config)

        result = await harness.run(BaseTask(description="route me", max_steps=1))

        assert result.final_output == "from-main"
        assert main.calls == 1
        assert small.calls == 0
        assert result.resume_state is not None
        assert result.resume_state.get_slot("model.route") is None

    @pytest.mark.asyncio
    async def test_harness_respects_model_router_custom_slot_key(self):
        main = _FakeProvider(model="main-model", reply="from-main")
        small = _RouterAndAnswerProvider(model="small-model")

        model = ModelConfig(main=main, small=small)
        config = HarnessConfig(
            tracer=NullTracer(),
            processors=[ModelRouterProcessor(slot_key="model.route.custom")],
        )
        harness = model.agentic(config)

        result = await harness.run(BaseTask(description="route me", max_steps=1))

        assert result.final_output == "answer-small-model"
        assert small.calls >= 2  # router sub-run + main run step
        assert main.calls == 0
        assert result.resume_state is not None
        assert result.resume_state.get_slot("model.route.custom") is None

    # ---------------------------------------------------------------------------
    # A. Parser tests — _parse_classifier_output
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # B. Router decision edge-case tests
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_model_router_error_falls_back_to_complex_key(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness(exc=RuntimeError("boom"))
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "any query"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "main"
        assert slot.content["source"] == "router_error"
        assert "boom" in slot.content["reason"]

    @pytest.mark.asyncio
    async def test_model_router_parse_error_falls_back_to_complex_key(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness("gibberish without any keywords here")
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "any query"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "main"
        assert slot.content["source"] == "router_parse_error"

    @pytest.mark.asyncio
    async def test_model_router_confidence_at_threshold_routes_normally(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor(confidence_threshold=0.7)
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.7,"reason":"borderline"}')
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "a query"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "small"
        assert slot.content["source"] == "router_llm"
        assert slot.content["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_model_router_complex_high_confidence_routes_to_main(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"complex","confidence":0.95,"reason":"hard task"}')
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "analyze this architecture"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "main"
        assert slot.content["label"] == "complex"
        assert slot.content["source"] == "router_llm"

    @pytest.mark.asyncio
    async def test_model_router_custom_simple_complex_keys(self):
        state = State(run_id="r1")
        proc = ModelRouterProcessor(
            router_key="fast",
            simple_key="fast",
            complex_key="powerful",
        )
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.9}')
        proc._bind_sub_harnesses({"fast": sub})

        await pipe_all(_task_start_event(state, "easy task"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "fast"
        assert slot.content["router_key"] == "fast"

    @pytest.mark.asyncio
    async def test_model_router_no_state_yields_event_only(self):
        proc = ModelRouterProcessor()
        proc._bind_sub_harnesses({"small": _FakeSubHarness()})

        event = TaskStartEvent(
            run_id="r1",
            step_id=0,
            task_description="hello",
            state=None,
        )
        events = await pipe_all(event, [proc])

        assert len(events) == 1
        assert isinstance(events[0], TaskStartEvent)
        triggers = _triggers(events)
        assert len(triggers) == 0

    # ---------------------------------------------------------------------------
    # C. Integration test — real ModelRouterProcessor inside Harness
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_harness_full_model_router_processor_routes_simple_to_small(self):
        main = _FakeProvider(model="main-model", reply="from-main")
        small = _RouterAndAnswerProvider(model="small-model")

        model = ModelConfig(main=main, small=small)
        config = HarnessConfig(
            tracer=NullTracer(),
            processors=[ModelRouterProcessor()],
        )
        harness = model.agentic(config)

        result = await harness.run(BaseTask(description="say hello", max_steps=1))

        assert result.final_output == "answer-small-model"
        assert small.calls >= 2  # classifier sub-run + main answer step
        assert main.calls == 0
        assert result.resume_state is not None
        assert result.resume_state.get_slot("model.route") is None

    # ===========================================================================
    # D. Extended edge-case & boundary tests (audit coverage gaps)
    # ===========================================================================

    # ---- D1. Decision logic: edge cases ----

    @pytest.mark.asyncio
    async def test_model_router_confidence_clamped_above_one(self):
        """Confidence > 1.0 from classifier should be clamped to 1.0."""
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":1.5}')
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "test"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_model_router_confidence_clamped_below_zero(self):
        """Negative confidence from classifier should be clamped to 0.0."""
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":-0.5}')
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "test"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["confidence"] == 0.0
        # Negative confidence < threshold → fallback to complex_key
        assert slot.content["selected_key"] == "main"
        assert slot.content["source"] == "router_low_confidence"

    @pytest.mark.asyncio
    async def test_model_router_none_final_output_yields_parse_error(self):
        """Sub-harness returning final_output=None triggers router_parse_error."""
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness(output="")  # empty string simulates None-ish output
        # Patch to return None explicitly
        original_run = sub.run

        async def _run_none(task, *, parent_run_id=None):
            result = await original_run(task, parent_run_id=parent_run_id)
            result.final_output = None
            return result

        sub.run = _run_none
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "query"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["source"] == "router_parse_error"
        assert slot.content["selected_key"] == "main"

    @pytest.mark.asyncio
    async def test_model_router_empty_task_description(self):
        """Empty / whitespace-only task_description should not crash the router."""
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.9}')
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "   "), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        # Router should still work; the prompt just has empty query section
        assert sub.calls and "[MODEL_ROUTER_CLASSIFY]" in sub.calls[0][0].description

    @pytest.mark.asyncio
    async def test_model_router_parse_error_reason_truncated_to_200(self):
        """Long unparseable output should have reason truncated to 200 chars."""
        state = State(run_id="r1")
        proc = ModelRouterProcessor()
        long_output = "x" * 500
        sub = _FakeSubHarness(long_output)
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state, "query"), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["source"] == "router_parse_error"
        assert len(slot.content["reason"]) == 200

    # ---- D2. Slot lifecycle: edge cases ----

    @pytest.mark.asyncio
    async def test_model_router_task_end_unregistered_run_id_is_noop(self):
        """on_task_end with a run_id that was never started should not error."""
        proc = ModelRouterProcessor()
        proc._bind_sub_harnesses({"small": _FakeSubHarness()})

        events = await pipe_all(
            TaskEndEvent(run_id="never-started", step_id=0, final_output="ok"),
            [proc],
        )
        # Should yield the event without error
        assert len(events) == 1
        assert isinstance(events[0], TaskEndEvent)

    @pytest.mark.asyncio
    async def test_model_router_task_end_mismatched_run_id_leaves_other_slot(self):
        """on_task_end for run_id B should not affect run_id A's slot."""
        state_a = State(run_id="a")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.9}')
        proc._bind_sub_harnesses({"small": sub})

        # Start task A
        await pipe_all(_task_start_event(state_a, "query a"), [proc])
        assert state_a.get_slot("model.route") is not None

        # End task B (never started)
        await pipe_all(TaskEndEvent(run_id="b", step_id=0, final_output="ok"), [proc])

        # Task A's slot should still be there
        assert state_a.get_slot("model.route") is not None

    @pytest.mark.asyncio
    async def test_model_router_duplicate_run_id_overwrites_state(self):
        """Second task_start with same run_id overwrites the state ref."""
        state1 = State(run_id="dup")
        state2 = State(run_id="dup")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.9}')
        proc._bind_sub_harnesses({"small": sub})

        await pipe_all(_task_start_event(state1, "first"), [proc])
        await pipe_all(_task_start_event(state2, "second"), [proc])

        # Both states have slots
        assert state1.get_slot("model.route") is not None
        assert state2.get_slot("model.route") is not None

        # task_end cleans up the second (latest) state
        await pipe_all(TaskEndEvent(run_id="dup", step_id=0, final_output="ok"), [proc])
        assert state2.get_slot("model.route") is None

    @pytest.mark.asyncio
    async def test_model_router_concurrent_tasks_isolated(self):
        """Two concurrent tasks on the same processor should have isolated slots."""
        import asyncio

        state1 = State(run_id="c1")
        state2 = State(run_id="c2")
        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.9}')
        proc._bind_sub_harnesses({"small": sub})

        # Run both task_starts concurrently
        await asyncio.gather(
            pipe_all(_task_start_event(state1, "task one"), [proc]),
            pipe_all(_task_start_event(state2, "task two"), [proc]),
        )

        assert state1.get_slot("model.route") is not None
        assert state2.get_slot("model.route") is not None

        # End task 1; task 2's slot should survive
        await pipe_all(TaskEndEvent(run_id="c1", step_id=0, final_output="ok"), [proc])
        assert state1.get_slot("model.route") is None
        assert state2.get_slot("model.route") is not None

        # End task 2
        await pipe_all(TaskEndEvent(run_id="c2", step_id=0, final_output="ok"), [proc])
        assert state2.get_slot("model.route") is None

    @pytest.mark.asyncio
    async def test_model_router_state_gc_before_task_end(self):
        """If State is GC'd before task_end, the end handler should not error."""
        import gc
        import weakref

        proc = ModelRouterProcessor()
        sub = _FakeSubHarness('{"complexity":"simple","confidence":0.9}')
        proc._bind_sub_harnesses({"small": sub})

        state = State(run_id="gc-test")
        await pipe_all(_task_start_event(state, "query"), [proc])
        assert state.get_slot("model.route") is not None

        # Drop the strong ref and force GC
        _weak = weakref.ref(state)
        del state
        gc.collect()

        # WeakValueDictionary entry may be gone; task_end should still be safe
        events = await pipe_all(
            TaskEndEvent(run_id="gc-test", step_id=0, final_output="ok"),
            [proc],
        )
        assert len(events) == 1

    # ---- D3. Harness _select_model_provider edge cases ----

    @pytest.mark.asyncio
    async def test_harness_route_slot_non_dict_content_falls_back_to_main(self):
        """slot.content that is not a dict should be ignored, falling back to main."""
        main = _FakeProvider(model="main-model", reply="from-main")
        small = _FakeProvider(model="small-model", reply="from-small")

        class _BadSlotProcessor(MultiHookProcessor):
            _singleton_group = "test_bad_slot"
            _order = 1

            async def on_task_start(self, event: TaskStartEvent):
                if event.state is not None:
                    event.state.set_slot("model.route", "model_route", "not-a-dict")
                yield event

        model = ModelConfig(main=main, small=small)
        config = HarnessConfig(
            tracer=NullTracer(),
            processors=[_BadSlotProcessor()],
        )
        harness = model.agentic(config)

        result = await harness.run(BaseTask(description="test", max_steps=1))
        assert result.final_output == "from-main"
        assert main.calls == 1

    @pytest.mark.asyncio
    async def test_harness_route_slot_empty_selected_key_falls_back_to_main(self):
        """selected_key='' should be treated as missing, falling back to main."""
        main = _FakeProvider(model="main-model", reply="from-main")
        small = _FakeProvider(model="small-model", reply="from-small")

        class _EmptyKeyProcessor(MultiHookProcessor):
            _singleton_group = "test_empty_key"
            _order = 1

            async def on_task_start(self, event: TaskStartEvent):
                if event.state is not None:
                    event.state.set_slot(
                        "model.route",
                        "model_route",
                        {"selected_key": "", "source": "test"},
                    )
                yield event

        model = ModelConfig(main=main, small=small)
        config = HarnessConfig(
            tracer=NullTracer(),
            processors=[_EmptyKeyProcessor()],
        )
        harness = model.agentic(config)

        result = await harness.run(BaseTask(description="test", max_steps=1))
        assert result.final_output == "from-main"
        assert main.calls == 1

    @pytest.mark.asyncio
    async def test_harness_cleanup_skips_slot_with_wrong_slot_type(self):
        """Post-run cleanup should NOT delete a slot if slot_type != 'model_route'."""
        main = _FakeProvider(model="main-model", reply="from-main")

        class _ForeignSlotProcessor(MultiHookProcessor):
            _singleton_group = "test_foreign_slot"
            _order = 1

            async def on_task_start(self, event: TaskStartEvent):
                if event.state is not None:
                    # Write a slot at "model.route" but with a different slot_type
                    event.state.set_slot("model.route", "slash_command", {"data": "keep-me"})
                yield event

        model = ModelConfig(main=main)
        config = HarnessConfig(
            tracer=NullTracer(),
            processors=[_ForeignSlotProcessor()],
        )
        harness = model.agentic(config)

        result = await harness.run(BaseTask(description="test", max_steps=1))

        # The slot should survive cleanup because slot_type != "model_route"
        slot = result.resume_state.get_slot("model.route")
        assert slot is not None
        assert slot.slot_type == "slash_command"
        assert slot.content["data"] == "keep-me"

    @pytest.mark.asyncio
    async def test_harness_multiple_route_slot_keys_second_valid(self):
        """When first route slot is invalid and second is valid, second one wins."""
        main = _FakeProvider(model="main-model", reply="from-main")
        small = _FakeProvider(model="small-model", reply="from-small")

        class _FirstBadSlotRouter(MultiHookProcessor):
            _singleton_group = "model_router"
            _order = 1

            def __init__(self):
                self.slot_key = "model.route.bad"

            async def on_task_start(self, event: TaskStartEvent):
                if event.state is not None:
                    event.state.set_slot("model.route.bad", "model_route", "not-a-dict")
                yield event

        class _SecondGoodSlotRouter(MultiHookProcessor):
            _singleton_group = "model_router"
            _order = 2

            def __init__(self):
                self.slot_key = "model.route.good"

            async def on_task_start(self, event: TaskStartEvent):
                if event.state is not None:
                    event.state.set_slot(
                        "model.route.good",
                        "model_route",
                        {"selected_key": "small", "source": "test"},
                    )
                yield event

        model = ModelConfig(main=main, small=small)
        config = HarnessConfig(
            tracer=NullTracer(),
            processors=[_FirstBadSlotRouter(), _SecondGoodSlotRouter()],
        )
        harness = model.agentic(config)

        result = await harness.run(BaseTask(description="test", max_steps=1))

        # Second valid slot should be picked — small provider answers
        assert result.final_output == "from-small"
        assert small.calls == 1
        assert main.calls == 0

    # ===========================================================================
    # E. Coverage gap tests — CLI short name, API file write, long input
    # ===========================================================================

    def test_harness_config_target_registers_model_router(self):
        """A full `_target_` router spec in HarnessConfig produces ModelRouterProcessor."""
        config = HarnessConfig(
            processors=[
                {
                    "_target_": "harnessx.processors.multi_model.model_router.ModelRouterProcessor",
                    "confidence_threshold": 0.8,
                }
            ]
        )

        from harnessx.core.harness import _instantiate_runtime

        all_procs = [p for procs in _instantiate_runtime(config).processors.values() for p in procs]
        routers = [p for p in all_procs if isinstance(p, ModelRouterProcessor)]
        assert len(routers) == 1
        assert routers[0].confidence_threshold == 0.8

    @pytest.mark.asyncio
    async def test_put_model_config_writes_file(self, tmp_path, monkeypatch):
        """PUT /api/model-config should write model_config.yaml to disk."""
        from harnessx.api.routes.model_config import (
            ModelConfigResponse,
            ModelDefItem,
            ModelSlotItem,
            put_model_config,
        )

        # Redirect agent_home() to tmp_path — patched at source since it's imported locally
        monkeypatch.setattr("harnessx.home.agent_home", lambda: tmp_path)

        req = ModelConfigResponse(
            registry=[
                ModelDefItem(
                    id="m-main",
                    display_name="Main",
                    vendor="anthropic",
                    model_id="main-model",
                ),
                ModelDefItem(
                    id="m-small",
                    display_name="Small",
                    vendor="anthropic",
                    model_id="small-model",
                ),
            ],
            slots=[
                ModelSlotItem(slot_name="main", model_ids=["m-main"]),
                ModelSlotItem(slot_name="small", model_ids=["m-small"]),
            ],
        )

        await put_model_config(req)

        written = tmp_path / "model_config.yaml"
        assert written.exists()
        text = written.read_text()
        assert "main" in text
        assert "small" in text
        assert "main-model" in text
        assert "small-model" in text

    @pytest.mark.asyncio
    async def test_model_router_long_input_does_not_crash(self):
        """A very long task description should not crash the router."""
        state = State(run_id="r1")
        proc = ModelRouterProcessor(router_token_budget=512)
        sub = _FakeSubHarness('{"complexity":"complex","confidence":0.85}')
        proc._bind_sub_harnesses({"small": sub})

        long_desc = "Analyze this: " + "word " * 2000  # ~2000 words ≈ ~2700 tokens
        await pipe_all(_task_start_event(state, long_desc), [proc])

        slot = state.get_slot("model.route")
        assert slot is not None
        assert slot.content["selected_key"] == "main"
        assert slot.content["source"] == "router_llm"
        # The prompt was forwarded to the sub-harness with token_budget set
        assert sub.calls
        assert sub.calls[0][0].token_budget == 512
