# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import re
import weakref
from typing import TYPE_CHECKING, Any

from ...core.events import TaskEndEvent, TaskStartEvent
from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from ...core.state import State

_ROUTER_PROMPT = """\
[MODEL_ROUTER_CLASSIFY]
You are a model routing classifier. Classify the user query complexity.

Rules:
- SIMPLE: short factual lookup, straightforward rewrite, basic translation,
  small formatting, or low-risk single-step tasks.
- COMPLEX: multi-step coding/debugging, architecture design, broad research,
  long-form synthesis, ambiguous requirements, or tasks likely needing tools.

Return strict JSON only:
{{"complexity":"simple|complex","confidence":0.0-1.0,"reason":"short reason"}}

User query:
{query}
"""


class ModelRouterProcessor(MultiHookProcessor):
    """Choose model key once at task_start and persist into state slot."""

    _singleton_group = "model_router"
    _order = 20

    def __init__(
        self,
        router_key: str = "small",
        simple_key: str = "small",
        complex_key: str = "main",
        slot_key: str = "model.route",
        confidence_threshold: float = 0.7,
        max_router_steps: int = 1,
        router_token_budget: int = 512,
        enabled: bool = True,
    ) -> None:
        self.router_key = router_key
        self.simple_key = simple_key
        self.complex_key = complex_key
        self.slot_key = slot_key
        self.confidence_threshold = confidence_threshold
        self.max_router_steps = max_router_steps
        self.router_token_budget = router_token_budget
        self.enabled = enabled
        self._run_states: weakref.WeakValueDictionary[str, State] = weakref.WeakValueDictionary()

    async def on_task_start(self, event: TaskStartEvent):
        if not self.enabled:
            yield event
            return
        state = event.state
        if state is None:
            yield event
            return

        self._run_states[event.run_id] = state

        decision = await self._decide(event)
        state.set_slot(self.slot_key, "model_route", decision)

        yield event

    async def on_task_end(self, event: TaskEndEvent):
        state = self._run_states.pop(event.run_id, None)
        if state is not None:
            state.delete_slot(self.slot_key)
        yield event

    async def _decide(self, event: TaskStartEvent) -> dict[str, Any]:
        router = self._sub_harnesses.get(self.router_key)
        if router is None:
            return self._decision(
                selected_key=self.complex_key,
                label="complex",
                confidence=0.0,
                source="router_missing",
                reason=f"sub_harness '{self.router_key}' not configured",
            )

        from ...core.harness import BaseTask

        prompt = _ROUTER_PROMPT.format(query=(event.task_description or "").strip())
        try:
            result = await router.run(
                BaseTask(
                    description=prompt,
                    max_steps=self.max_router_steps,
                    token_budget=self.router_token_budget,
                ),
                parent_run_id=event.run_id,
            )
        except Exception as exc:
            return self._decision(
                selected_key=self.complex_key,
                label="complex",
                confidence=0.0,
                source="router_error",
                reason=str(exc),
            )

        raw = (result.final_output or "").strip()
        parsed = self._parse_classifier_output(raw)
        if parsed is None:
            return self._decision(
                selected_key=self.complex_key,
                label="complex",
                confidence=0.0,
                source="router_parse_error",
                reason=raw[:200],
            )

        label, confidence, reason = parsed
        if confidence < self.confidence_threshold:
            return self._decision(
                selected_key=self.complex_key,
                label=label,
                confidence=confidence,
                source="router_low_confidence",
                reason=reason,
            )

        selected_key = self.simple_key if label == "simple" else self.complex_key
        return self._decision(
            selected_key=selected_key,
            label=label,
            confidence=confidence,
            source="router_llm",
            reason=reason,
        )

    def _decision(
        self,
        selected_key: str,
        label: str,
        confidence: float,
        source: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "selected_key": selected_key,
            "label": label,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "source": source,
            "reason": reason,
            "router_key": self.router_key,
            "version": 1,
        }

    def _parse_classifier_output(self, text: str) -> tuple[str, float, str] | None:
        if not text:
            return None

        # 1) strict JSON / fenced JSON
        payload: dict[str, Any] | None = None
        try:
            payload = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    payload = json.loads(m.group(0))
                except Exception:
                    payload = None

        if isinstance(payload, dict):
            raw_label = (
                str(payload.get("complexity") or payload.get("label") or payload.get("route") or "").strip().lower()
            )
            if raw_label in ("simple", "complex"):
                try:
                    conf = float(payload.get("confidence", 1.0))
                except Exception:
                    conf = 1.0
                reason = str(payload.get("reason", "")).strip()
                return raw_label, conf, reason

        # 2) regex fallback
        ltext = text.lower()
        if "simple" in ltext:
            label = "simple"
        elif "complex" in ltext:
            label = "complex"
        else:
            return None

        cm = re.search(r"confidence[^0-9]*([01](?:\.\d+)?)", ltext)
        conf = float(cm.group(1)) if cm else 0.6
        return label, conf, text[:120]
