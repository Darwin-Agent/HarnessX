"""
HarnessX agent adapter for tau2-bench.

Implements tau2's HalfDuplexAgent interface.  Each call to
``generate_next_message`` runs ``Harness.run()`` with the full processor
pipeline (ContextAssemblyProcessor, StructlogTracer, …).

How the per-turn / full-run-loop bridge works
---------------------------------------------
tau2 drives the outer loop: it calls ``generate_next_message`` once per
agent turn (user message or tool results), and it executes tool calls itself.
HarnessX's ``Harness.run()`` also wants to own a loop.

We reconcile the two loops via **interrupt_on + resume_state**:

1. ``generate_next_message(message, state)`` is called by tau2.
2. We append the incoming message to the OH ``State`` kept in ``AgentState``.
3. We call ``Harness.run(task, resume_state=oh_state)`` where
   ``task.interrupt_on`` lists every tool name the environment offers.
4. Inside ``Harness.run()``:
   - ContextAssemblyProcessor assembles context (memory, history truncation).
   - LiteLLMProvider calls the model.
   - If the model emits a tool call → ``run_loop`` hits ``interrupt_on`` and
     exits immediately (no tool is executed by the Harness).
   - If the model emits plain text → ``run_loop`` exits normally.
5. On interrupt: we convert the OH ``AssistantMessage`` (already stored in the
   OH state by ``run_loop``) to a tau2 ``AssistantMessage`` with tool calls and
   return it to tau2.  tau2 executes the tool and calls us again with the
   result(s).
6. On normal exit: we return the plain-text ``AssistantMessage`` to tau2.
7. We save ``result.resume_state`` back into ``AgentState.oh_state`` so the
   next call can resume seamlessly.

HarnessJournal writes per-run JSONL logs to ``runs/{run_id}/``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from harnessx.core.events import (
        Message as OHMessage,
        ToolSchema,
    )
    from harnessx.core.harness import HarnessConfig

logger = logging.getLogger(__name__)

# ─── Async/sync bridge ──────────────────────────────────────────────────────

_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# asyncio.run() creates a new event loop per call.  When the loop closes,
# httpx/anyio async clients attempt aclose() and raise
# "RuntimeError: Event loop is closed".  These are printed as unraisable
# exceptions and "Task exception was never retrieved" warnings — pure noise
# that does not affect results.  Suppress them here once at import time.
import sys as _sys

_orig_unraisablehook = _sys.unraisablehook


def _quiet_unraisable(exc_info: Any) -> None:
    msg = str(exc_info.exc_value)
    if isinstance(exc_info.exc_value, RuntimeError) and (
        "event loop" in msg.lower() or "no running event loop" in msg.lower()
    ):
        return
    _orig_unraisablehook(exc_info)


_sys.unraisablehook = _quiet_unraisable

import logging as _logging

_logging.getLogger("asyncio").setLevel(_logging.CRITICAL)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous code.

    Handles the case where we're already inside an event loop
    (e.g. tau2's ThreadPoolExecutor-based batch runner).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    future = _THREAD_POOL.submit(asyncio.run, coro)
    return future.result()


# ─── Agent state ─────────────────────────────────────────────────────────────


@dataclass
class AgentState:
    """Conversation state for the HarnessX adapter.

    ``system_messages`` and ``messages`` mirror tau2's expected state shape so
    ``get_init_state`` satisfies the interface contract.

    ``oh_state`` is the live HarnessX ``State`` that accumulates the full
    conversation and is passed as ``resume_state`` to each ``Harness.run()``
    call.
    """

    system_messages: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    oh_state: Any = field(default=None, repr=False)  # harnessx.core.state.State


# ─── Message conversion ─────────────────────────────────────────────────────


def _tau2_tool_to_oh_schema(tool: Any) -> "ToolSchema":
    """Convert a tau2 Tool to HarnessX ToolSchema."""
    from harnessx.core.events import ToolSchema

    schema = tool.openai_schema
    func = schema["function"]
    return ToolSchema(
        name=func["name"],
        description=func.get("description", ""),
        input_schema=func.get("parameters", {"type": "object", "properties": {}}),
    )


def _build_tau2_tool_registry(tools: list) -> Any:
    """Build an InMemoryToolRegistry with stub tau2 tools.

    The model needs tool schemas in the run_loop's StepStartEvent to know
    what tools are available.  tau2 executes tools itself — the Harness
    exits via ``interrupt_on`` before calling ``registry.execute()``.
    The stub functions are therefore never actually invoked.
    """
    from harnessx.tools.base import Tool
    from harnessx.tools.inmemory import InMemoryToolRegistry

    registry = InMemoryToolRegistry()
    for t in tools:
        schema = t.openai_schema
        func = schema["function"]
        name = func["name"]
        description = func.get("description", "")
        input_schema = func.get("parameters", {"type": "object", "properties": {}})

        def _make_stub(n: str):
            def _stub(**_kwargs: Any) -> str:  # pragma: no cover
                raise RuntimeError(f"Tool {n!r} stub called — should be intercepted by interrupt_on")

            _stub.__name__ = n
            return _stub

        registry.register(
            Tool(
                name=name,
                description=description,
                input_schema=input_schema,
                fn=_make_stub(name),
            )
        )
    return registry


def _tau2_messages_to_oh(messages: list) -> "list[OHMessage]":
    """Convert a list of tau2 messages to HarnessX Message objects.

    Handles: SystemMessage, UserMessage, AssistantMessage,
             ToolMessage, MultiToolMessage.
    """
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolMessage,
        UserMessage,
    )
    from harnessx.core.events import Message, ToolCall

    oh_messages: list[Message] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            oh_messages.append(Message(role="system", content=msg.content or ""))

        elif isinstance(msg, UserMessage):
            oh_messages.append(Message(role="user", content=msg.content or ""))

        elif isinstance(msg, AssistantMessage):
            tc_tuple: tuple = ()
            if msg.tool_calls:
                tc_tuple = tuple(ToolCall(id=tc.id, name=tc.name, input=tc.arguments) for tc in msg.tool_calls)
            oh_messages.append(
                Message(
                    role="assistant",
                    content=msg.content or "",
                    tool_calls=tc_tuple,
                )
            )

        elif isinstance(msg, ToolMessage):
            oh_messages.append(
                Message(
                    role="tool",
                    content=msg.content or "",
                    tool_call_id=msg.id,
                )
            )

        elif isinstance(msg, MultiToolMessage):
            for tm in msg.tool_messages:
                oh_messages.append(
                    Message(
                        role="tool",
                        content=tm.content or "",
                        tool_call_id=tm.id,
                    )
                )

        else:
            logger.warning("Unknown tau2 message type: %s, skipping", type(msg).__name__)

    return oh_messages


# ─── Agent adapter ───────────────────────────────────────────────────────────


def _build_agent_class() -> type:
    """Build the HarnessXAgent class with proper tau2 inheritance.

    Creates the class at import time so isinstance() checks and
    super().__init__() work correctly.  Falls back to a plain class
    if tau2 is not installed.
    """
    try:
        from tau2.agent.base_agent import HalfDuplexAgent

        base_classes: tuple = (HalfDuplexAgent,)
    except ImportError:
        base_classes = (object,)

    class _HarnessXAgent(*base_classes):
        """tau2 HalfDuplexAgent that runs each turn through Harness.run().

        Tool calls are handled by tau2 (via interrupt_on); the Harness
        manages context assembly, memory, tracing, and the run-loop itself.
        """

        def __init__(
            self,
            tools: list,
            domain_policy: str,
            model: str = "gpt-4.1",
            api_base: Optional[str] = None,
            api_key: str = "EMPTY",
            logs_dir: str = "runs",
            policy_hints: bool = False,
            harness_config: str = "harness_config.yaml",
            extended_thinking: bool = False,
            thinking_budget_tokens: int = 8000,
            request_timeout: float | None = None,
            **_kwargs: Any,
        ):
            super().__init__(tools=tools, domain_policy=domain_policy)

            self._model = model
            self._api_base = api_base
            self._api_key = api_key

            # Build a tool registry with stub tools so the run_loop exposes
            # correct schemas to the model via StepStartEvent.tools.
            # The stubs are never executed: interrupt_on fires first.
            _registry = _build_tau2_tool_registry(tools)

            # Build Harness pipeline from YAML config
            import yaml as _yaml
            from pathlib import Path as _Path
            from harnessx.core.builder import build_from_config
            from harnessx.core.harness import Harness
            from harnessx.core.model_config import ModelConfig
            from harnessx.tracing.journal import HarnessJournal
            from .harness import make_tau2_provider

            _raw = _yaml.safe_load((_Path(__file__).parent / harness_config).read_text())
            _config = build_from_config(_raw).copy(
                tool_registry=_registry,
                tracer=HarnessJournal(base_dir=logs_dir, export_jsonl=True),
            )
            if policy_hints:
                from .policy_hint import PolicyHintProcessor as _PHProc

                _rt = getattr(_config, "_rt_procs", None) or []
                _already = any(isinstance(p, _PHProc) for p in _rt)
                if not _already:
                    _config = _config.copy()
                    _config._rt_procs = list(getattr(_config, "_rt_procs", []))
                    _config._rt_procs.append(_PHProc())
            self._config: HarnessConfig = _config
            _provider = make_tau2_provider(
                model=model,
                api_base=api_base,
                api_key=api_key,
                extended_thinking=extended_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                timeout=request_timeout,
            )
            _model_config = ModelConfig(main=_provider)
            self._harness = Harness(_model_config, self._config)

            # Collect tool names for interrupt_on
            self._tool_names: list[str] = [s.name for s in _registry.get_schemas()]

        def get_init_state(
            self,
            message_history: Optional[list] = None,
        ) -> AgentState:
            """Build initial conversation state.

            Creates the OH State and pre-loads the system prompt so that the
            first ``Harness.run()`` call sees a properly primed context.
            """
            from tau2.data_model.message import SystemMessage
            from harnessx.core.state import State
            from harnessx.core.events import Message, make_run_id

            # Match the official tau2 LLMAgent system prompt format exactly
            # (see tau2/agent/llm_agent.py: AGENT_INSTRUCTION + SYSTEM_PROMPT)
            agent_instruction = (
                "You are a customer service agent that helps the user according to "
                "the <policy> provided below.\n"
                "In each turn you can either:\n"
                "- Send a message to the user.\n"
                "- Make a tool call.\n"
                "You cannot do both at the same time.\n\n"
                "Try to be helpful and always follow the policy. "
                "Always make sure you generate valid JSON only."
            )
            system_prompt = (
                f"<instructions>\n{agent_instruction}\n</instructions>\n<policy>\n{self.domain_policy}\n</policy>"
            )

            # OH State — carries the full conversation for resume_state
            oh_state = State(
                run_id=make_run_id(),
                max_steps=200,  # tau2 caps actual steps; this is just a ceiling
            )
            oh_state.add_message(Message(role="system", content=system_prompt))

            # tau2-compatible fields
            tau2_sys = SystemMessage(role="system", content=system_prompt)
            return AgentState(
                system_messages=[tau2_sys],
                messages=list(message_history) if message_history else [],
                oh_state=oh_state,
            )

        def generate_next_message(
            self,
            message: Any,
            state: AgentState,
        ) -> tuple[Any, AgentState]:
            """Generate the next response via Harness.run() + interrupt_on.

            The incoming tau2 message (user turn or tool results) is appended
            to the OH State, then ``Harness.run()`` executes one logical step:

            * **Tool call** → run_loop hits interrupt_on, exits early.  We
              extract the AssistantMessage from the OH State and return it to
              tau2 so tau2 can execute the tool.
            * **Text response** → run_loop exits normally.  We return the
              text AssistantMessage to tau2.

            In both cases we save ``result.resume_state`` back into
            ``state.oh_state`` so the next call continues seamlessly.
            """
            from tau2.data_model.message import (
                AssistantMessage,
                MultiToolMessage,
                ToolCall as Tau2ToolCall,
            )
            from harnessx.core.harness import BaseTask

            # ── 1. Mirror tau2 state (for compat) ────────────────────────────
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)

            # ── 2. Inject incoming message into OH State ──────────────────────
            oh_msgs = _tau2_messages_to_oh(
                message.tool_messages if isinstance(message, MultiToolMessage) else [message]
            )
            for m in oh_msgs:
                state.oh_state.add_message(m)

            # ── 3. Build task: interrupt_on = every domain tool ───────────────
            task = BaseTask(
                description="",  # system msg already in oh_state
                max_steps=200,  # per-run ceiling; actual limit is tau2's
                interrupt_on=self._tool_names,
            )

            # ── 4. Run the Harness (full run_loop with processors + tracer) ───
            result = _run_async(self._harness.run(task, _resume_state=state.oh_state))
            state.oh_state = result.resume_state

            # ── 5. Convert Harness result → tau2 AssistantMessage ─────────────
            if result.is_interrupted:
                # Model made a tool call(s).  The AssistantMessage with ALL
                # tool_calls was already added to oh_state by run_loop.
                # Find it and convert; tau2 will execute the tools.
                last_assistant_oh = next(
                    (m for m in reversed(state.oh_state.messages) if m.role == "assistant"),
                    None,
                )
                tau2_tool_calls: list[Tau2ToolCall] = []
                if last_assistant_oh and last_assistant_oh.tool_calls:
                    tau2_tool_calls = [
                        Tau2ToolCall(id=tc.id, name=tc.name, arguments=tc.input) for tc in last_assistant_oh.tool_calls
                    ]
                # tau2 convention: content=None when tool_calls present
                assistant_msg = AssistantMessage.text(
                    content=None,
                    tool_calls=tau2_tool_calls or None,
                )
            else:
                # Plain text response — run_loop already appended it to oh_state.
                final_text = result.task_end.final_output or ""
                assistant_msg = AssistantMessage.text(content=final_text, tool_calls=None)

            state.messages.append(assistant_msg)
            return assistant_msg, state

    _HarnessXAgent.__name__ = "HarnessXAgent"
    _HarnessXAgent.__qualname__ = "HarnessXAgent"
    return _HarnessXAgent


HarnessXAgent = _build_agent_class()


# ─── Factory function for tau2 registry ──────────────────────────────────────


def create_harnessx_agent(
    tools: list,
    domain_policy: str,
    **kwargs: Any,
) -> HarnessXAgent:
    """Factory function compatible with tau2 registry.register_agent_factory().

    Kwargs passed by tau2's build_agent:
        llm (str): model name
        llm_args (dict): may contain 'api_base', 'api_key'
        task: tau2 Task object (unused)
        solo_mode (bool): unused
    """
    llm_args = kwargs.get("llm_args") or {}
    return HarnessXAgent(
        tools=tools,
        domain_policy=domain_policy,
        model=kwargs.get("llm", "gpt-4.1"),
        api_base=llm_args.get("api_base"),
        api_key=llm_args.get("api_key", "EMPTY"),
        policy_hints=llm_args.get("policy_hints", False),
        harness_config=llm_args.get("harness_config", "harness_config.yaml"),
        extended_thinking=llm_args.get("extended_thinking", False),
        thinking_budget_tokens=llm_args.get("thinking_budget_tokens", 8000),
        request_timeout=llm_args.get("request_timeout"),
        logs_dir=llm_args.get("logs_dir", "runs"),
    )
