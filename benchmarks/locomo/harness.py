"""make_locomo_harness — factory for the LoCoMo evaluation harness."""

from __future__ import annotations

import os
from datetime import datetime

from harnessx.core.events import Message
from harnessx.core.harness import Harness
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.processors.context.user_wrapper import UserWrapperProcessor
from harnessx.processors.control.token_budget import TokenBudgetProcessor
from harnessx.processors.memory.memory_retrieval import MemoryRetrievalProcessor
from harnessx.processors.memory.strategies.base import BaseMemory, compress_by_token_budget
from harnessx.processors.memory.strategies.custom import InMemoryMemory
from harnessx.processors.memory.strategies.policy import AlwaysPolicy, MemoryPolicy
from harnessx.providers.litellm_provider import LiteLLMProvider
from harnessx.tracing.journal import HarnessJournal

_LOCOMO_SYSTEM = """\
You are a helpful assistant answering questions based on retrieved memory records \
from past conversations between two people.

Before answering, reason through these steps internally (do NOT output the steps):

1. Identify every memory that could relate to the question, including those with relative time phrases.
2. Cross-memory reasoning: if multiple memories describe the same person or topic, combine their \
information. If one memory uses a vague reference ("a friend", "her job") and another provides the \
specific name, substitute it. Pay attention to WHO did or said what — the memories involve two different \
speakers.
3. Time resolution: memory dates in the header indicate WHEN it was discussed, not necessarily when it \
happened. Resolve relative time expressions ("yesterday", "last week") using the memory's session date. \
Distinguish similar events at different dates.
4. If the memories do not contain even tangentially related information, respond with \
"I don't have that information".

Answer format:
- State the answer directly and concisely (a name, date, or short phrase)
- Always prefer specific numbers, names, and dates over vague descriptions
- If the question asks "what" and there are multiple items, list ALL of them
- Do NOT restate the question or add explanation"""


class _StaticSystemPromptBuilder:
    """Minimal system prompt builder that returns a fixed string."""

    def __init__(self, text: str):
        self._text = text

    async def build(self, workspace=None) -> str:
        return self._text


class LightMemoryBackend:
    """``BaseMemory`` adapter that reads from the light-memory file store.

    Ingestion (``add``) is a no-op; writes happen via the compressor before QA.

    Retrieval uses either:
    - Rule-based keyword + decay ranking (``call_llm=None``, default)
    - Two-stage LLM recall: query expansion → grep → LLM rerank (``call_llm`` set)
    """

    def __init__(
        self,
        cfg: object,
        eval_now: datetime | None = None,
        call_llm=None,
    ) -> None:
        self._cfg = cfg
        self._eval_now = eval_now
        self._call_llm = call_llm

    def set_eval_now(self, dt: datetime) -> None:
        self._eval_now = dt

    async def add(self, messages: list[Message], **_kwargs) -> None:
        pass  # writes handled by the compressor

    async def retrieve(self, query: str, k: int = 10) -> list[Message]:
        if self._call_llm is not None:
            from harnessx.plugins.dimensions.light_memory._core.lifecycle import (
                read_recalled_memories_with_llm,
            )

            text = await read_recalled_memories_with_llm(self._cfg, query, self._call_llm, now=self._eval_now)
        else:
            from harnessx.plugins.dimensions.light_memory._core.lifecycle import (
                read_recalled_memories,
            )

            text = read_recalled_memories(self._cfg, query, now=self._eval_now)
        if not text.strip():
            return []
        return [Message(role="user", content=text)]

    async def compress(self, messages: list[Message], budget: int) -> list[Message]:
        return compress_by_token_budget(messages, budget)

    async def persist(self) -> None:
        from harnessx.plugins.dimensions.light_memory._core.backend import (
            get_all_memory_documents,
        )
        from harnessx.plugins.dimensions.light_memory._core.index_file import (
            generate_index_file,
        )

        docs = get_all_memory_documents(self._cfg)
        generate_index_file(self._cfg, docs)

    async def load(self, run_id: str) -> list[Message]:
        return []


def _make_provider(
    model: str,
    extended_thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    extra_headers: dict | None = None,
):
    """Create the best provider for the given model string."""
    if os.environ.get("ANTHROPIC_BASE_URL"):
        from harnessx.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=model,
            max_tokens=4096,
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
    headers = {"X-Model-Provider-Id": "YOUR_PROVIDER_ID"}
    if extra_headers:
        headers.update(extra_headers)
    return LiteLLMProvider(model=model, extra_headers=headers)


def make_locomo_harness(
    model: str = "openai/pa/claude-sonnet-4-6",
    extra_headers: dict | None = None,
    memory: "BaseMemory | None" = None,
    memory_policy: MemoryPolicy | None = None,
    verbose: bool = False,
    extended_thinking: bool = False,
    thinking_budget_tokens: int = 8000,
) -> "tuple[Harness, BaseMemory]":
    """Build a Harness configured for LoCoMo QA evaluation.

    Returns both the harness and the memory backend so the caller can
    pass the same memory instance to ``SessionIngester.ingest()``.

    Args:
        model:          LLM to use for answering questions
        memory:         memory backend; defaults to InMemoryMemory(max_messages=2000)
        memory_policy:  retrieval/compress/store policy; defaults to AlwaysPolicy
        verbose:        enable structured trace logging

    Returns:
        (harness, memory) tuple
    """
    if memory is None:
        memory = InMemoryMemory(max_messages=2000)
    if memory_policy is None:
        memory_policy = AlwaysPolicy()

    from harnessx.core.builder import HarnessBuilder
    from harnessx.core.model_config import ModelConfig

    provider = _make_provider(
        model,
        extended_thinking=extended_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        extra_headers=extra_headers,
    )

    builder = (
        HarnessBuilder()
        .add(SystemPromptProcessor(_StaticSystemPromptBuilder(_LOCOMO_SYSTEM)))
        .add(MemoryRetrievalProcessor(memory, memory_policy=memory_policy, top_k=30))
        .add(TokenBudgetProcessor())
        .add(UserWrapperProcessor())
    )
    if verbose:
        builder = builder.slot(tracer=HarnessJournal(agent_id="locomo_agent"))

    harness = ModelConfig(main=provider).agentic(builder.build())
    return harness, memory
