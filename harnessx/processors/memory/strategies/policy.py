# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ....core.events import Message, _extract_text, rough_token_count

# Keywords that signal a knowledge-seeking query
_RETRIEVAL_TRIGGERS = frozenset(
    {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "whose",
        "tell me",
        "do you remember",
        "recall",
        "remind",
        "what was",
        "what did",
        "when did",
        "where did",
        "who did",
        "have you",
        "did you",
        "did we",
    }
)


@runtime_checkable
class MemoryPolicy(Protocol):
    """Decide at each step which memory operations to perform.

    All three methods receive lightweight scalars so policies stay
    stateless and easy to test without a full ``State`` object.

    Parameters shared across methods
    ---------------------------------
    query         : text of the last user message (empty string if none)
    token_count   : token count of the current assembled context
    context_window: model's context window size (from HarnessConfig)
    messages      : the current in-session message list (before compression)
    budget        : token budget for in-session history
    new_messages  : messages produced during this step (to be stored)
    """

    async def should_retrieve(self, query: str, token_count: int, context_window: int) -> bool: ...

    async def should_compress(self, messages: list[Message], budget: int) -> bool: ...

    async def should_store(self, new_messages: list[Message]) -> bool: ...


class AlwaysPolicy:
    """Always retrieve and store; compress only when over token budget.

    Equivalent to the original ``ContextAssemblyProcessor`` behavior,
    except retrieval now uses the actual query instead of an empty string.
    """

    async def should_retrieve(self, query: str, token_count: int, context_window: int) -> bool:
        return True

    async def should_compress(self, messages: list[Message], budget: int) -> bool:
        return rough_token_count(messages) > budget

    async def should_store(self, new_messages: list[Message]) -> bool:
        return True


class RelevancePolicy:
    """Gate operations on content relevance.

    Retrieval: only when the query looks like a knowledge-seeking question
    (contains a question mark, trigger words, or is long enough to be substantive).

    Storage: only when new messages contain substantive text (skips empty
    responses, short acks, and pure tool-call metadata).
    """

    def __init__(self, min_query_tokens: int = 4, min_store_words: int = 6):
        self.min_query_tokens = min_query_tokens
        self.min_store_words = min_store_words

    async def should_retrieve(self, query: str, token_count: int, context_window: int) -> bool:
        if not query:
            return False
        q = query.lower()
        if "?" in q:
            return True
        if len(q.split()) < self.min_query_tokens:
            return False
        return any(trigger in q for trigger in _RETRIEVAL_TRIGGERS)

    async def should_compress(self, messages: list[Message], budget: int) -> bool:
        return rough_token_count(messages) > budget

    async def should_store(self, new_messages: list[Message]) -> bool:
        for m in new_messages:
            if m.role not in ("user", "assistant"):
                continue
            text = _extract_text(m.content)
            if len(text.split()) >= self.min_store_words:
                return True
        return False


class EvalReadOnlyPolicy:
    """Retrieve always; never store.

    Intended for benchmark evaluation where pre-ingested history should be
    available for retrieval, but QA answers must not pollute memory between
    tasks that share the same memory instance.
    """

    async def should_retrieve(self, query: str, token_count: int, context_window: int) -> bool:
        return True

    async def should_compress(self, messages: list[Message], budget: int) -> bool:
        return rough_token_count(messages) > budget

    async def should_store(self, new_messages: list[Message]) -> bool:
        return False


class BudgetPolicy:
    """Retrieve only when context is crowded; store always.

    Retrieval is skipped when the context window is still mostly empty
    (no need to pull more memories in). Useful for long tasks where early
    steps are self-contained and only later steps need cross-session recall.
    """

    def __init__(self, retrieve_threshold: float = 0.5):
        """
        Args:
            retrieve_threshold: fraction of context window that must be filled
                before retrieval is triggered. Default 0.5 = 50% full.
        """
        self.retrieve_threshold = retrieve_threshold

    async def should_retrieve(self, query: str, token_count: int, context_window: int) -> bool:
        if not query:
            return False
        if context_window and context_window > 0:
            return (token_count / context_window) >= self.retrieve_threshold
        return True

    async def should_compress(self, messages: list[Message], budget: int) -> bool:
        return rough_token_count(messages) > budget

    async def should_store(self, new_messages: list[Message]) -> bool:
        return True
