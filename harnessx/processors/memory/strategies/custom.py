# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from typing import Any

from ....core.events import Message, _extract_text, message_to_dict, dict_to_message
from .base import compress_by_token_budget


class InMemoryMemory:
    """
    In-memory store with CRUD support.  Good for testing and short sessions.
    Messages are lost when the process exits.

    Implements ``MutableMemory`` (BaseMemory + update/delete).

    Usage:
        from harnessx.processors.strategies.memory.custom import InMemoryMemory
        memory = InMemoryMemory(max_messages=200)
    """

    def __init__(self, max_messages: int = 200):
        self.max_messages = max_messages
        self._store: OrderedDict[str, Message] = OrderedDict()
        self._sessions: dict[str, list[Message]] = {}

    def _evict(self) -> None:
        """Remove oldest entries when over capacity."""
        while len(self._store) > self.max_messages:
            self._store.popitem(last=False)

    async def add(self, messages: list[Message]) -> None:
        for m in messages:
            mid = str(uuid.uuid4())
            self._store[mid] = m
        self._evict()

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        query_blocks: list[dict] | None = None,
    ) -> list[Message]:
        """Simple recency-based retrieval (no semantic search).

        Uses ``_extract_text`` so multimodal messages are searched by their
        text portion.  ``query_blocks`` is accepted but not used — subclass
        or replace with a multimodal backend to leverage it.
        """
        messages = list(self._store.values())
        query_lower = query.lower()
        if query_lower:
            relevant = [m for m in messages if query_lower in _extract_text(m.content).lower()]
            return relevant[-k:] if relevant else messages[-k:]
        return messages[-k:]

    async def compress(self, messages: list[Message], budget: int) -> list[Message]:
        return compress_by_token_budget(messages, budget)

    async def persist(self) -> None:
        pass  # in-memory only

    async def load(self, run_id: str) -> list[Message]:
        return self._sessions.get(run_id, [])

    # ── MutableMemory extensions ─────────────────────────────────────────

    async def update(self, message_id: str, message: Message) -> bool:
        """Replace the message at *message_id*.  Returns ``False`` if not found."""
        if message_id not in self._store:
            return False
        self._store[message_id] = message
        return True

    async def delete(self, message_id: str) -> bool:
        """Delete the message at *message_id*.  Returns ``False`` if not found."""
        if message_id not in self._store:
            return False
        del self._store[message_id]
        return True

    def list_ids(self) -> list[str]:
        """Return all stored message IDs in insertion order (utility for tests/debug)."""
        return list(self._store.keys())


class RedisMemory:
    """
    Redis-backed persistent memory.
    Messages are serialized as JSON and stored in a Redis list per agent.

    Usage:
        pip install redis
        from harnessx.processors.strategies.memory.custom import RedisMemory
        memory = RedisMemory(agent_id="my-agent", redis_url="redis://localhost:6379")
    """

    def __init__(
        self,
        agent_id: str = "default",
        redis_url: str = "redis://localhost:6379",
        max_messages: int = 1000,
        ttl_seconds: int = 86400,  # 24 hours
    ):
        self.agent_id = agent_id
        self.redis_url = redis_url
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self._redis: Any = None

    def _get_redis(self) -> Any:
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as e:
                raise ImportError("redis is required: pip install redis") from e
            self._redis = aioredis.from_url(self.redis_url)
        return self._redis

    def _key(self, suffix: str = "messages") -> str:
        return f"harnessx:{self.agent_id}:{suffix}"

    async def add(self, messages: list[Message]) -> None:
        if not messages:
            return
        r = self._get_redis()
        key = self._key()
        pipe = r.pipeline()
        for m in messages:
            pipe.rpush(key, json.dumps(message_to_dict(m)))
        pipe.ltrim(key, -self.max_messages, -1)
        pipe.expire(key, self.ttl_seconds)
        await pipe.execute()

    async def retrieve(self, query: str, k: int = 10) -> list[Message]:
        """Returns the most recent k messages (Redis doesn't do semantic search)."""
        r = self._get_redis()
        raw = await r.lrange(self._key(), -k, -1)
        return [dict_to_message(json.loads(item)) for item in raw]

    async def compress(self, messages: list[Message], budget: int) -> list[Message]:
        return compress_by_token_budget(messages, budget)

    async def persist(self) -> None:
        pass  # Redis auto-persists

    async def load(self, run_id: str) -> list[Message]:
        r = self._get_redis()
        raw = await r.lrange(self._key(f"session:{run_id}"), 0, -1)
        return [dict_to_message(json.loads(item)) for item in raw]
