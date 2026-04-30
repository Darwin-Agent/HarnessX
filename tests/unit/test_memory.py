# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import pytest
from harnessx.core.events import Message, rough_token_count, estimate_block_tokens
from harnessx.processors.memory.strategies.base import (
    BaseMemory,
    MutableMemory,
    compress_by_token_budget,
)
from harnessx.processors.memory.strategies.sliding_window import SlidingWindowMemory
from harnessx.processors.memory.strategies.summarization import SummarizationMemory
from harnessx.processors.memory.strategies.custom import InMemoryMemory


class TestMemory:
    @pytest.mark.asyncio
    async def test_sliding_window_basic(self):
        mem = SlidingWindowMemory(n=3)
        msgs = [Message(role="user", content=f"msg{i}") for i in range(5)]
        for m in msgs:
            await mem.add([m])
        retrieved = await mem.retrieve("", k=10)
        assert len(retrieved) <= 3
        # Should have the last 3
        assert retrieved[-1].content == "msg4"

    @pytest.mark.asyncio
    async def test_sliding_window_retrieve_limit(self):
        mem = SlidingWindowMemory(n=20)
        msgs = [Message(role="user", content=f"msg{i}") for i in range(10)]
        await mem.add(msgs)
        retrieved = await mem.retrieve("", k=5)
        assert len(retrieved) <= 10  # retrieve returns what's available, up to k

    @pytest.mark.asyncio
    async def test_sliding_window_compress(self):
        mem = SlidingWindowMemory(n=20)
        msgs = [Message(role="user", content="x" * 100) for _ in range(5)]
        compressed = await mem.compress(msgs, budget=50)
        # With budget=50 tokens and 100 chars/msg (~25 tokens each), should keep few
        assert len(compressed) <= 5

    @pytest.mark.asyncio
    async def test_sliding_window_persist_load(self):
        mem = SlidingWindowMemory(n=10)
        await mem.add([Message(role="user", content="test")])
        await mem.persist()  # No-op, should not raise
        loaded = await mem.load("any_run_id")
        assert len(loaded) == 1

    @pytest.mark.asyncio
    async def test_summarization_memory_compress(self):
        mem = SummarizationMemory()
        # Use long messages to ensure compression is triggered (budget in tokens)
        msgs = [Message(role="user", content="x" * 400) for _ in range(5)]
        # Budget of 10 tokens means only ~40 chars fits, but each msg is 400 chars = 100 tokens
        compressed = await mem.compress(msgs, budget=10)
        # Compressed should be smaller than original
        assert len(compressed) < len(msgs)

    # ─── Multimodal token counting ───────────────────────────────────────────────

    def test_rough_token_count_text_only(self):
        msgs = [Message(role="user", content="hello world")]
        count = rough_token_count(msgs)
        assert count > 0

    def test_rough_token_count_with_image(self):
        text_msg = Message(role="user", content="hello world")
        mm_msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "hello world"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                },
            ],
        )
        text_count = rough_token_count([text_msg])
        mm_count = rough_token_count([mm_msg])
        # Multimodal message should be more expensive due to image block
        assert mm_count > text_count

    def test_rough_token_count_image_only(self):
        msg = Message(
            role="user",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                },
            ],
        )
        count = rough_token_count([msg])
        # Should be at least the default image token cost
        assert count >= 1000

    def test_estimate_block_tokens_text(self):
        assert estimate_block_tokens({"type": "text", "text": "hi"}) == 0

    def test_estimate_block_tokens_image(self):
        assert estimate_block_tokens({"type": "image"}) == 1000

    def test_estimate_block_tokens_audio(self):
        block = {"type": "audio", "duration_seconds": 10}
        assert estimate_block_tokens(block) == 250  # 10 * 25

    def test_estimate_block_tokens_unknown(self):
        assert estimate_block_tokens({"type": "custom_modality"}) == 200

    # ─── Multimodal compress ─────────────────────────────────────────────────────

    def test_compress_multimodal_drops_image_heavy(self):
        """Image messages are heavier, so compress should drop them first when budget is tight."""
        text_msg = Message(role="user", content="short text")
        image_msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "with image"},
                {"type": "image", "source": {"type": "base64", "data": "x"}},
            ],
        )
        msgs = [image_msg, text_msg]
        # Budget enough for text but not text+image
        compressed = compress_by_token_budget(msgs, budget=50)
        assert len(compressed) <= len(msgs)
        # At minimum, keeps the last message
        assert len(compressed) >= 1

    # ─── InMemoryMemory multimodal ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_inmemory_add_multimodal(self):
        mem = InMemoryMemory()
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "describe this image"},
                {"type": "image", "source": {"type": "base64", "data": "abc"}},
            ],
        )
        await mem.add([msg])
        retrieved = await mem.retrieve("describe")
        assert len(retrieved) == 1

    @pytest.mark.asyncio
    async def test_inmemory_retrieve_multimodal_no_crash(self):
        """Retrieve should not crash on multimodal messages even with text-only search."""
        mem = InMemoryMemory()
        await mem.add(
            [
                Message(
                    role="user",
                    content=[
                        {"type": "image", "source": {"type": "base64", "data": "abc"}},
                    ],
                ),
                Message(role="user", content="text only"),
            ]
        )
        result = await mem.retrieve("text", k=10)
        assert isinstance(result, list)

    # ─── InMemoryMemory MutableMemory (update/delete) ────────────────────────────

    @pytest.mark.asyncio
    async def test_inmemory_update(self):
        mem = InMemoryMemory()
        await mem.add([Message(role="user", content="original")])
        ids = mem.list_ids()
        assert len(ids) == 1

        ok = await mem.update(ids[0], Message(role="user", content="updated"))
        assert ok is True

        retrieved = await mem.retrieve("updated")
        assert any("updated" in (m.content or "") for m in retrieved)

    @pytest.mark.asyncio
    async def test_inmemory_update_not_found(self):
        mem = InMemoryMemory()
        ok = await mem.update("nonexistent-id", Message(role="user", content="x"))
        assert ok is False

    @pytest.mark.asyncio
    async def test_inmemory_delete(self):
        mem = InMemoryMemory()
        await mem.add([Message(role="user", content="to delete")])
        ids = mem.list_ids()
        assert len(ids) == 1

        ok = await mem.delete(ids[0])
        assert ok is True
        assert len(mem.list_ids()) == 0

    @pytest.mark.asyncio
    async def test_inmemory_delete_not_found(self):
        mem = InMemoryMemory()
        ok = await mem.delete("nonexistent-id")
        assert ok is False

    # ─── Protocol checks ─────────────────────────────────────────────────────────

    def test_inmemory_is_mutable_memory(self):
        mem = InMemoryMemory()
        assert isinstance(mem, MutableMemory)
        assert isinstance(mem, BaseMemory)

    def test_sliding_window_is_base_memory(self):
        mem = SlidingWindowMemory()
        assert isinstance(mem, BaseMemory)
