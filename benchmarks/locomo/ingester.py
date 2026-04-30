"""SessionIngester — pluggable history injection for LoCoMo.

Architecture
------------
``SessionIngester`` drives the ingestion loop.  It delegates the actual
message construction to a ``SessionCompressor``, which decides how much
(and in what form) each session's conversation is stored in memory.

Three built-in compressors
--------------------------
VerbatimCompressor
    Stores every turn as-is, prefixed with ``[YYYY-MM-DD S<n>] Speaker:``.
    Zero LLM cost; highest recall for exact-match questions; noisy for
    long conversations (300+ turns).

SummaryCompressor
    Calls an LLM to produce one summary message per session.
    Reduces storage by ~10–20×; good signal-to-noise for multi-hop QA.
    Requires a model provider (default: claude-haiku).

FactCompressor
    Calls an LLM to extract a bullet list of key facts from each session.
    Highest precision for temporal-reasoning and multi-hop questions;
    may miss fine-grained details needed for single-hop exact matches.
    Requires a model provider (default: claude-haiku).

Custom compressors
------------------
Implement the ``SessionCompressor`` protocol::

    class MyCompressor:
        async def compress(
            self,
            session: LoCoMoSession,
        ) -> list[Message]:
            ...

Usage
-----
::

    from benchmarks.locomo.ingester import SessionIngester, SummaryCompressor
    from harnessx.processors.memory.strategies.custom import InMemoryMemory

    memory = InMemoryMemory(max_messages=5000)
    ingester = SessionIngester(compressor=SummaryCompressor())
    await ingester.ingest(task.sessions, memory)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from harnessx.core.events import Message

from .task import LoCoMoSession


def _parse_session_date(date_str: str) -> datetime:
    """Parse ``YYYY-MM-DD`` session date to a datetime for daily capture."""
    try:
        return datetime.fromisoformat(date_str + "T12:00:00+00:00")
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


# ─── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class SessionCompressor(Protocol):
    """Transform one LoCoMo session into a list of ``Message`` objects.

    Implementations decide how to represent session history:
    verbatim turns, an LLM-generated summary, or extracted facts.
    The returned messages are passed directly to ``BaseMemory.add()``.
    """

    async def compress(self, session: LoCoMoSession) -> list[Message]: ...


# ─── VerbatimCompressor ───────────────────────────────────────────────────────


class VerbatimCompressor:
    """Store every utterance verbatim with a temporal prefix.

    Each turn becomes one ``Message`` with content::

        [2024-03-15 S4] Alice: I finally got the promotion!

    Pros:  zero LLM cost; deterministic; preserves all details.
    Cons:  noisy at scale (300+ utterances per conversation).
    Best for: single-hop QA baselines; ablation studies.
    """

    def __init__(self, speaker_map: dict[str, str] | None = None):
        """
        Args:
            speaker_map: optional mapping from "A"/"B" to real names,
                         e.g. ``{"A": "Alice", "B": "Bob"}``.
                         Defaults to the raw speaker labels from the dataset.
        """
        self.speaker_map = speaker_map or {}

    async def compress(self, session: LoCoMoSession) -> list[Message]:
        messages: list[Message] = []
        prefix = f"[{session.date} S{session.session_id}]"
        for turn in session.turns:
            speaker = self.speaker_map.get(turn.speaker, turn.speaker)
            content = f"{prefix} {speaker}: {turn.text}"
            messages.append(Message(role="user", content=content))
        return messages


# ─── SummaryCompressor ───────────────────────────────────────────────────────

_SUMMARY_PROMPT = """\
You are summarising a conversation session for long-term memory storage.
Capture all important facts, events, decisions, and emotional moments.
Be concise but complete. Write in third-person past tense.
Include the date and participants.

Session date: {date}
Session {session_id} of the conversation:
{dialogue}

Write a 3-5 sentence summary:"""


class SummaryCompressor:
    """Summarise each session into a single memory entry using an LLM.

    Produces one ``Message`` per session with the LLM-generated summary.
    Compresses storage by roughly 10–20× compared to verbatim.

    Best for: multi-hop QA; summarization tasks.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        speaker_map: dict[str, str] | None = None,
    ):
        self.model = model
        self.speaker_map = speaker_map or {}

    def _format_dialogue(self, session: LoCoMoSession) -> str:
        lines: list[str] = []
        for turn in session.turns:
            speaker = self.speaker_map.get(turn.speaker, turn.speaker)
            lines.append(f"{speaker}: {turn.text}")
        return "\n".join(lines)

    async def compress(self, session: LoCoMoSession) -> list[Message]:
        dialogue = self._format_dialogue(session)
        prompt = _SUMMARY_PROMPT.format(
            date=session.date,
            session_id=session.session_id,
            dialogue=dialogue,
        )
        summary = await self._call_llm(prompt)
        content = f"[Summary S{session.session_id} {session.date}] {summary}"
        return [Message(role="user", content=content)]

    async def _call_llm(self, prompt: str) -> str:
        try:
            import litellm  # type: ignore

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            # Fallback: return first 200 chars of dialogue as a stub summary
            return f"(summary unavailable: {exc})"


# ─── FactCompressor ──────────────────────────────────────────────────────────

_FACTS_PROMPT = """\
Extract a list of key facts from this conversation session.
Each fact should be a single, self-contained sentence.
Include: events, decisions, personal details, plans, emotional reactions.
Prepend each fact with the date in brackets.

Session date: {date}
Session {session_id}:
{dialogue}

List of facts (one per line, starting with "- "):"""


class FactCompressor:
    """Extract structured facts from each session using an LLM.

    Produces one ``Message`` per extracted fact (multiple per session).
    Highest precision for temporal-reasoning and multi-hop queries because
    each fact is individually embedded and retrieved.

    Best for: temporal reasoning; multi-hop QA; adversarial questions.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        speaker_map: dict[str, str] | None = None,
    ):
        self.model = model
        self.speaker_map = speaker_map or {}

    def _format_dialogue(self, session: LoCoMoSession) -> str:
        lines: list[str] = []
        for turn in session.turns:
            speaker = self.speaker_map.get(turn.speaker, turn.speaker)
            lines.append(f"{speaker}: {turn.text}")
        return "\n".join(lines)

    async def compress(self, session: LoCoMoSession) -> list[Message]:
        dialogue = self._format_dialogue(session)
        prompt = _FACTS_PROMPT.format(
            date=session.date,
            session_id=session.session_id,
            dialogue=dialogue,
        )
        raw = await self._call_llm(prompt)
        facts = [line.lstrip("- ").strip() for line in raw.splitlines() if line.strip().startswith("-")]
        if not facts:
            # Fallback: treat entire response as one fact
            facts = [raw.strip()]
        return [
            Message(role="user", content=f"[FACT S{session.session_id} {session.date}] {fact}")
            for fact in facts
            if fact
        ]

    async def _call_llm(self, prompt: str) -> str:
        try:
            import litellm  # type: ignore

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            return f"- (fact extraction unavailable: {exc})"


# ─── LightMemoryLLMCompressor ────────────────────────────────────────────────


class LightMemoryLLMCompressor:
    """Write session turns to the light-memory store via LLM fact extraction.

    Uses ``write_memories_from_session_with_llm`` which calls an LLM to extract
    semantic facts (title, summary, keywords, importance) from chunks of turns,
    producing fewer but higher-quality memories than the rule-based path.

    Returns ``[]`` — storage is self-managed on disk.
    """

    def __init__(self, cfg: object, call_llm) -> None:
        self._cfg = cfg
        self._call_llm = call_llm

    async def compress(self, session: LoCoMoSession) -> list[Message]:
        from harnessx.plugins.dimensions.light_memory._core.lifecycle import (
            perform_capture,
            write_memories_from_session_with_llm,
        )

        turns = [{"speaker": t.speaker, "text": t.text} for t in session.turns]
        await write_memories_from_session_with_llm(
            self._cfg,
            self._call_llm,
            turns,
            session_id=session.session_id,
            session_date=session.date,
        )

        session_dt = _parse_session_date(session.date)
        for turn in session.turns:
            perform_capture(self._cfg, turn.speaker, turn.text, session_dt)

        return []


# ─── LightMemorySessionCompressor ───────────────────────────────────────────


class LightMemorySessionCompressor:
    """Write each session's turns directly to the light-memory file store.

    Uses the plugin's rule-based ``write_memories_from_session`` from
    ``harnessx.plugins.dimensions.light_memory._core.lifecycle``.

    Returns ``[]`` — storage is self-managed on disk; callers should pass
    a ``LightMemoryBackend`` (not ``InMemoryMemory``) as the memory backend.
    """

    def __init__(self, cfg: object) -> None:
        self._cfg = cfg  # PluginConfig from _core/types.py

    async def compress(self, session: LoCoMoSession) -> list[Message]:
        from harnessx.plugins.dimensions.light_memory._core.lifecycle import (
            perform_capture,
            write_memories_from_session,
        )

        turns = [{"speaker": t.speaker, "text": t.text} for t in session.turns]
        write_memories_from_session(
            self._cfg,
            turns,
            session_id=session.session_id,
            session_date=session.date,
        )

        session_dt = _parse_session_date(session.date)
        for turn in session.turns:
            perform_capture(self._cfg, turn.speaker, turn.text, session_dt)

        return []


# ─── SessionIngester ─────────────────────────────────────────────────────────


class SessionIngester:
    """Pre-load LoCoMo session history into a memory backend.

    Iterates over the provided sessions in chronological order, compresses
    each one via ``compressor``, and writes the resulting messages to
    ``memory`` with session-level metadata (``session_id``, ``timestamp``).

    This runs *before* ``Harness.run()``.  After ingestion the memory
    backend contains the full conversation history; the agent's
    ``ContextAssemblyProcessor`` will retrieve relevant fragments at each
    step via semantic search.

    Args:
        compressor:  a ``SessionCompressor`` instance (Verbatim/Summary/Fact)
        concurrency: max parallel LLM calls for LLM-based compressors.
                     Set to 1 for sequential processing (safer for rate limits).
    """

    def __init__(
        self,
        compressor: SessionCompressor | None = None,
        concurrency: int = 4,
    ):
        if compressor is None:
            compressor = VerbatimCompressor()
        self.compressor = compressor
        self.concurrency = concurrency

    async def ingest(
        self,
        sessions: list[LoCoMoSession],
        memory: object,
    ) -> int:
        """Compress and store all sessions into ``memory``.

        Args:
            sessions: list of ``LoCoMoSession`` in chronological order
            memory:   any ``BaseMemory`` implementation; if it is a
                      ``ChromaMemory``, session_id and timestamp are written
                      as metadata for time-range filtering.

        Returns:
            Total number of messages written to memory.
        """
        sem = asyncio.Semaphore(self.concurrency)
        total = 0

        async def _ingest_one(session: LoCoMoSession) -> int:
            async with sem:
                messages = await self.compressor.compress(session)
            if not messages:
                return 0
            # Write with temporal metadata if the backend supports it
            try:
                await memory.add(
                    messages,
                    session_id=session.session_id,
                    timestamp=session.date,
                )
            except TypeError:
                # Fallback for backends that don't accept extra kwargs
                await memory.add(messages)
            return len(messages)

        tasks = [_ingest_one(s) for s in sessions]
        counts = await asyncio.gather(*tasks)
        total = sum(counts)
        return total
