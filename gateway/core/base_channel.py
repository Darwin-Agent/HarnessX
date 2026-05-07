from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cachetools import TTLCache

from .config import get_dedup_dir

if TYPE_CHECKING:
    from .dispatch import ChannelDispatcher

logger = logging.getLogger(__name__)

_DEDUP_TTL = 600  # 10 minutes
_DEDUP_SIZE = 2048
_DEDUP_GC_INTERVAL = 64  # GC after every N writes
_TEXT_DEBOUNCE_S = 0.7  # 700ms merge window for fragmented/rapid text
_MEDIA_FALLBACK_S = 10.0  # flush media-only after 10s if no text follows


class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    FILE = "file"
    VIDEO = "video"
    STICKER = "sticker"
    SYSTEM = "system"


class ConversationType(str, Enum):
    DM = "dm"
    GROUP = "group"
    TOPIC = "topic"
    COMMENT = "comment"


@dataclass
class ConversationContext:
    type: ConversationType
    chat_id: str
    group_id: str | None = None
    group_name: str | None = None
    topic_id: str | None = None
    topic_name: str | None = None
    post_id: str | None = None
    mentioned: bool = False
    is_dm: bool = False


@dataclass
class MessageEvent:
    text: str
    sender_id: str
    sender_name: str
    platform: str
    message_id: str
    conversation: ConversationContext
    message_type: MessageType = MessageType.TEXT
    reply_to: str | None = None
    media_paths: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    raw: dict = field(default_factory=dict)

    def is_command(self) -> bool:
        return self.text.strip().startswith("/")

    def get_command(self) -> str:
        return self.text.strip().split()[0][1:].lower()

    def get_command_args(self) -> list[str]:
        return self.text.strip().split()[1:]


@dataclass
class ReplyTarget:
    chat_id: str
    thread_id: str | None = None
    quote_message_id: str | None = None

    @classmethod
    def from_event(cls, event: MessageEvent, create_thread: bool = False) -> "ReplyTarget":
        ctx = event.conversation
        return cls(
            chat_id=ctx.chat_id,
            thread_id=ctx.topic_id if ctx.type == ConversationType.TOPIC else None,
            quote_message_id=event.message_id if create_thread else None,
        )


@dataclass
class SendResult:
    success: bool
    message_id: str | None = None
    error: str | None = None
    retryable: bool = False


class StallError(Exception):
    pass


class BaseChannel(ABC):
    name: str
    display_name: str
    config_schema: dict = {}

    reconnect_backoff: tuple[float, ...] = (5, 10, 30, 60, 120)
    max_reconnect_attempts: int = 0
    stall_timeout: float = 120.0
    stream_edit_interval: float = 0.8
    stream_buffer_threshold: int = 20

    # ── Platform capability flags ───────────────────────────────────────────
    supports_edit: bool = True  # can edit an already-sent message
    supports_reactions: bool = False  # can add emoji reactions to messages
    supports_threads: bool = False  # can create / reply in threads
    max_message_length: int = 4096

    # ── Debounce configuration (overridable per channel) ───────────────────
    text_debounce_s: float = _TEXT_DEBOUNCE_S
    media_fallback_s: float = _MEDIA_FALLBACK_S

    def __init__(self, config: dict, dispatcher: "ChannelDispatcher") -> None:
        self.config = config
        self._dispatcher = dispatcher
        self.connection_state: Literal["connecting", "online", "offline", "error"] = "offline"
        self._dedup_cache: TTLCache = TTLCache(maxsize=_DEDUP_SIZE, ttl=_DEDUP_TTL)
        self._dedup_store: dict[str, float] = {}  # key → expiry ts (file-backed)
        self._dedup_writes = 0
        self._dedup_dirty = False
        self._dedup_last_flush = time.time()
        self._last_message_ts: float = time.time()
        self._pending: dict[str, tuple[MessageEvent, asyncio.TimerHandle]] = {}
        self._pending_media: dict[str, tuple[list[str], asyncio.TimerHandle | None]] = {}
        self._load_dedup_store()

    # ── Persistent dedup ────────────────────────────────────────────────────

    @property
    def _dedup_path(self) -> Path:
        return get_dedup_dir() / f"{self.name}.json"

    def _load_dedup_store(self) -> None:
        try:
            if self._dedup_path.exists():
                raw: dict = json.loads(self._dedup_path.read_text(encoding="utf-8"))
                now = time.time()
                self._dedup_store = {k: v for k, v in raw.items() if v > now}
                for mid, exp in self._dedup_store.items():
                    self._dedup_cache[mid] = exp
                if len(self._dedup_store) < len(raw):
                    self._save_dedup_store()
        except Exception:
            self._dedup_store = {}

    def _save_dedup_store(self) -> None:
        try:
            self._dedup_path.write_text(json.dumps(self._dedup_store), encoding="utf-8")
        except Exception as e:
            logger.debug("[%s] dedup store write failed: %s", self.name, e)

    def _gc_dedup_store(self) -> None:
        now = time.time()
        self._dedup_store = {k: v for k, v in self._dedup_store.items() if v > now}

    # ── Reconnect + Stall-Watchdog framework ───────────────────────────────

    def _backoff_sequence(self):
        backoff = list(self.reconnect_backoff)
        i = 0
        while True:
            yield backoff[min(i, len(backoff) - 1)]
            i += 1

    async def start(self) -> None:
        attempts = 0
        for delay in self._backoff_sequence():
            try:
                self.connection_state = "connecting"
                logger.info("[%s] connecting…", self.name)
                await self._connect()
                self.connection_state = "online"
                self._last_message_ts = time.time()
                logger.info("[%s] online", self.name)
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._listen())
                    tg.create_task(self._watchdog())
            except* (StallError, Exception) as eg:
                attempts += 1
                first_exc = eg.exceptions[0]
                if self.max_reconnect_attempts and attempts >= self.max_reconnect_attempts:
                    self.connection_state = "error"
                    logger.error("[%s] max reconnect attempts reached: %s", self.name, first_exc)
                    raise
                self.connection_state = "offline"
                logger.warning(
                    "[%s] disconnected (%s), retrying in %.0fs (attempt %d)",
                    self.name,
                    first_exc,
                    delay,
                    attempts,
                )
                await asyncio.sleep(delay)

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(30)
            elapsed = time.time() - self._last_message_ts
            if elapsed > self.stall_timeout:
                raise StallError(f"{self.name} stalled: no messages for {elapsed:.0f}s")
            if self._dedup_dirty:
                self._save_dedup_store()
                self._dedup_dirty = False
                self._dedup_last_flush = time.time()

    async def _enqueue(self, event: MessageEvent) -> None:
        is_dup = self.is_duplicate(event)
        logger.info(
            "[%s] _enqueue chat_id=%r msg_id=%r is_dup=%s text=%r",
            self.name,
            event.conversation.chat_id,
            event.message_id,
            is_dup,
            event.text[:40],
        )
        if is_dup:
            return
        self._last_message_ts = time.time()

        chat_id = event.conversation.chat_id
        has_media = bool(event.media_paths)
        has_text = bool(event.text and event.text not in ("[image]", "[voice]", "[video]", ""))

        # ── Layer 1: Media buffer ──────────────────────────────────────────
        # Media without text → buffer it and wait for text to arrive.
        if has_media and not has_text:
            if chat_id in self._pending_media:
                paths, old_handle = self._pending_media[chat_id]
                if old_handle is not None:
                    old_handle.cancel()
                paths.extend(event.media_paths)
            else:
                paths = list(event.media_paths)

            loop = asyncio.get_running_loop()
            handle = loop.call_later(
                self.media_fallback_s,
                lambda cid=chat_id, e=event: self._schedule_task(
                    self._flush_media_only(cid, e)
                ),
            )
            self._pending_media[chat_id] = (paths, handle)
            logger.info("[%s] media buffered chat_id=%r paths=%d", self.name, chat_id, len(paths))
            return

        # Text arrived → absorb any buffered media into this event
        if has_text and chat_id in self._pending_media:
            buffered_paths, media_handle = self._pending_media.pop(chat_id)
            if media_handle is not None:
                media_handle.cancel()
            event.media_paths = buffered_paths + event.media_paths
            if event.media_paths:
                event.message_type = MessageType.IMAGE
            logger.info("[%s] media merged into text event chat_id=%r", self.name, chat_id)

        # ── Layer 2: Text debounce (0.7s) ──────────────────────────────────
        if chat_id in self._pending:
            prev_event, handle = self._pending[chat_id]
            handle.cancel()
            if event.text and prev_event.text and event.text != prev_event.text:
                prev_event.text = prev_event.text + "\n" + event.text
            elif event.text:
                prev_event.text = event.text
            prev_event.media_paths.extend(event.media_paths)
            if event.media_paths:
                prev_event.message_type = MessageType.IMAGE
            event = prev_event

        loop = asyncio.get_running_loop()
        handle = loop.call_later(
            self.text_debounce_s,
            lambda e=event, cid=chat_id: self._schedule_task(
                self._flush_pending(cid, e)
            ),
        )
        self._pending[chat_id] = (event, handle)
        logger.info("[%s] text debounce scheduled chat_id=%r delay=%.3fs", self.name, chat_id, self.text_debounce_s)

    def _schedule_task(self, coro) -> None:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        task.add_done_callback(
            lambda t: (
                logger.error("[%s] flush failed: %s", self.name, t.exception())
                if not t.cancelled() and t.exception()
                else None
            )
        )

    async def _flush_pending(self, chat_id: str, event: MessageEvent) -> None:
        logger.info("[%s] _flush_pending chat_id=%r text=%r", self.name, chat_id, event.text[:40] if event.text else "")
        self._pending.pop(chat_id, None)
        await self._dispatcher.enqueue(self, event)

    async def _flush_media_only(self, chat_id: str, event: MessageEvent) -> None:
        media_entry = self._pending_media.pop(chat_id, None)
        if media_entry is None:
            return
        paths, _ = media_entry
        event.media_paths = paths
        event.message_type = MessageType.IMAGE
        logger.info("[%s] media fallback flush chat_id=%r paths=%d", self.name, chat_id, len(paths))
        await self._flush_pending(chat_id, event)

    # ── Abstract methods ────────────────────────────────────────────────────

    @abstractmethod
    async def _connect(self) -> None:
        """Establish connection (register webhook / WS handshake / start long poll)."""

    @abstractmethod
    async def _listen(self) -> None:
        """Listen for incoming messages and call self._enqueue(event). Raise on disconnect."""

    @abstractmethod
    async def stop(self, timeout: float = 10.0) -> None:
        """Graceful shutdown: stop receiving, wait for in-flight runs, then close."""

    @abstractmethod
    async def send_typing(self, target: ReplyTarget) -> None:
        """Send typing indicator. Called by dispatcher before harness.run()."""

    @abstractmethod
    async def send(self, target: ReplyTarget, text: str, **kwargs) -> SendResult:
        """Send a complete message."""

    @abstractmethod
    async def send_stream(
        self,
        target: ReplyTarget,
        queue: asyncio.Queue,
        edit_interval: float = 0.8,
        buffer_threshold: int = 20,
    ) -> SendResult:
        """
        Consume tokens from queue and send incrementally.
        Terminates when queue receives None sentinel.
        """

    # ── Rate limiting helper ────────────────────────────────────────────────

    async def _rate_acquire(self) -> None:
        """Acquire a rate-limit token before making an outbound API call."""
        await self._dispatcher._rate_limiter.acquire(self.name)

    # ── Retry helper ───────────────────────────────────────────────────────

    async def _send_with_retry(
        self,
        target: ReplyTarget,
        text: str,
        max_retries: int = 3,
        **kwargs,
    ) -> SendResult:
        """Call send() with exponential backoff on retryable failures."""
        if self.max_message_length and len(text) > self.max_message_length:
            text = text[: self.max_message_length]
        delay = 0.5
        result = SendResult(success=False, error="not attempted")
        for attempt in range(max_retries + 1):
            result = await self.send(target, text, **kwargs)
            if result.success or not result.retryable or attempt == max_retries:
                return result
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
        return result

    # ── Reaction interface ─────────────────────────────────────────────────

    async def add_reaction(self, message_id: str, emoji: str) -> str | None:
        """Add an emoji reaction to a message. Returns reaction_id if supported."""
        return None

    async def remove_reaction(self, message_id: str, reaction_id: str) -> None:
        """Remove a previously added reaction."""

    def resolve_stream_reply(self, chat_id: str, text: str) -> None:
        """Resolve a pending stream reply future with the final text.

        Called by dispatcher after harness.run() completes, in case the
        channel's send_stream() didn't resolve the future (e.g. webhook failed).
        Default implementation is a no-op; DingTalk overrides this.
        """

    # ── Optional overrides ──────────────────────────────────────────────────

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        raise NotImplementedError(f"{self.name} must implement verify_webhook()")

    def resolve_session_id(self, event: MessageEvent) -> str:
        ctx = event.conversation
        match ctx.type:
            case ConversationType.DM:
                return f"{self.name}-dm-{event.sender_id}"
            case ConversationType.TOPIC:
                return f"{self.name}-g-{ctx.group_id}-t-{ctx.topic_id}"
            case ConversationType.COMMENT:
                return f"{self.name}-post-{ctx.post_id}-c-{ctx.chat_id}"
            case ConversationType.GROUP:
                mode = self.config.get("session_mode", "shared")
                if mode == "per_user":
                    return f"{self.name}-g-{ctx.chat_id}-u-{event.sender_id}"
                return f"{self.name}-g-{ctx.chat_id}"
        return f"{self.name}-dm-{event.sender_id}"

    def make_reply_target(self, event: MessageEvent) -> ReplyTarget:
        create_thread = self.config.get("reply_in_thread", False)
        return ReplyTarget.from_event(event, create_thread=create_thread)

    def should_handle(self, event: MessageEvent) -> bool:
        allowed = self.config.get("allowed_users", [])
        if allowed and event.sender_id not in allowed and "*" not in allowed:
            return False
        if event.conversation.type in (ConversationType.GROUP, ConversationType.TOPIC):
            if self.config.get("require_mention", False) and not event.conversation.mentioned:
                return False
        return True

    def is_duplicate(self, event: MessageEvent) -> bool:
        key = f"{event.platform}:{event.message_id}"
        if key in self._dedup_cache:
            return True
        expiry = self._dedup_store.get(key)
        if expiry and expiry > time.time():
            self._dedup_cache[key] = expiry
            return True
        exp = time.time() + _DEDUP_TTL
        self._dedup_cache[key] = exp
        self._dedup_store[key] = exp
        self._dedup_writes += 1
        self._dedup_dirty = True
        if self._dedup_writes >= _DEDUP_GC_INTERVAL:
            self._gc_dedup_store()
            self._dedup_writes = 0
        return False

    def system_prompt(self) -> str:
        return ""

    def help_text(self) -> str:
        return (
            f"*{self.display_name} Bot*\n\n"
            "Available commands:\n"
            "/help - Show this help message\n"
            "/reset - Clear conversation history and start a new session\n"
            "/cancel - Cancel the currently running task\n"
            "/status - Show session info, model, and queue depth\n"
            "/usage - Show token usage and cost for this session\n"
            "/model - Show the current model\n"
            "/logs [n] - Show last N lines of the gateway log (default 20)\n"
            "/version - Show gateway version and runtime info\n"
            "/compact - Compress conversation history to free context window\n"
            "/skills - List installed skills\n"
            "/restart - Hot-restart this channel\n"
            "/reload-config - Reload gateway.yaml config without restarting"
        )

    async def on_session_start(self, session_id: str, event: MessageEvent) -> None:
        pass

    async def on_session_reset(self, session_id: str) -> None:
        pass
