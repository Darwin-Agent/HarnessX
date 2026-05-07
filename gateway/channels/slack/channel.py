from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError as _e:
    raise ImportError(
        "Slack channel requires 'slack-bolt>=1.20' and 'slack-sdk>=3.27'. "
        "Install with: pip install 'slack-bolt>=1.20' 'slack-sdk>=3.27'"
    ) from _e

from ...core.base_channel import (
    BaseChannel,
    ConversationContext,
    ConversationType,
    MessageEvent,
    MessageType,
    ReplyTarget,
    SendResult,
)
from ...core.config import get_media_cache_dir
from .. import register_builtin
from .formatter import to_mrkdwn
from .utils import verify_slack_signature, download_slack_file

logger = logging.getLogger(__name__)

_BOT_ID_PATTERN = re.compile(r"<@([A-Z0-9]+)>")

_SLACK_USER_CACHE_TTL = 300  # 5 minutes
_SLACK_MENTIONED_THREADS_MAX = 1000

_SLACK_REACTION_MAP: dict[str, str] = {
    "⏳": "hourglass_flowing_sand",
    "✅": "white_check_mark",
    "❌": "x",
}


class SlackChannel(BaseChannel):
    name = "slack"
    display_name = "Slack"
    stall_timeout = 3600.0
    stream_edit_interval = 0.8
    stream_buffer_threshold = 20

    supports_edit = True
    supports_threads = True
    supports_reactions = True
    max_message_length = 40000
    text_debounce_s = 0.0
    media_fallback_s = 0.0

    config_schema = {
        "type": "object",
        "required": ["bot_token", "app_token"],
        "properties": {
            "bot_token": {"type": "string", "title": "Bot Token (xoxb-…)", "format": "password"},
            "app_token": {"type": "string", "title": "App Token (xapp-…)", "format": "password"},
            "signing_secret": {"type": "string", "title": "Signing Secret", "format": "password"},
            "require_mention": {"type": "boolean", "default": True},
            "free_response_channels": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Channel IDs that do not require mention",
            },
            "reply_in_thread": {"type": "boolean", "default": True},
            "reply_broadcast": {"type": "boolean", "default": False},
        },
    }

    def __init__(self, config: dict, dispatcher) -> None:
        super().__init__(config, dispatcher)
        self._bolt_app: AsyncApp | None = None
        self._handler: AsyncSocketModeHandler | None = None
        self._client: AsyncWebClient | None = None
        self._bot_id: str | None = None
        self._stop_event = asyncio.Event()
        _media_dir_cfg = config.get("_workspace_media_dir")
        self._media_cache = Path(_media_dir_cfg) if _media_dir_cfg else get_media_cache_dir()
        self._reaction_msg_cache: dict[str, str] = {}  # ts → channel_id
        self._user_name_cache: dict[str, tuple[str, float]] = {}  # user_id → (name, ts)
        self._mentioned_threads: set[str] = set()  # thread_ts where bot was @mentioned
        self._bot_message_ts: set[str] = set()  # ts of messages bot sent
        # Dedup cache: event_id → expiry timestamp. Slack retries same event up to 3x.
        self._seen_event_ids: OrderedDict[str, float] = OrderedDict()
        self._seen_event_ids_ttl: float = 60.0

    def make_reply_target(self, event: MessageEvent) -> ReplyTarget:
        ctx = event.conversation
        if self.config.get("reply_in_thread", True) and not ctx.is_dm:
            # event.message_id is the Slack message ts — using it as thread_ts groups
            # all bot replies under the original message, keeping the channel clean.
            return ReplyTarget(chat_id=ctx.chat_id, thread_id=event.message_id)
        return ReplyTarget(chat_id=ctx.chat_id)

    def should_handle(self, event) -> bool:
        if not super().should_handle(event):
            # Check free_response_channels before giving up
            free = set(self.config.get("free_response_channels", []))
            if event.conversation.chat_id not in free:
                return False
        return True

    async def _connect(self) -> None:
        bot_token = self.config["bot_token"]
        self._bolt_app = AsyncApp(token=bot_token)
        self._client = self._bolt_app.client
        self._handler = AsyncSocketModeHandler(self._bolt_app, self.config["app_token"])

        # Register handlers
        self._bolt_app.event("message")(self._on_event)
        self._bolt_app.event("app_mention")(self._on_event)

        # Resolve bot user ID
        try:
            info = await self._client.auth_test()
            self._bot_id = info.get("user_id") or info.get("bot_id")
        except Exception as e:
            logger.warning("[slack] auth_test failed: %s", e)

    async def _listen(self) -> None:
        await self._handler.start_async()  # blocks; SDK reconnects internally

    async def _on_event(self, body: dict, client: AsyncWebClient) -> None:
        # Deduplicate Slack retries (same event delivered up to 3x via X-Slack-Retry-Num)
        event_id = body.get("event_id")
        if event_id:
            now = time.time()
            # Evict expired entries
            expired = [k for k, exp in self._seen_event_ids.items() if exp < now]
            for k in expired:
                del self._seen_event_ids[k]
            if event_id in self._seen_event_ids:
                logger.debug("[slack] duplicate event %s skipped", event_id)
                return
            self._seen_event_ids[event_id] = now + self._seen_event_ids_ttl

        event = body.get("event", body)
        # Filter bot's own messages and edits
        if event.get("bot_id") or event.get("subtype") in ("message_changed", "message_deleted", "bot_message"):
            return
        if event.get("user") == self._bot_id:
            return

        raw_text = event.get("text", "")
        # Remove @bot mention prefix
        text = _BOT_ID_PATTERN.sub("", raw_text).strip()

        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")

        # Download media files
        media_paths: list[str] = []
        mtype = MessageType.TEXT
        for f in event.get("files", []):
            url = f.get("url_private_download") or f.get("url_private")
            if not url:
                continue
            data = await download_slack_file(url, self.config["bot_token"])
            if data:
                fname = f.get("name", f"slack_{f.get('id', 'file')}")
                path = self._media_cache / fname
                path.write_bytes(data)
                media_paths.append(str(path))
                mime = f.get("mimetype", "")
                if mime.startswith("image/"):
                    mtype = MessageType.IMAGE
                elif mime.startswith("audio/"):
                    mtype = MessageType.VOICE
                elif mtype == MessageType.TEXT:
                    mtype = MessageType.FILE

        if not text and not media_paths:
            return

        # Determine if the bot was explicitly @mentioned in this message
        is_dm = channel_id.startswith("D")
        bot_mentioned = bool(self._bot_id and f"<@{self._bot_id}>" in raw_text)
        is_thread_reply = bool(thread_ts and thread_ts != ts)

        # Track mentioned threads: once bot is @mentioned in a thread, all
        # future replies in that thread pass the require_mention gate automatically.
        if bot_mentioned and thread_ts:
            self._mentioned_threads.add(thread_ts)
            if len(self._mentioned_threads) > _SLACK_MENTIONED_THREADS_MAX:
                # Trim oldest half
                to_remove = list(self._mentioned_threads)[: _SLACK_MENTIONED_THREADS_MAX // 2]
                for t in to_remove:
                    self._mentioned_threads.discard(t)

        # A message "counts as mentioned" if:
        # 1. Bot was explicitly @mentioned in this message, OR
        # 2. This is a thread reply and the thread was previously mentioned, OR
        # 3. This is a reply to a message the bot sent
        mentioned = (
            bot_mentioned
            or (is_thread_reply and thread_ts in self._mentioned_threads)
            or (is_thread_reply and thread_ts in self._bot_message_ts)
        )

        if is_dm:
            conv = ConversationContext(type=ConversationType.DM, chat_id=channel_id, is_dm=True)
        elif is_thread_reply:
            conv = ConversationContext(
                type=ConversationType.TOPIC,
                chat_id=channel_id,
                group_id=channel_id,
                topic_id=thread_ts,
                mentioned=mentioned,
            )
        else:
            conv = ConversationContext(
                type=ConversationType.GROUP,
                chat_id=channel_id,
                group_id=channel_id,
                mentioned=mentioned,
            )

        # Resolve display name (cached to avoid per-message API calls)
        sender_name = await self._resolve_user_name(client, user_id)

        event_obj = MessageEvent(
            text=text,
            sender_id=user_id,
            sender_name=sender_name,
            platform=self.name,
            message_id=ts,
            message_type=mtype,
            conversation=conv,
            media_paths=media_paths,
            raw={},
        )
        # Cache for reaction support
        if ts and channel_id:
            self._reaction_msg_cache[ts] = channel_id
            if len(self._reaction_msg_cache) > 500:
                oldest = next(iter(self._reaction_msg_cache))
                del self._reaction_msg_cache[oldest]
        await self._enqueue(event_obj)

    async def _resolve_user_name(self, client: AsyncWebClient, user_id: str) -> str:
        cached = self._user_name_cache.get(user_id)
        if cached:
            name, ts = cached
            if time.time() - ts < _SLACK_USER_CACHE_TTL:
                return name
        name = user_id
        try:
            info = await client.users_info(user=user_id)
            profile = info.get("user", {}).get("profile", {})
            name = profile.get("display_name") or profile.get("real_name") or user_id
        except Exception:
            pass
        self._user_name_cache[user_id] = (name, time.time())
        if len(self._user_name_cache) > 500:
            oldest = min(self._user_name_cache, key=lambda k: self._user_name_cache[k][1])
            del self._user_name_cache[oldest]
        return name

    # ── Webhook entry (Event API mode) ────────────────────────────────────

    async def _on_webhook(self, payload: dict) -> None:
        """Called by server.py in Event API mode."""
        # URL verification
        if "challenge" in payload:
            return
        await self._on_event(payload, self._client)

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        secret = self.config.get("signing_secret", "")
        if not secret:
            logger.warning("[slack] signing_secret not configured — rejecting webhook request")
            return False
        return verify_slack_signature(headers, body, secret)

    # ── Sending ───────────────────────────────────────────────────────────

    async def send_typing(self, target: ReplyTarget) -> None:
        if target.thread_id and self._client:
            try:
                await self._rate_acquire()
                await self._client.assistant_threads_setStatus(
                    channel_id=target.chat_id,
                    thread_ts=target.thread_id,
                    status="is thinking…",
                )
            except Exception:
                pass

    async def add_reaction(self, message_id: str, emoji: str) -> str | None:
        channel_id = self._reaction_msg_cache.get(message_id)
        if not channel_id or not self._client:
            return None
        name = _SLACK_REACTION_MAP.get(emoji, emoji.strip(":").lower().replace(" ", "_"))
        try:
            await self._rate_acquire()
            await self._client.reactions_add(channel=channel_id, timestamp=message_id, name=name)
            return name
        except Exception as e:
            logger.debug("[slack] add_reaction error: %s", e)
            return None

    async def remove_reaction(self, message_id: str, reaction_id: str) -> None:
        channel_id = self._reaction_msg_cache.get(message_id)
        if not channel_id or not self._client:
            return
        try:
            await self._rate_acquire()
            await self._client.reactions_remove(channel=channel_id, timestamp=message_id, name=reaction_id)
        except Exception as e:
            logger.debug("[slack] remove_reaction error: %s", e)

    async def send(self, target: ReplyTarget, text: str, **kwargs) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="not connected")
        await self._rate_acquire()
        try:
            resp = await self._client.chat_postMessage(
                channel=target.chat_id,
                text=to_mrkdwn(text),
                thread_ts=target.thread_id,
                reply_broadcast=self.config.get("reply_broadcast", False),
                mrkdwn=True,
            )
            if resp["ok"]:
                sent_ts = resp.get("ts")
                # Track bot-sent message ts so replies to them auto-pass the mention gate
                if sent_ts:
                    self._bot_message_ts.add(sent_ts)
                    if len(self._bot_message_ts) > 500:
                        oldest = next(iter(self._bot_message_ts))
                        self._bot_message_ts.discard(oldest)
                return SendResult(success=True, message_id=sent_ts)
            err = resp.get("error", "unknown")
            return SendResult(success=False, error=err, retryable=err == "ratelimited")
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_stream(
        self,
        target: ReplyTarget,
        queue: asyncio.Queue,
        edit_interval: float = 0.8,
        buffer_threshold: int = 20,
    ) -> SendResult:
        if not self._client:
            while True:
                if await queue.get() is None:
                    break
            return SendResult(success=False, error="not connected")

        channel = target.chat_id
        thread_ts = target.thread_id

        async def _send_new(text: str) -> str | None:
            try:
                resp = await self._client.chat_postMessage(
                    channel=channel,
                    text=to_mrkdwn(text),
                    thread_ts=thread_ts,
                    reply_broadcast=self.config.get("reply_broadcast", False),
                    mrkdwn=True,
                )
                if resp["ok"]:
                    return resp.get("ts")
            except Exception as e:
                logger.debug("[slack] send error: %s", e)
            return None

        buf = ""
        last_ts: str | None = None

        while True:
            try:
                delta = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if delta is None:
                break

            if isinstance(delta, dict):
                if delta.get("type") == "tool_start":
                    name = delta.get("name", "tool")
                    if buf:
                        last_ts = await _send_new(buf)
                        buf = ""
                    await _send_new(f"⚙️ {name}…")
                continue

            buf += delta

        if buf:
            last_ts = await _send_new(buf)

        return SendResult(success=True, message_id=last_ts)

    async def _edit_slack(self, channel: str, ts: str, text: str) -> None:
        await self._rate_acquire()
        try:
            await self._client.chat_update(
                channel=channel,
                ts=ts,
                text=to_mrkdwn(text[:39000]),
                mrkdwn=True,
            )
        except Exception as e:
            logger.debug("[slack] edit error: %s", e)

    async def send_blocks(self, target: ReplyTarget, blocks: list[dict], fallback_text: str = "") -> SendResult:
        """Send a Slack Block Kit message."""
        if not self._client:
            return SendResult(success=False, error="not connected")
        await self._rate_acquire()
        try:
            resp = await self._client.chat_postMessage(
                channel=target.chat_id,
                blocks=blocks,
                text=fallback_text,
                thread_ts=target.thread_id,
                reply_broadcast=self.config.get("reply_broadcast", False),
            )
            if resp["ok"]:
                return SendResult(success=True, message_id=resp.get("ts"))
            err = resp.get("error", "unknown")
            return SendResult(success=False, error=err, retryable=err == "ratelimited")
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._handler:
            try:
                await asyncio.wait_for(self._handler.close_async(), timeout=timeout)
            except Exception:
                pass

    def system_prompt(self) -> str:
        return "Reply format: use Slack mrkdwn (*bold*, _italic_, `code`, ```code block```)."


register_builtin(SlackChannel)
