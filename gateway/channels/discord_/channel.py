from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
from collections import deque
from pathlib import Path

try:
    import discord
    from discord.ext import commands  # noqa: F401
except ImportError as _e:
    raise ImportError("Discord channel requires 'discord.py>=2.4'. Install with: pip install 'discord.py>=2.4'") from _e

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
from .formatter import split_text, truncate
from .utils import verify_discord_interaction, download_attachment

logger = logging.getLogger(__name__)


class DiscordChannel(BaseChannel):
    name = "discord"
    display_name = "Discord"
    stream_edit_interval = 1.5
    stream_buffer_threshold = 120

    supports_edit = True
    supports_threads = True
    supports_reactions = True
    max_message_length = 2000
    text_debounce_s = 0.0
    media_fallback_s = 0.0
    _MAX_CACHED_MESSAGE_IDS = 500

    config_schema = {
        "type": "object",
        "required": ["bot_token"],
        "properties": {
            "bot_token": {"type": "string", "title": "Bot Token", "format": "password"},
            "public_key": {"type": "string", "title": "Application Public Key (Interactions mode)"},
            "application_id": {"type": "string", "title": "Application ID (required for slash command replies)"},
            "require_mention": {"type": "boolean", "default": False},
            "allowed_guilds": {"type": "array", "items": {"type": "string"}, "title": "Allowed server IDs"},
            "reply_in_thread": {"type": "boolean", "default": False},
        },
    }

    def __init__(self, config: dict, dispatcher) -> None:
        super().__init__(config, dispatcher)
        self._client: discord.Client | None = None
        self._stop_event = asyncio.Event()
        _media_dir_cfg = config.get("_workspace_media_dir")
        self._media_cache = Path(_media_dir_cfg) if _media_dir_cfg else get_media_cache_dir()
        self._msg_cache: dict[str, discord.Message] = {}
        self._thread_cache: dict[str, str] = {}  # original_msg_id → thread_id
        self._interaction_cache: dict[str, dict] = {}  # interaction_id → {token}
        # Dedup: bound to _MAX_CACHED_MESSAGE_IDS, FIFO eviction via deque
        self._processed_message_ids: set[str] = set()
        self._processed_message_id_queue: deque[str] = deque()
        # Race-condition cache: starter_message_id → thread_id
        self._recent_thread_starts: dict[str, str] = {}

    async def _connect(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # needed for guild.get_member() in role mention detection
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_message(message: discord.Message):
            await self._on_message(message)

        @self._client.event
        async def on_thread_create(thread: discord.Thread):
            # Cache thread_id keyed by the starter message id.
            # In Discord, public threads created from a message share the same id.
            # This resolves the race where on_thread_create fires before on_message
            # for the starter message, so _on_message can route it to the thread.
            self._recent_thread_starts[str(thread.id)] = str(thread.id)
            if len(self._recent_thread_starts) > 200:
                oldest = next(iter(self._recent_thread_starts))
                del self._recent_thread_starts[oldest]

        @self._client.event
        async def on_ready():
            logger.info("[discord] logged in as %s", self._client.user)

    async def _listen(self) -> None:
        try:
            await self._client.start(self.config["bot_token"])
        except discord.LoginFailure as e:
            raise ConnectionError(f"Discord login failed: {e}") from e

    async def _on_message(self, message: discord.Message) -> None:
        # Ignore own messages and other bots
        if message.author.bot:
            return

        # Dedup: guard against duplicate deliveries
        msg_id = str(message.id)
        if msg_id in self._processed_message_ids:
            logger.debug("[discord] duplicate message %s skipped", msg_id)
            return
        if len(self._processed_message_ids) >= self._MAX_CACHED_MESSAGE_IDS:
            oldest = self._processed_message_id_queue.popleft()
            self._processed_message_ids.discard(oldest)
        self._processed_message_ids.add(msg_id)
        self._processed_message_id_queue.append(msg_id)

        logger.debug(
            "[discord] message from %s in %s (guild=%s): %r",
            message.author,
            message.channel,
            message.guild,
            (message.content or "")[:80],
        )

        # Guild filter — coerce scalar string to list to handle YAML scalar values
        raw_guilds = self.config.get("allowed_guilds", [])
        if isinstance(raw_guilds, str):
            allowed_guilds = [g.strip() for g in raw_guilds.split(",") if g.strip()]
        else:
            allowed_guilds = list(raw_guilds) if raw_guilds else []
        if allowed_guilds and message.guild and str(message.guild.id) not in allowed_guilds:
            logger.debug("[discord] guild %s not in allowed_guilds %s — skipped", message.guild.id, allowed_guilds)
            return

        text = message.content or ""
        # Remove @mention prefix
        if self._client.user:
            text = re.sub(rf"<@!?{self._client.user.id}>\s*", "", text).strip()

        # Attachments
        media_paths: list[str] = []
        mtype = MessageType.TEXT
        for att in message.attachments:
            data = await download_attachment(att.url)
            if data:
                path = self._media_cache / f"discord_{att.id}_{att.filename}"
                path.write_bytes(data)
                media_paths.append(str(path))
                content_type = att.content_type or mimetypes.guess_type(att.filename)[0] or ""
                if content_type.startswith("image/") and mtype == MessageType.TEXT:
                    mtype = MessageType.IMAGE
                elif content_type.startswith("audio/") and mtype == MessageType.TEXT:
                    mtype = MessageType.VOICE
                elif mtype == MessageType.TEXT:
                    mtype = MessageType.FILE

        if not text and not media_paths:
            logger.debug(
                "[discord] message has no text and no attachments — skipped (check Message Content Intent in Discord Developer Portal)"
            )
            return

        if not text and media_paths:
            text = f"[{mtype.value}]"

        # Save message for send_stream
        self._msg_cache[str(message.id)] = message
        if len(self._msg_cache) > 200:
            oldest = next(iter(self._msg_cache))
            del self._msg_cache[oldest]

        # Conversation type — determine mention status
        bot_user = self._client.user
        mentioned = bot_user in message.mentions if bot_user else False
        # Role mention: check if any mentioned role belongs to the bot
        if not mentioned and message.guild and bot_user:
            bot_member = message.guild.get_member(bot_user.id)
            if bot_member:
                mentioned_role_ids = {r.id for r in getattr(message, "role_mentions", [])}
                bot_role_ids = {r.id for r in bot_member.roles}
                matched = mentioned_role_ids & bot_role_ids
                if matched:
                    mentioned = True
                    for role_id in matched:
                        text = re.sub(rf"<@&{role_id}>\s*", "", text).strip()
        logger.debug("[discord] mentioned=%s text=%r", mentioned, text[:80])
        if isinstance(message.channel, discord.DMChannel):
            conv = ConversationContext(
                type=ConversationType.DM,
                chat_id=str(message.channel.id),
                is_dm=True,
            )
        elif isinstance(message.channel, discord.Thread):
            conv = ConversationContext(
                type=ConversationType.TOPIC,
                chat_id=str(message.channel.parent_id or message.channel.id),
                group_id=str(message.guild.id) if message.guild else None,
                group_name=message.guild.name if message.guild else None,
                topic_id=str(message.channel.id),
                topic_name=message.channel.name,
                mentioned=mentioned,
            )
        else:
            # Thread race: on_thread_create may have fired before on_message for the
            # starter message. If the message has a thread (via flags or cache), route
            # the conversation to that thread so replies land in the right place.
            thread_started = getattr(message, "thread", None)
            race_thread_id = self._recent_thread_starts.get(str(message.id))
            effective_thread_id = str(thread_started.id) if thread_started else race_thread_id
            if effective_thread_id:
                conv = ConversationContext(
                    type=ConversationType.TOPIC,
                    chat_id=str(message.channel.id),
                    group_id=str(message.guild.id) if message.guild else None,
                    group_name=message.guild.name if message.guild else None,
                    topic_id=effective_thread_id,
                    mentioned=mentioned,
                )
            else:
                conv = ConversationContext(
                    type=ConversationType.GROUP,
                    chat_id=str(message.channel.id),
                    group_id=str(message.guild.id) if message.guild else None,
                    group_name=message.guild.name if message.guild else None,
                    mentioned=mentioned,
                )

        # Handle reply/quote: extract referenced message text
        reply_to_id = None
        if message.reference and message.reference.resolved:
            ref_msg = message.reference.resolved
            reply_to_id = str(ref_msg.id)
            quoted = getattr(ref_msg, "content", "") or ""
            if quoted:
                text = f"[quoted message: {quoted.strip()[:500]}]\n\n{text}"

        event = MessageEvent(
            text=text,
            sender_id=str(message.author.id),
            sender_name=message.author.display_name,
            platform=self.name,
            message_id=str(message.id),
            message_type=mtype,
            conversation=conv,
            reply_to=reply_to_id,
            media_paths=media_paths,
            raw={},
        )
        await self._enqueue(event)

    # ── Webhook (Interactions Endpoint) ───────────────────────────────────

    async def _on_webhook(self, payload: dict) -> dict | None:
        interaction_type = payload.get("type")
        if interaction_type == 1:  # PING — verify endpoint
            return {"type": 1}
        if interaction_type == 2:  # APPLICATION_COMMAND — defer + handle async
            task = asyncio.create_task(self._handle_interaction(payload))
            task.add_done_callback(
                lambda t: (
                    logger.warning("[discord] interaction error: %s", t.exception())
                    if not t.cancelled() and t.exception()
                    else None
                )
            )
            return {"type": 5}  # DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE
        return None

    async def _handle_interaction(self, payload: dict) -> None:
        data = payload.get("data", {})
        member = payload.get("member") or {}
        user = member.get("user") or payload.get("user") or {}
        guild_id = payload.get("guild_id")
        channel_id = payload.get("channel_id", "")  # noqa: F841

        cmd_name = data.get("name", "")
        options = data.get("options") or []
        option_text = " ".join(str(o.get("value", "")) for o in options)
        text = f"/{cmd_name} {option_text}".strip()

        interaction_id = payload.get("id", "")
        interaction_token = payload.get("token", "")
        if interaction_id and interaction_token:
            self._interaction_cache[interaction_id] = {"token": interaction_token}
            if len(self._interaction_cache) > 200 and self._interaction_cache:
                oldest = next(iter(self._interaction_cache))
                del self._interaction_cache[oldest]

        conv = ConversationContext(
            type=ConversationType.DM if not guild_id else ConversationType.GROUP,
            chat_id=f"ix:{interaction_id}",
            group_id=guild_id,
            mentioned=True,
        )
        event = MessageEvent(
            text=text,
            sender_id=str(user.get("id", "")),
            sender_name=user.get("global_name") or user.get("username", ""),
            platform=self.name,
            message_id=interaction_id,
            message_type=MessageType.TEXT,
            conversation=conv,
            raw={"interaction": True},
        )
        await self._enqueue(event)

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        public_key = self.config.get("public_key", "")
        if not public_key:
            return True
        return verify_discord_interaction(headers, body, public_key)

    def make_reply_target(self, event: MessageEvent) -> ReplyTarget:
        ctx = event.conversation
        # Already in a Discord Thread — reply in the same thread
        if ctx.type == ConversationType.TOPIC:
            return ReplyTarget(chat_id=ctx.chat_id, thread_id=ctx.topic_id)
        # Configured to create a thread from the original message
        if self.config.get("reply_in_thread", False) and not ctx.is_dm:
            return ReplyTarget(chat_id=ctx.chat_id, quote_message_id=event.message_id)
        return ReplyTarget(chat_id=ctx.chat_id)

    async def _ensure_thread_id(self, target: ReplyTarget) -> str:
        """Return the channel/thread ID to post into.

        If quote_message_id is set and we haven't created a thread yet, create
        one from the original user message and cache the result.
        """
        if target.thread_id:
            return target.thread_id
        if target.quote_message_id:
            if target.quote_message_id in self._thread_cache:
                return self._thread_cache[target.quote_message_id]
            orig = self._msg_cache.get(target.quote_message_id)
            if orig and not isinstance(orig.channel, discord.Thread):
                try:
                    thread = await orig.create_thread(name="Conversation")
                    self._thread_cache[target.quote_message_id] = str(thread.id)
                    return str(thread.id)
                except discord.HTTPException as e:
                    logger.debug("[discord] create_thread failed: %s", e)
        return target.chat_id

    async def add_reaction(self, message_id: str, emoji: str) -> str | None:
        msg = self._msg_cache.get(message_id)
        if not msg:
            return None
        try:
            await msg.add_reaction(emoji)
            return emoji
        except Exception as e:
            logger.debug("[discord] add_reaction error: %s", e)
            return None

    async def remove_reaction(self, message_id: str, reaction_id: str) -> None:
        if not self._client:
            return
        msg = self._msg_cache.get(message_id)
        if not msg:
            return
        try:
            await msg.remove_reaction(reaction_id, self._client.user)
        except Exception as e:
            logger.debug("[discord] remove_reaction error: %s", e)

    # ── Sending ───────────────────────────────────────────────────────────

    async def send_typing(self, target: ReplyTarget) -> None:
        if not self._client or target.chat_id.startswith("ix:"):
            return
        await self._rate_acquire()
        ch_id = int(await self._ensure_thread_id(target))
        ch = self._client.get_channel(ch_id)
        if ch and hasattr(ch, "trigger_typing"):
            try:
                await ch.trigger_typing()
            except Exception:
                pass

    async def send(self, target: ReplyTarget, text: str, **kwargs) -> SendResult:
        if target.chat_id.startswith("ix:"):
            return await self._send_interaction_followup(target.chat_id[3:], text)
        if not self._client:
            return SendResult(success=False, error="not connected")
        await self._rate_acquire()
        ch_id = int(await self._ensure_thread_id(target))
        ch = self._client.get_channel(ch_id)
        if not ch:
            return SendResult(success=False, error=f"channel {ch_id} not found")
        chunks = split_text(text)
        last_id: str | None = None
        for chunk in chunks:
            try:
                msg = await ch.send(chunk)
                last_id = str(msg.id)
            except discord.HTTPException as e:
                return SendResult(success=False, error=str(e), retryable=True)
        return SendResult(success=True, message_id=last_id)

    async def send_embed(
        self,
        target: ReplyTarget,
        title: str = "",
        description: str = "",
        color: int = 0x5865F2,
        fields: list[dict] | None = None,
    ) -> SendResult:
        """Send a Discord embed message.

        fields format: [{"name": "...", "value": "...", "inline": True}, ...]
        """
        if not self._client:
            return SendResult(success=False, error="not connected")
        await self._rate_acquire()
        ch_id = int(await self._ensure_thread_id(target))
        ch = self._client.get_channel(ch_id)
        if not ch:
            return SendResult(success=False, error=f"channel {ch_id} not found")
        embed = discord.Embed(title=title, description=description, color=color)
        for f in fields or []:
            embed.add_field(
                name=f.get("name", ""),
                value=f.get("value", ""),
                inline=f.get("inline", False),
            )
        try:
            msg = await ch.send(embed=embed)
            return SendResult(success=True, message_id=str(msg.id))
        except discord.HTTPException as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def _send_interaction_followup(self, interaction_id: str, text: str) -> SendResult:
        info = self._interaction_cache.get(interaction_id)
        if not info:
            return SendResult(success=False, error="interaction not found or expired")
        app_id = self.config.get("application_id", "")
        if not app_id:
            return SendResult(success=False, error="application_id not configured")
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"https://discord.com/api/v10/webhooks/{app_id}/{info['token']}",
                    json={"content": text[:2000]},
                )
            if resp.status_code in (200, 204):
                return SendResult(success=True)
            return SendResult(success=False, error=f"HTTP {resp.status_code}", retryable=resp.status_code >= 500)
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_stream(
        self,
        target: ReplyTarget,
        queue: asyncio.Queue,
        edit_interval: float = 1.5,
        buffer_threshold: int = 120,
    ) -> SendResult:
        if target.chat_id.startswith("ix:"):
            return await self._send_stream_interaction(target.chat_id[3:], queue, edit_interval, buffer_threshold)

        async def _drain():
            while True:
                if await queue.get() is None:
                    break

        if not self._client:
            await _drain()
            return SendResult(success=False, error="not connected")

        ch_id = int(await self._ensure_thread_id(target))
        ch = self._client.get_channel(ch_id)
        if not ch:
            await _drain()
            return SendResult(success=False, error="channel not found")

        # Persistent typing loop keeps the indicator alive during long tool calls.
        async def _typing_loop():
            while True:
                try:
                    await ch.trigger_typing()
                except Exception:
                    pass
                try:
                    await asyncio.sleep(8)
                except asyncio.CancelledError:
                    return

        async def _send_new(text: str) -> discord.Message | None:
            try:
                return await ch.send(text[:2000])
            except discord.HTTPException as e:
                logger.debug("[discord] send error: %s", e)
                return None

        typing_task = asyncio.create_task(_typing_loop())
        last_msg: discord.Message | None = None
        buf = ""

        try:
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
                            last_msg = await _send_new(buf)
                            buf = ""
                        await _send_new(f"*⚙️ {name}…*")
                    continue

                buf += delta
        finally:
            typing_task.cancel()

        if buf:
            last_msg = await _send_new(buf[:2000])

        return SendResult(success=True, message_id=str(last_msg.id) if last_msg else None)

    async def _edit_discord(self, msg: discord.Message, text: str) -> None:
        await self._rate_acquire()
        try:
            await msg.edit(content=truncate(text))
        except discord.RateLimited as e:
            logger.debug("[discord] rate limited, waiting %.1fs", e.retry_after)
            await asyncio.sleep(e.retry_after)
            try:
                await msg.edit(content=truncate(text))
            except Exception:
                pass
        except discord.HTTPException as e:
            logger.debug("[discord] edit error: %s", e)

    async def _send_stream_interaction(
        self,
        interaction_id: str,
        queue: asyncio.Queue,
        edit_interval: float = 1.5,
        buffer_threshold: int = 120,
    ) -> SendResult:
        info = self._interaction_cache.get(interaction_id)
        if not info:
            while True:
                if await queue.get() is None:
                    break
            return SendResult(success=False, error="interaction not found or expired")
        app_id = self.config.get("application_id", "")
        if not app_id:
            while True:
                if await queue.get() is None:
                    break
            return SendResult(success=False, error="application_id not configured")

        import httpx

        token = info["token"]
        base_url = f"https://discord.com/api/v10/webhooks/{app_id}/{token}"
        buf = ""

        async with httpx.AsyncClient(timeout=10) as client:
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
                            try:
                                for chunk in split_text(buf, 1990):
                                    await client.post(base_url, json={"content": chunk})
                            except Exception as e:
                                logger.debug("[discord] interaction send error: %s", e)
                            buf = ""
                        try:
                            await client.post(base_url, json={"content": f"*⚙️ {name}…*"})
                        except Exception as e:
                            logger.debug("[discord] interaction tool_start error: %s", e)
                    continue
                buf += delta

            if buf:
                try:
                    for chunk in split_text(buf, 1990):
                        await client.post(base_url, json={"content": chunk})
                except Exception as e:
                    logger.debug("[discord] interaction final send error: %s", e)

        return SendResult(success=True)

    async def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._client:
            try:
                await asyncio.wait_for(self._client.close(), timeout=timeout)
            except Exception:
                pass

    def system_prompt(self) -> str:
        return "Reply format: Discord Markdown (**bold**, *italic*, `code`). Max 2000 characters per message."


register_builtin(DiscordChannel)
