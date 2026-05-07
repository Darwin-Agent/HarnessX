from __future__ import annotations

import asyncio
import hmac
import logging
from pathlib import Path

try:
    from telegram import Update, Bot, InlineKeyboardMarkup, InlineKeyboardButton  # noqa: F401
    from telegram.ext import (
        Application,
        MessageHandler,
        CommandHandler,
        CallbackQueryHandler,
        filters,
        ContextTypes,
    )
    from telegram.constants import ParseMode, ChatAction
    from telegram.error import TelegramError, RetryAfter
except ImportError as _e:
    raise ImportError(
        "Telegram channel requires 'python-telegram-bot>=21.0'. Install with: pip install 'python-telegram-bot>=21.0'"
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
from .formatter import to_markdown_v2
from .utils import split_text, _prefix_within_utf16_limit

logger = logging.getLogger(__name__)

# Telegram Bot API 7.0+ allows only specific emojis in setMessageReaction.
_TG_REACTION_MAP: dict[str, str] = {
    "⏳": "🤔",
    "✅": "👍",
    "❌": "👎",
}


class TelegramChannel(BaseChannel):
    name = "telegram"
    display_name = "Telegram"
    stall_timeout = 3600.0
    stream_edit_interval = 1.0
    stream_buffer_threshold = 30

    supports_edit = True
    supports_threads = True
    supports_reactions = True
    max_message_length = 4096
    text_debounce_s = 0.0
    media_fallback_s = 0.0

    config_schema = {
        "type": "object",
        "required": ["bot_token"],
        "properties": {
            "bot_token": {"type": "string", "title": "Bot Token", "format": "password"},
            "allowed_users": {"type": "array", "items": {"type": "string"}, "title": "Allowed user_id list"},
            "require_mention": {"type": "boolean", "default": False},
            "webhook_secret_token": {"type": "string", "title": "Webhook Secret Token"},
            "reply_in_thread": {"type": "boolean", "default": False},
        },
    }

    _MEDIA_GROUP_WAIT_S = 0.8

    def __init__(self, config: dict, dispatcher) -> None:
        super().__init__(config, dispatcher)
        self._app: Application | None = None
        self._stop_event = asyncio.Event()
        _media_dir_cfg = config.get("_workspace_media_dir")
        self._media_cache = Path(_media_dir_cfg) if _media_dir_cfg else get_media_cache_dir()
        self._reaction_msg_cache: dict[str, int] = {}  # message_id → chat_id
        self._media_groups: dict[str, tuple[MessageEvent, asyncio.TimerHandle]] = {}

    async def _connect(self) -> None:
        self._app = Application.builder().token(self.config["bot_token"]).build()
        self._app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, self._on_message))
        for cmd in ("reset", "help", "cancel", "start", "pair"):
            self._app.add_handler(CommandHandler(cmd, self._on_command))
        self._app.add_handler(CallbackQueryHandler(self._on_callback_query))
        await self._app.initialize()

    async def _listen(self) -> None:
        await self._app.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=False,
        )
        await self._app.start()
        # Block until stop() is called
        await self._stop_event.wait()

    async def _on_command(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await self._process_update(update)

    async def _on_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await self._process_update(update)

    async def _process_update(self, update: Update) -> None:
        msg = update.message or update.edited_message
        if not msg:
            return
        # Ignore messages from other bots
        if msg.from_user and msg.from_user.is_bot:
            return

        text = msg.text or msg.caption or ""
        media_paths: list[str] = []
        mtype = MessageType.TEXT

        if msg.photo:
            mtype = MessageType.IMAGE
            text = msg.caption or "[image]"
            try:
                tg_file = await msg.photo[-1].get_file()
                path = self._media_cache / f"tg_{msg.photo[-1].file_id}.jpg"
                await tg_file.download_to_drive(str(path))
                media_paths.append(str(path))
            except Exception as e:
                logger.warning("[telegram] photo download failed: %s", e)
        elif msg.voice or msg.audio:
            mtype = MessageType.VOICE
            obj = msg.voice or msg.audio
            text = "[voice]"
            try:
                tg_file = await obj.get_file()
                ext = ".ogg" if msg.voice else ".mp3"
                path = self._media_cache / f"tg_{obj.file_id}{ext}"
                await tg_file.download_to_drive(str(path))
                media_paths.append(str(path))
            except Exception as e:
                logger.warning("[telegram] audio download failed: %s", e)
        elif msg.video:
            mtype = MessageType.VIDEO
            text = msg.caption or "[video]"
            try:
                tg_file = await msg.video.get_file()
                path = self._media_cache / f"tg_{msg.video.file_id}.mp4"
                await tg_file.download_to_drive(str(path))
                media_paths.append(str(path))
            except Exception as e:
                logger.warning("[telegram] video download failed: %s", e)
        elif msg.document:
            mtype = MessageType.FILE
            text = f"[file: {msg.document.file_name or 'unknown'}]"
            try:
                tg_file = await msg.document.get_file()
                path = self._media_cache / f"tg_{msg.document.file_id}_{msg.document.file_name or 'file'}"
                await tg_file.download_to_drive(str(path))
                media_paths.append(str(path))
            except Exception as e:
                logger.warning("[telegram] document download failed: %s", e)

        chat = msg.chat
        user = msg.from_user
        sender_id = str(user.id) if user else "unknown"

        if chat.type == "private":
            conv = ConversationContext(type=ConversationType.DM, chat_id=str(chat.id), is_dm=True)
        elif msg.message_thread_id:
            bot_mentioned = self._bot_mentioned(msg)
            conv = ConversationContext(
                type=ConversationType.TOPIC,
                chat_id=str(chat.id),
                group_id=str(chat.id),
                group_name=chat.title or "",
                topic_id=str(msg.message_thread_id),
                mentioned=bot_mentioned,
            )
        else:
            bot_mentioned = self._bot_mentioned(msg)
            conv = ConversationContext(
                type=ConversationType.GROUP,
                chat_id=str(chat.id),
                group_id=str(chat.id),
                group_name=chat.title or "",
                mentioned=bot_mentioned,
            )

        # Handle reply/quote: extract quoted message text
        reply_to_id = None
        if msg.reply_to_message:
            reply_to_id = str(msg.reply_to_message.message_id)
            quoted = msg.reply_to_message.text or msg.reply_to_message.caption or ""
            if quoted:
                text = f'[quoted message: {quoted.strip()[:500]}]\n\n{text}'

        event = MessageEvent(
            text=text,
            sender_id=sender_id,
            sender_name=user.full_name if user else sender_id,
            platform=self.name,
            message_id=str(msg.message_id),
            message_type=mtype,
            conversation=conv,
            reply_to=reply_to_id,
            media_paths=media_paths,
            raw={},
        )
        # Cache for reaction support
        self._reaction_msg_cache[str(msg.message_id)] = chat.id
        if len(self._reaction_msg_cache) > 500:
            oldest = next(iter(self._reaction_msg_cache))
            del self._reaction_msg_cache[oldest]

        # Media group (album) handling: accumulate photos with same media_group_id
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id and media_paths:
            if media_group_id in self._media_groups:
                prev_event, handle = self._media_groups[media_group_id]
                handle.cancel()
                prev_event.media_paths.extend(media_paths)
                if text and text not in ("[image]", "[video]") and not prev_event.text:
                    prev_event.text = text
                event = prev_event
            loop = asyncio.get_running_loop()
            handle = loop.call_later(
                self._MEDIA_GROUP_WAIT_S,
                lambda mgid=media_group_id: self._flush_media_group(mgid),
            )
            self._media_groups[media_group_id] = (event, handle)
            return

        await self._enqueue(event)

    def _flush_media_group(self, media_group_id: str) -> None:
        entry = self._media_groups.pop(media_group_id, None)
        if entry is None:
            return
        event, _ = entry
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._enqueue(event))
        task.add_done_callback(
            lambda t: (
                logger.error("[telegram] media group flush failed: %s", t.exception())
                if not t.cancelled() and t.exception()
                else None
            )
        )

    def _bot_mentioned(self, msg) -> bool:
        entities = list(msg.entities or []) + list(msg.caption_entities or [])
        if not entities:
            return False
        bot_username = ""
        if self._app:
            try:
                bot_username = (self._app.bot.username or "").lower()
            except Exception:
                pass
        for e in entities:
            if e.type == "mention":
                if not bot_username:
                    return True  # can't verify username, assume mentioned
                try:
                    mentioned = msg.parse_entity(e).lstrip("@").lower()
                except Exception:
                    return True
                if mentioned == bot_username:
                    return True
            elif e.type == "text_mention":
                # user without a username — matched by user ID
                euser = getattr(e, "user", None)
                if euser and self._app:
                    try:
                        if str(euser.id) == str(self._app.bot.id):
                            return True
                    except Exception:
                        pass
        return False

    async def _on_callback_query(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query:
            return
        await query.answer()
        chat = query.message.chat if query.message else None
        if not chat:
            return
        conv = ConversationContext(
            type=ConversationType.DM if chat.type == "private" else ConversationType.GROUP,
            chat_id=str(chat.id),
            is_dm=chat.type == "private",
        )
        event = MessageEvent(
            text=query.data or "",
            sender_id=str(query.from_user.id) if query.from_user else "unknown",
            sender_name=query.from_user.full_name if query.from_user else "unknown",
            platform=self.name,
            message_id=str(query.id),
            message_type=MessageType.SYSTEM,
            conversation=conv,
            raw={"callback_query": True},
        )
        await self._enqueue(event)

    # ── Sending ───────────────────────────────────────────────────────────

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        expected = self.config.get("webhook_secret_token", "")
        if not expected:
            return False
        actual = headers.get("x-telegram-bot-api-secret-token", "")
        return hmac.compare_digest(expected, actual)

    async def send_typing(self, target: ReplyTarget) -> None:
        if not self._app:
            return
        try:
            await self._rate_acquire()
            await self._app.bot.send_chat_action(
                chat_id=int(target.chat_id),
                action=ChatAction.TYPING,
                message_thread_id=int(target.thread_id) if target.thread_id else None,
            )
        except Exception:
            pass

    async def add_reaction(self, message_id: str, emoji: str) -> str | None:
        chat_id = self._reaction_msg_cache.get(message_id)
        if not chat_id or not self._app:
            return None
        reaction_emoji = _TG_REACTION_MAP.get(emoji, emoji)
        try:
            from telegram import ReactionTypeEmoji

            await self._app.bot.set_message_reaction(
                chat_id=chat_id,
                message_id=int(message_id),
                reaction=[ReactionTypeEmoji(emoji=reaction_emoji)],
            )
            return reaction_emoji
        except Exception as e:
            logger.debug("[telegram] add_reaction error: %s", e)
            return None

    async def remove_reaction(self, message_id: str, reaction_id: str) -> None:
        chat_id = self._reaction_msg_cache.get(message_id)
        if not chat_id or not self._app:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=chat_id,
                message_id=int(message_id),
                reaction=[],
            )
        except Exception as e:
            logger.debug("[telegram] remove_reaction error: %s", e)

    async def send(self, target: ReplyTarget, text: str, **kwargs) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="not connected")
        await self._rate_acquire()
        chunks = split_text(text, 4000)
        last_id: str | None = None
        for chunk in chunks:
            try:
                msg = await self._app.bot.send_message(
                    chat_id=int(target.chat_id),
                    text=to_markdown_v2(chunk),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=int(target.thread_id) if target.thread_id else None,
                    disable_web_page_preview=True,
                )
                last_id = str(msg.message_id)
            except RetryAfter as e:
                logger.warning("[telegram] flood control, waiting %ss", e.retry_after)
                await asyncio.sleep(e.retry_after)
                try:
                    msg = await self._app.bot.send_message(
                        chat_id=int(target.chat_id),
                        text=to_markdown_v2(chunk),
                        parse_mode=ParseMode.MARKDOWN_V2,
                        message_thread_id=int(target.thread_id) if target.thread_id else None,
                        disable_web_page_preview=True,
                    )
                    last_id = str(msg.message_id)
                except Exception as e2:
                    return SendResult(success=False, error=str(e2), retryable=True)
            except TelegramError as e:
                logger.warning("[telegram] send failed: %s", e)
                # Fallback: plain text (UTF-16 aware cut)
                try:
                    msg = await self._app.bot.send_message(
                        chat_id=int(target.chat_id),
                        text=_prefix_within_utf16_limit(chunk, 4096),
                        message_thread_id=int(target.thread_id) if target.thread_id else None,
                    )
                    last_id = str(msg.message_id)
                except Exception as e2:
                    return SendResult(success=False, error=str(e2), retryable=True)
        return SendResult(success=True, message_id=last_id)

    async def send_stream(
        self,
        target: ReplyTarget,
        queue: asyncio.Queue,
        edit_interval: float = 1.0,
        buffer_threshold: int = 30,
    ) -> SendResult:
        if not self._app:
            while True:
                if await queue.get() is None:
                    break
            return SendResult(success=False, error="not connected")

        chat_id = int(target.chat_id)
        thread_id = int(target.thread_id) if target.thread_id else None

        async def _send_new(text: str) -> int | None:
            try:
                msg = await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=to_markdown_v2(_prefix_within_utf16_limit(text, 4000)),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=thread_id,
                    disable_web_page_preview=True,
                )
                return msg.message_id
            except RetryAfter as e:
                logger.warning("[telegram] flood control in stream, waiting %ss", e.retry_after)
                await asyncio.sleep(e.retry_after)
                try:
                    msg = await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=to_markdown_v2(_prefix_within_utf16_limit(text, 4000)),
                        parse_mode=ParseMode.MARKDOWN_V2,
                        message_thread_id=thread_id,
                        disable_web_page_preview=True,
                    )
                    return msg.message_id
                except Exception as e2:
                    logger.debug("[telegram] send error after retry: %s", e2)
                    return None
            except TelegramError:
                try:
                    msg = await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=_prefix_within_utf16_limit(text, 4096),
                        message_thread_id=thread_id,
                    )
                    return msg.message_id
                except Exception as e:
                    logger.debug("[telegram] send error: %s", e)
                    return None

        buf = ""
        last_msg_id: int | None = None
        last_edit_time = 0.0

        while True:
            try:
                delta = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                # Periodically flush buffer: send first message or edit existing one
                if buf:
                    now = asyncio.get_event_loop().time()
                    if last_msg_id is None:
                        last_msg_id = await _send_new(buf)
                        last_edit_time = now
                    elif now - last_edit_time >= edit_interval:
                        await self._edit_tg(chat_id, last_msg_id, buf)
                        last_edit_time = now
                continue

            if delta is None:
                break

            if isinstance(delta, dict):
                if delta.get("type") == "tool_start":
                    name = delta.get("name", "tool")
                    if buf:
                        # Finalize current segment before showing tool name
                        if last_msg_id is None:
                            last_msg_id = await _send_new(buf)
                        else:
                            await self._edit_tg(chat_id, last_msg_id, buf)
                        buf = ""
                        last_msg_id = None
                    await _send_new(f"⚙️ {name}…")
                continue

            buf += delta

            # Send initial message once buffer reaches threshold
            if last_msg_id is None and len(buf) >= buffer_threshold:
                last_msg_id = await _send_new(buf)
                last_edit_time = asyncio.get_event_loop().time()

        # Final flush with definitive content
        if buf:
            if last_msg_id is None:
                last_msg_id = await _send_new(buf)
            else:
                await self._edit_tg(chat_id, last_msg_id, buf)

        return SendResult(success=True, message_id=str(last_msg_id) if last_msg_id else None)

    async def send_with_keyboard(
        self,
        target: ReplyTarget,
        text: str,
        buttons: list[list[dict]],
    ) -> SendResult:
        """Send a message with an inline keyboard.

        buttons format: [[{"text": "label", "callback_data": "value"}, ...], ...]
        """
        if not self._app:
            return SendResult(success=False, error="not connected")
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(b["text"], callback_data=b.get("callback_data", "")) for b in row]
                for row in buttons
            ]
        )
        try:
            msg = await self._app.bot.send_message(
                chat_id=int(target.chat_id),
                text=to_markdown_v2(_prefix_within_utf16_limit(text, 4000)),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard,
                message_thread_id=int(target.thread_id) if target.thread_id else None,
                disable_web_page_preview=True,
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except TelegramError:
            try:
                msg = await self._app.bot.send_message(
                    chat_id=int(target.chat_id),
                    text=_prefix_within_utf16_limit(text, 4096),
                    reply_markup=keyboard,
                    message_thread_id=int(target.thread_id) if target.thread_id else None,
                )
                return SendResult(success=True, message_id=str(msg.message_id))
            except Exception as e:
                return SendResult(success=False, error=str(e), retryable=True)

    async def send_photo(self, target: ReplyTarget, path: str, caption: str = "") -> SendResult:
        """Send a photo file to the chat."""
        if not self._app:
            return SendResult(success=False, error="not connected")
        await self._rate_acquire()
        try:
            with open(path, "rb") as f:
                msg = await self._app.bot.send_photo(
                    chat_id=int(target.chat_id),
                    photo=f,
                    caption=caption[:1024] if caption else None,
                    message_thread_id=int(target.thread_id) if target.thread_id else None,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_document(self, target: ReplyTarget, path: str, filename: str = "", caption: str = "") -> SendResult:
        """Send a document/file to the chat."""
        if not self._app:
            return SendResult(success=False, error="not connected")
        import os as _os

        await self._rate_acquire()
        try:
            fname = filename or _os.path.basename(path)
            with open(path, "rb") as f:
                msg = await self._app.bot.send_document(
                    chat_id=int(target.chat_id),
                    document=f,
                    filename=fname,
                    caption=caption[:1024] if caption else None,
                    message_thread_id=int(target.thread_id) if target.thread_id else None,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def _edit_tg(self, chat_id: int, msg_id: int, text: str) -> None:
        try:
            await self._app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=to_markdown_v2(_prefix_within_utf16_limit(text, 4000)),
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            err = str(e).lower()
            if "not modified" in err:
                return  # harmless — content unchanged
            # MarkdownV2 parse failure: retry as plain text
            logger.debug("[telegram] edit md2 error, retrying plain: %s", e)
            try:
                await self._app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=_prefix_within_utf16_limit(text, 4096),
                )
            except Exception:
                pass

    async def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._app:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        self._app.updater.stop(),
                        self._app.stop(),
                        self._app.shutdown(),
                    ),
                    timeout=timeout,
                )
            except Exception:
                pass

    def system_prompt(self) -> str:
        return "Reply using standard Markdown. Messages over 4096 characters are split automatically."


register_builtin(TelegramChannel)
