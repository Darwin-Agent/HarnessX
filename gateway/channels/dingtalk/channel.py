from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import dingtalk_stream
    from dingtalk_stream import ChatbotMessage, DingTalkStreamClient
    from dingtalk_stream.frames import AckMessage
    from dingtalk_stream.frames import CallbackMessage as _CallbackMessage

    class _BotHandler(dingtalk_stream.ChatbotHandler):
        """Async wrapper: receive → enqueue → await reply_future → reply_text().

        Implements the same request-reply pattern as the DingTalk Stream SDK
        expects: process() must not return until a reply is ready, because
        reply_text() can only be called while the callback is still active.

        When the channel sends via sessionWebhook (stream path), it resolves
        the future with _SENT_VIA_WEBHOOK sentinel so reply_text(" ") fires as
        a minimal ack to keep the stream healthy.
        """

        _SENT_VIA_WEBHOOK = "__sent_via_webhook__"

        def __init__(self, callback) -> None:
            super().__init__()
            self._cb = callback

        async def process(self, message: _CallbackMessage):  # noqa: D102
            loop = asyncio.get_running_loop()
            reply_future: asyncio.Future[str] = loop.create_future()
            raw_data = message.data if isinstance(message.data, dict) else {}
            chatbot_msg = ChatbotMessage.from_dict(raw_data)
            try:
                await self._cb(chatbot_msg, reply_future, raw_data)
            except Exception as exc:
                if not reply_future.done():
                    reply_future.set_exception(exc)

            try:
                reply_text = await asyncio.wait_for(reply_future, timeout=300.0)
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("[dingtalk] reply_future timeout/error: %s", exc)
                reply_text = self._SENT_VIA_WEBHOOK  # fall back to ack-only

            if reply_text == self._SENT_VIA_WEBHOOK:
                self.reply_text(" ", chatbot_msg)
            elif reply_text:
                self.reply_text(reply_text, chatbot_msg)
            else:
                self.reply_text(" ", chatbot_msg)
            return AckMessage.STATUS_OK, "ok"

except ImportError as _e:
    raise ImportError(
        "DingTalk channel requires 'dingtalk-stream>=0.20'. Install with: pip install 'dingtalk-stream>=0.20'"
    ) from _e

try:
    import httpx
except ImportError as _e:
    raise ImportError("DingTalk channel requires 'httpx'. Install with: pip install httpx") from _e

from ...core.base_channel import (
    BaseChannel,
    ConversationContext,
    ConversationType,
    MessageEvent,
    MessageType,
    ReplyTarget,
    SendResult,
)
from ...core.config import get_media_cache_dir, get_store_dir
from .. import register_builtin
from .formatter import markdown_payload
from .utils import verify_dingtalk_signature, is_webhook_expired

logger = logging.getLogger(__name__)

_DINGTALK_API = "https://api.dingtalk.com/v1.0"
_AI_CARD_STREAM_MIN_INTERVAL = 0.3  # minimum seconds between card updates
_TOKEN_PREEMPTIVE_REFRESH = 300  # refresh access token 5min before expiry


class DingTalkChannel(BaseChannel):
    name = "dingtalk"
    display_name = "DingTalk"
    stall_timeout = 3600.0
    stream_edit_interval = 0.5
    stream_buffer_threshold = 15

    supports_edit = False  # DingTalk cannot edit sent messages
    supports_reactions = True  # DingTalk supports emotion reactions
    max_message_length = 20000

    config_schema = {
        "type": "object",
        "required": ["client_id", "client_secret"],
        "properties": {
            "client_id": {"type": "string", "title": "App Key (Client ID)"},
            "client_secret": {"type": "string", "title": "App Secret", "format": "password"},
            "card_template_id": {"type": "string", "title": "AI card template ID (optional)"},
            "msg_key": {"type": "string", "default": "sampleMarkdown", "title": "Open API message template type"},
            "require_mention": {"type": "boolean", "default": False},
            "free_response_chats": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Group chat IDs that do not require mention",
            },
            "mention_patterns": {"type": "array", "items": {"type": "string"}, "title": "Wake-word regex patterns"},
        },
    }

    def __init__(self, config: dict, dispatcher) -> None:
        super().__init__(config, dispatcher)
        self._stream_client: DingTalkStreamClient | None = None
        self._stop_event = asyncio.Event()
        _media_dir_cfg = config.get("_workspace_media_dir")
        self._media_cache = Path(_media_dir_cfg) if _media_dir_cfg else get_media_cache_dir()
        # Webhook persistence
        self._webhook_store: dict[str, dict] = {}
        self._load_webhook_store()
        # Access token cache
        self._access_token: str | None = None
        self._token_expiry: float = 0.0
        self._token_lock = asyncio.Lock()
        # AI Card active state persistence
        self._active_cards: dict[str, dict] = {}
        self._load_active_cards()
        # reply_future map: chat_id → (stream_loop, future, msg_id)
        # stream_loop is the loop the future was created on (DingTalk stream thread).
        # Resolutions must use call_soon_threadsafe to avoid cross-loop errors.
        self._reply_futures: dict[str, tuple] = {}
        # Dedup: msg_ids currently being processed (stream thread + main loop access)
        self._processing_msg_ids: set[str] = set()
        self._processing_msg_ids_lock: threading.Lock = threading.Lock()

    # ── Persistence helpers ────────────────────────────────────────────────

    @property
    def _webhook_store_path(self) -> Path:
        return get_store_dir() / "dingtalk_webhooks.json"

    @property
    def _active_cards_path(self) -> Path:
        return get_store_dir() / "dingtalk_active_cards.json"

    def _load_webhook_store(self) -> None:
        try:
            if self._webhook_store_path.exists():
                self._webhook_store = json.loads(self._webhook_store_path.read_text(encoding="utf-8"))
        except Exception:
            self._webhook_store = {}

    def _save_webhook_store(self) -> None:
        try:
            # GC truly expired entries (those with a known non-zero expiry that has passed)
            self._webhook_store = {
                k: v for k, v in self._webhook_store.items() if not is_webhook_expired(v.get("expired") or 0)
            }
            self._webhook_store_path.write_text(json.dumps(self._webhook_store), encoding="utf-8")
        except Exception as e:
            logger.warning("[dingtalk] webhook store write failed: %s", e)

    def _load_active_cards(self) -> None:
        try:
            if self._active_cards_path.exists():
                self._active_cards = json.loads(self._active_cards_path.read_text(encoding="utf-8"))
        except Exception:
            self._active_cards = {}

    def _save_active_cards(self) -> None:
        try:
            self._active_cards_path.write_text(json.dumps(self._active_cards), encoding="utf-8")
        except Exception as e:
            logger.warning("[dingtalk] active cards write failed: %s", e)

    async def _recover_active_cards(self) -> None:
        """Set orphaned AI cards (from a previous run) to FAILED state."""
        for trace_id, info in list(self._active_cards.items()):
            try:
                await self._update_ai_card(trace_id, info.get("content", "(process restarted)"), "FAILED")
                logger.info("[dingtalk] recovered orphan card %s → FAILED", trace_id)
            except Exception as e:
                logger.debug("[dingtalk] card recovery failed for %s: %s", trace_id, e)
        self._active_cards.clear()
        self._save_active_cards()

    # ── Mention gate ──────────────────────────────────────────────────────

    def should_handle(self, event) -> bool:
        if not super().should_handle(event):
            # free_response_chats bypasses require_mention for specific group chats
            free = set(self.config.get("free_response_chats", []))
            if event.conversation.chat_id not in free:
                return False
        return True

    # ── Access token ───────────────────────────────────────────────────────

    async def _get_access_token(self) -> str | None:
        now = time.time()
        if self._access_token and now < self._token_expiry - _TOKEN_PREEMPTIVE_REFRESH:
            return self._access_token
        async with self._token_lock:
            # Double-checked locking: another coroutine may have refreshed already
            now = time.time()
            if self._access_token and now < self._token_expiry - _TOKEN_PREEMPTIVE_REFRESH:
                return self._access_token
            client_id = self.config["client_id"]
            client_secret = self.config["client_secret"]
            try:
                async with httpx.AsyncClient(timeout=10) as hc:
                    resp = await hc.post(
                        f"{_DINGTALK_API}/oauth2/accessToken",
                        json={"appKey": client_id, "appSecret": client_secret},
                    )
                    data = resp.json()
                    token = data.get("accessToken")
                    expires_in = data.get("expireIn", 7200)
                    if token:
                        self._access_token = token
                        self._token_expiry = now + expires_in
                        return token
            except Exception as e:
                logger.warning("[dingtalk] access token refresh failed: %s", e)
            return None

    # ── Connection ─────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        credential = dingtalk_stream.Credential(self.config["client_id"], self.config["client_secret"])
        self._stream_client = DingTalkStreamClient(credential)
        self._stream_client.register_callback_handler(
            ChatbotMessage.TOPIC,
            _BotHandler(self._on_message),
        )
        await self._recover_active_cards()

    async def _listen(self) -> None:
        await self._stream_client.start()

    async def _on_message(
        self,
        msg: ChatbotMessage,
        reply_future: "asyncio.Future[str]",
        raw_data: dict | None = None,
    ) -> None:
        try:
            await self._process_message(msg, reply_future, raw_data or {})
        except Exception as e:
            logger.error("[dingtalk] message processing error: %s", e, exc_info=True)
            if not reply_future.done():
                reply_future.set_result(_BotHandler._SENT_VIA_WEBHOOK)

    async def _process_message(
        self,
        msg: ChatbotMessage,
        reply_future: "asyncio.Future[str] | None" = None,
        raw_data: dict | None = None,
    ) -> None:
        raw_data = raw_data or {}
        raw_text = (msg.text.content if hasattr(msg, "text") and msg.text else "") or ""
        text = re.sub(r"@[^\s]+\s*", "", raw_text).strip()

        media_paths: list[str] = []
        mtype = MessageType.TEXT

        msg_type = getattr(msg, "msgtype", "text") or "text"
        if msg_type == "picture":
            mtype = MessageType.IMAGE
            text = "[image]"
            code = getattr(getattr(msg, "image_content", None), "download_code", None) or ""
            if code:
                path = await self._download_resource(code, "image.jpg")
                if path:
                    media_paths.append(path)
        elif msg_type == "audio":
            mtype = MessageType.VOICE
            text = "[voice]"
            code = getattr(getattr(msg, "audio_content", None), "download_code", None) or ""
            if code:
                path = await self._download_resource(code, "voice.mp3")
                if path:
                    media_paths.append(path)
        elif msg_type == "richText":
            rich = getattr(msg, "rich_text_content", None)
            if rich:
                text, imgs = self._parse_rich_text(rich.rich_text_list or [])
                for code, fname in imgs:
                    path = await self._download_resource(code, fname)
                    if path:
                        media_paths.append(path)
                if imgs:
                    mtype = MessageType.IMAGE

        # Prefer raw_data dict (camelCase), fall back to ChatbotMessage attributes
        def _raw(key_camel: str, key_snake: str = "", default: Any = "") -> Any:
            v = raw_data.get(key_camel)
            if v is None and key_snake:
                v = raw_data.get(key_snake)
            if v is None:
                v = getattr(msg, key_camel, None) or (getattr(msg, key_snake, None) if key_snake else None)
            return v if v is not None else default

        sender_id = _raw("senderStaffId", "sender_staff_id") or _raw("senderId", "sender_id") or ""
        sender_name = _raw("senderNick", "sender_nick") or sender_id
        conv_id = str(_raw("conversationId", "conversation_id") or "")
        conv_type = str(_raw("conversationType", "conversation_type") or "1")
        conv_title = str(_raw("conversationTitle", "conversation_title") or "")
        # isInAtList and msgId live in the raw callback dict, not on ChatbotMessage
        is_in_at = (
            raw_data.get("isInAtList")
            or getattr(msg, "isInAtList", None)
            or getattr(msg, "is_in_at_list", False)
            or False
        )
        msg_id = (
            str(raw_data.get("msgId") or raw_data.get("msg_id") or "").strip()
            or getattr(msg, "msgId", None)
            or getattr(msg, "msg_id", "")
            or f"dt_{uuid.uuid4().hex}"
        )

        # Dedup: reject messages already in flight (DingTalk retransmits on timeout)
        with self._processing_msg_ids_lock:
            if msg_id in self._processing_msg_ids:
                logger.info("[dingtalk] dedup: msg_id %r already in progress, dropping", msg_id)
                return
            self._processing_msg_ids.add(msg_id)

        logger.info(
            "[dingtalk] recv conv_id=%r conv_type=%s sender=%r msg_id=%r is_in_at=%s has_webhook=%s text=%r",
            conv_id,
            conv_type,
            sender_id,
            msg_id,
            is_in_at,
            bool(getattr(msg, "sessionWebhook", None) or raw_data.get("sessionWebhook")),
            text[:80],
        )

        # Cache + persist session_webhook for replies
        webhook_url = str(msg.session_webhook or raw_data.get("sessionWebhook") or "")
        # session_webhook_expired_time is int (ms epoch) or None; default 0 means "no expiry known"
        webhook_exp = msg.session_webhook_expired_time or raw_data.get("sessionWebhookExpiredTime") or 0
        try:
            webhook_exp = int(webhook_exp) if webhook_exp else 0
        except (TypeError, ValueError):
            webhook_exp = 0
        # Use conv_id as the store key; for DM fallback to sender_id
        store_key = conv_id or sender_id
        if webhook_url and store_key:
            self._webhook_store[store_key] = {
                "url": webhook_url,
                "expired": webhook_exp,
                "ts": time.time(),
            }
            self._save_webhook_store()
            logger.info(
                "[dingtalk] stored webhook key=%r exp=%s url_prefix=%s", store_key, webhook_exp, webhook_url[:40]
            )
        else:
            logger.warning("[dingtalk] no sessionWebhook in message (conv_id=%r)", conv_id)

        mentioned = bool(is_in_at)
        if not mentioned:
            for pat in self.config.get("mention_patterns", []):
                if re.search(pat, raw_text):
                    mentioned = True
                    break

        # Always use conv_id (openConversationId) as chat_id so webhook
        # lookup and Open API both work. Fall back to sender_id for DMs
        # when conv_id is unavailable.
        chat_id = conv_id or sender_id
        if conv_type == "1":
            conv = ConversationContext(type=ConversationType.DM, chat_id=chat_id, is_dm=True)
        else:
            conv = ConversationContext(
                type=ConversationType.GROUP,
                chat_id=conv_id,
                group_id=conv_id,
                group_name=conv_title,
                mentioned=mentioned,
            )

        # Register reply_future so send() can resolve it on first successful send.
        # Store (stream_loop, future, msg_id) so resolutions use call_soon_threadsafe.
        if reply_future is not None and not reply_future.done():
            self._reply_futures[chat_id] = (asyncio.get_running_loop(), reply_future, msg_id)

        # Encode conv_id into message_id as "conv_id|msg_id" so reaction APIs
        # can extract both parts from event.message_id alone.
        event = MessageEvent(
            text=text,
            sender_id=sender_id,
            sender_name=sender_name,
            platform=self.name,
            message_id=f"{conv_id}|{msg_id}" if conv_id else msg_id,
            message_type=mtype,
            conversation=conv,
            media_paths=media_paths,
            raw={"conv_id": conv_id, "sender_id": sender_id},
        )
        logger.info("[dingtalk] calling _enqueue chat_id=%r msg_id=%r mentioned=%s", chat_id, msg_id, mentioned)
        await self._enqueue(event)
        logger.info("[dingtalk] _enqueue returned chat_id=%r", chat_id)

    def _parse_rich_text(self, elements: list) -> tuple[str, list[tuple[str, str]]]:
        texts: list[str] = []
        images: list[tuple[str, str]] = []
        for elem in elements:
            if isinstance(elem, dict):
                tag = elem.get("type", "")
                if tag == "text":
                    texts.append(elem.get("text", ""))
                elif tag in ("picture", "image"):
                    code = elem.get("downloadCode", "")
                    if code:
                        images.append((code, "image.jpg"))
            elif hasattr(elem, "type"):
                if elem.type == "text":
                    texts.append(getattr(elem, "text", ""))
                elif elem.type in ("picture", "image"):
                    code = getattr(elem, "downloadCode", "")
                    if code:
                        images.append((code, "image.jpg"))
        return " ".join(texts), images

    async def _download_resource(self, download_code: str, filename: str) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=30) as hc:
                if download_code.startswith("http"):
                    resp = await hc.get(download_code)
                else:
                    # Non-URL codes require the DingTalk download API
                    token = await self._get_access_token()
                    if not token:
                        return None
                    resp = await hc.post(
                        f"{_DINGTALK_API}/robot/messageFiles/download",
                        headers={"x-acs-dingtalk-access-token": token},
                        json={"downloadCode": download_code, "robotCode": self.config["client_id"]},
                    )
                if resp.status_code == 200:
                    path = self._media_cache / f"dt_{filename}"
                    path.write_bytes(resp.content)
                    return str(path)
        except Exception as e:
            logger.warning("[dingtalk] download failed: %s", e)
        return None

    def _get_webhook(self, chat_id: str) -> str | None:
        wh = self._webhook_store.get(chat_id)
        if not wh:
            return None
        if is_webhook_expired(wh.get("expired")):
            return None
        return wh["url"]

    # ── Emotion reactions ─────────────────────────────────────────────────

    # Map our emoji chars to DingTalk emotionName strings
    _EMOTION_NAMES: dict[str, str] = {
        "⏳": "🤔Thinking",
        "✅": "🥳Done",
        "❌": "😢Failed",
    }

    async def add_reaction(self, message_id: str, emoji: str) -> str | None:
        """Send a DingTalk emotion reaction to an incoming message.

        Returns the emoji_name string (used by remove_reaction to recall).
        message_id is encoded as "conv_id|msg_id".
        """
        parts = message_id.split("|", 1)
        if len(parts) != 2:
            return None
        conv_id, orig_msg_id = parts
        if not conv_id or not orig_msg_id:
            return None
        token = await self._get_access_token()
        if not token:
            return None
        emoji_name = self._EMOTION_NAMES.get(emoji, "🤔Thinking")
        text_emotion = {
            "emotionId": "2659900",
            "emotionName": emoji_name,
            "text": emoji_name,
            "backgroundId": "im_bg_1",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as hc:
                resp = await hc.post(
                    f"{_DINGTALK_API}/robot/emotion/reply",
                    headers={"x-acs-dingtalk-access-token": token},
                    json={
                        "robotCode": self.config["client_id"],
                        "openConversationId": conv_id,
                        "openMsgId": orig_msg_id,
                        "emotionType": 2,
                        "emotionName": emoji_name,
                        "textEmotion": text_emotion,
                    },
                )
                if resp.status_code == 200:
                    logger.info("[dingtalk] add_reaction %s on msg=%s", emoji_name, orig_msg_id[:20])
                    return emoji_name  # return name so remove_reaction can recall by name
                logger.debug("[dingtalk] add_reaction failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.debug("[dingtalk] add_reaction error: %s", e)
        return None

    async def remove_reaction(self, message_id: str, reaction_id: str) -> None:
        """Recall a previously sent emotion reaction.

        reaction_id is the emoji_name string returned by add_reaction.
        """
        parts = message_id.split("|", 1)
        if len(parts) != 2:
            return
        conv_id, orig_msg_id = parts
        if not conv_id or not orig_msg_id or not reaction_id:
            return
        token = await self._get_access_token()
        if not token:
            return
        emoji_name = reaction_id  # add_reaction returns the emoji_name
        text_emotion = {
            "emotionId": "2659900",
            "emotionName": emoji_name,
            "text": emoji_name,
            "backgroundId": "im_bg_1",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as hc:
                resp = await hc.post(
                    f"{_DINGTALK_API}/robot/emotion/recall",
                    headers={"x-acs-dingtalk-access-token": token},
                    json={
                        "robotCode": self.config["client_id"],
                        "openConversationId": conv_id,
                        "openMsgId": orig_msg_id,
                        "emotionType": 2,
                        "emotionName": emoji_name,
                        "textEmotion": text_emotion,
                    },
                )
                if resp.status_code == 200:
                    logger.info("[dingtalk] remove_reaction %s on msg=%s", emoji_name, orig_msg_id[:20])
                else:
                    logger.debug("[dingtalk] remove_reaction failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.debug("[dingtalk] remove_reaction error: %s", e)

    # ── Sending ───────────────────────────────────────────────────────────

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        secret = self.config.get("webhook_secret", "")
        if not secret:
            return True
        return verify_dingtalk_signature(headers, body, secret)

    async def send_typing(self, target: ReplyTarget) -> None:
        pass  # DingTalk has no native typing indicator

    def _resolve_reply_future(self, chat_id: str) -> None:
        """Resolve the pending reply_future for this chat with SENT_VIA_WEBHOOK sentinel."""
        entry = self._reply_futures.pop(chat_id, None)
        if not entry:
            return
        stream_loop, fut, msg_id = entry
        if not fut.done():
            stream_loop.call_soon_threadsafe(fut.set_result, _BotHandler._SENT_VIA_WEBHOOK)
        with self._processing_msg_ids_lock:
            self._processing_msg_ids.discard(msg_id)

    def resolve_stream_reply(self, chat_id: str, text: str) -> None:
        """Fallback: resolve reply_future with actual text so reply_text() sends it.

        Called by dispatcher when harness.run() is done but reply_future
        is still pending (e.g. sessionWebhook failed or produced no tokens).
        DingTalk's process() will then call reply_text(text, incoming_message).
        """
        entry = self._reply_futures.pop(chat_id, None)
        if not entry:
            return
        stream_loop, fut, msg_id = entry
        if not fut.done():
            logger.info("[dingtalk] reply_future resolved via fallback reply_text chat_id=%r", chat_id)
            stream_loop.call_soon_threadsafe(fut.set_result, text or " ")
        with self._processing_msg_ids_lock:
            self._processing_msg_ids.discard(msg_id)

    async def send(self, target: ReplyTarget, text: str, **kwargs) -> SendResult:
        text = text[: self.max_message_length]
        # Tier 1: session_webhook
        webhook = self._get_webhook(target.chat_id)
        logger.info("[dingtalk] send chat_id=%r has_webhook=%s text_len=%d", target.chat_id, bool(webhook), len(text))
        if webhook:
            await self._rate_acquire()
            logger.info("[dingtalk] rate_acquire done, POSTing to webhook url_prefix=%s", webhook[:40])
            payload = markdown_payload(text)
            try:
                async with httpx.AsyncClient(timeout=15) as hc:
                    resp = await hc.post(webhook, json=payload)
                    data = resp.json()
                    if resp.status_code == 200 and data.get("errcode") == 0:
                        logger.info(
                            "[dingtalk] sent via webhook chat_id=%r text_len=%d",
                            target.chat_id,
                            len(text),
                        )
                        self._resolve_reply_future(target.chat_id)
                        return SendResult(success=True)
                    logger.warning(
                        "[dingtalk] webhook send failed: status=%s body=%s, trying Open API", resp.status_code, data
                    )
            except Exception as e:
                logger.warning("[dingtalk] webhook send error: %s, trying Open API", e)

        # Tier 2: Open API (requires access_token + openConversationId)
        result = await self._send_via_open_api(target.chat_id, text)
        logger.info("[dingtalk] Open API result: success=%s error=%s", result.success, result.error)
        if result.success:
            self._resolve_reply_future(target.chat_id)
        return result

    async def _send_via_open_api(self, conv_id: str, text: str) -> SendResult:
        token = await self._get_access_token()
        if not token:
            return SendResult(success=False, error="no access token available", retryable=True)
        msg_key = self.config.get("msg_key", "sampleMarkdown")
        try:
            async with httpx.AsyncClient(timeout=15) as hc:
                resp = await hc.post(
                    f"{_DINGTALK_API}/robot/oToH/send",
                    headers={"x-acs-dingtalk-access-token": token},
                    json={
                        "robotCode": self.config["client_id"],
                        "openConversationId": conv_id,
                        "msgKey": msg_key,
                        "msgParam": json.dumps({"title": "Message", "text": text}),
                    },
                )
                data = resp.json()
                if resp.status_code == 200:
                    return SendResult(success=True, message_id=data.get("processQueryKey"))
                return SendResult(
                    success=False,
                    error=data.get("message", resp.text),
                    retryable=resp.status_code in (429, 500, 502, 503),
                )
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    # ── AI Card helpers ───────────────────────────────────────────────────

    async def _create_ai_card(self, conv_id: str, content: str) -> str | None:
        """Create an AI streaming card. Returns outTraceId or None."""
        token = await self._get_access_token()
        if not token:
            return None
        trace_id = str(uuid.uuid4())
        try:
            async with httpx.AsyncClient(timeout=15) as hc:
                resp = await hc.post(
                    f"{_DINGTALK_API}/card/instances",
                    headers={"x-acs-dingtalk-access-token": token},
                    json={
                        "cardTemplateId": self.config["card_template_id"],
                        "outTraceId": trace_id,
                        "callbackType": "STREAM",
                        "openSpaceId": f"dtv1.card//IM_GROUP.{conv_id}",
                        "openDeliverOptions": {"robotCode": self.config["client_id"]},
                        "cardData": {"cardParamMap": {"content": content, "status": "PROCESSING"}},
                    },
                )
                if resp.status_code == 200:
                    self._active_cards[trace_id] = {"conv_id": conv_id, "content": content}
                    self._save_active_cards()
                    return trace_id
        except Exception as e:
            logger.debug("[dingtalk] create_ai_card error: %s", e)
        return None

    async def _update_ai_card(self, trace_id: str, content: str, status: str = "INPUTING") -> None:
        token = await self._get_access_token()
        if not token:
            return
        try:
            async with httpx.AsyncClient(timeout=15) as hc:
                await hc.put(
                    f"{_DINGTALK_API}/card/instances/{trace_id}",
                    headers={"x-acs-dingtalk-access-token": token},
                    json={"cardData": {"cardParamMap": {"content": content, "status": status}}},
                )
            if status in ("FINISHED", "FAILED"):
                self._active_cards.pop(trace_id, None)
                self._save_active_cards()
            else:
                if trace_id in self._active_cards:
                    self._active_cards[trace_id]["content"] = content
                    self._save_active_cards()
        except Exception as e:
            logger.debug("[dingtalk] update_ai_card error: %s", e)

    # ── Streaming ─────────────────────────────────────────────────────────

    async def send_stream(
        self,
        target: ReplyTarget,
        queue: asyncio.Queue,
        edit_interval: float = 0.5,
        buffer_threshold: int = 15,
    ) -> SendResult:
        template_id = self.config.get("card_template_id", "")

        if template_id:
            return await self._send_stream_ai_card(target, queue, edit_interval, buffer_threshold)

        # No card template: accumulate all tokens and send once
        buf = ""
        while True:
            try:
                delta = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if delta is None:
                break
            if isinstance(delta, dict):
                kind = delta.get("type")
                if kind == "tool_start":
                    name = delta.get("name", "tool")
                    if buf:
                        await self._send_with_retry(target, buf)
                        buf = ""
                    await self._send_with_retry(target, f"⚙️ {name}…")
                continue
            buf += delta

        if buf:
            return await self._send_with_retry(target, buf)
        return SendResult(success=True)

    async def _send_stream_ai_card(
        self,
        target: ReplyTarget,
        queue: asyncio.Queue,
        edit_interval: float,
        buffer_threshold: int,
    ) -> SendResult:
        conv_id = target.chat_id
        buf = ""
        trace_id: str | None = None
        card_failed = False  # set True after first failed creation; avoids per-token retries
        last_update = 0.0

        while True:
            try:
                delta = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if delta is None:
                break

            if isinstance(delta, dict):
                kind = delta.get("type")
                if kind == "tool_start":
                    name = delta.get("name", "tool")
                    tool_msg = f"⚙️ {name}…"
                    if not card_failed:
                        if trace_id and buf:
                            await self._update_ai_card(trace_id, buf, "INPUTING")
                            trace_id = await self._create_ai_card(conv_id, tool_msg)
                            if trace_id is None:
                                card_failed = True
                        elif trace_id:
                            await self._update_ai_card(trace_id, tool_msg, "INPUTING")
                        else:
                            trace_id = await self._create_ai_card(conv_id, tool_msg)
                            if trace_id is None:
                                card_failed = True
                    buf = ""
                    last_update = time.time()
                continue

            buf += delta
            now = time.time()
            if trace_id is None and not card_failed:
                trace_id = await self._create_ai_card(conv_id, buf)
                if trace_id is None:
                    card_failed = True
                last_update = now
            elif trace_id and (
                len(buf) >= buffer_threshold or now - last_update >= max(edit_interval, _AI_CARD_STREAM_MIN_INTERVAL)
            ):
                await self._update_ai_card(trace_id, buf, "INPUTING")
                last_update = now

        if trace_id:
            await self._update_ai_card(trace_id, buf or "(done)", "FINISHED")
            self._resolve_reply_future(target.chat_id)
        elif buf:
            return await self._send_with_retry(target, buf)

        return SendResult(success=True)

    async def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        # Unblock any waiting reply_futures on shutdown
        for stream_loop, fut, msg_id in list(self._reply_futures.values()):
            if not fut.done():
                stream_loop.call_soon_threadsafe(fut.set_result, _BotHandler._SENT_VIA_WEBHOOK)
        self._reply_futures.clear()
        with self._processing_msg_ids_lock:
            self._processing_msg_ids.clear()
        if self._stream_client:
            # Close the WebSocket to unblock start()'s `async for` loop
            ws = getattr(self._stream_client, "websocket", None)
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass

    def system_prompt(self) -> str:
        return "Reply format: DingTalk Markdown (supports headings, lists, and code blocks)."


register_builtin(DingTalkChannel)
