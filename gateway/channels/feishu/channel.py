from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        UpdateMessageRequest,
        UpdateMessageRequestBody,
        GetMessageResourceRequest,
        ReplyMessageRequest,
        ReplyMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        DeleteMessageReactionRequest,
        CreateImageRequest,
        CreateImageRequestBody,
        CreateFileRequest,
        CreateFileRequestBody,
    )
    from lark_oapi.ws import Client as WSClient
    import lark_oapi.ws.client as _lark_ws_mod

    class _EventLoopProxy:
        """Resolve lark_oapi.ws.client.loop to the calling thread's running loop.

        Prevents "Future attached to a different loop" errors across WS reconnects
        by dynamically delegating to asyncio.get_running_loop() rather than holding
        a stale reference to a previous thread's event loop.
        """

        def __getattr__(self, name: str):
            try:
                return getattr(asyncio.get_running_loop(), name)
            except RuntimeError:
                return getattr(asyncio.get_event_loop(), name)

    _lark_ws_mod.loop = _EventLoopProxy()

except ImportError as _e:
    raise ImportError("Feishu channel requires 'lark-oapi>=1.3.0'. Install with: pip install lark-oapi") from _e

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
from .constants import (
    FEISHU_MAX_TEXT_LEN,
    MSG_TYPE_AUDIO,
    MSG_TYPE_FILE,
    MSG_TYPE_IMAGE,
    MSG_TYPE_POST,
)
from .formatter import (
    build_text_payload_candidates,
)
from .utils import verify_feishu_signature, strip_mentions, extract_post_text, truncate, decrypt_feishu_payload

logger = logging.getLogger(__name__)


class FeishuChannel(BaseChannel):
    name = "feishu"
    display_name = "Feishu"
    stall_timeout = 3600.0
    stream_edit_interval = 0.2
    stream_buffer_threshold = 10

    supports_reactions = True
    supports_threads = True
    max_message_length = 30000

    config_schema = {
        "type": "object",
        "required": ["app_id", "app_secret"],
        "properties": {
            "app_id": {"type": "string", "title": "App ID"},
            "app_secret": {"type": "string", "title": "App Secret", "format": "password"},
            "verification_token": {"type": "string", "title": "Verification Token (webhook mode)"},
            "encrypt_key": {"type": "string", "title": "Encrypt Key (AES encryption, optional)", "format": "password"},
            "mode": {"type": "string", "enum": ["websocket", "webhook"], "default": "websocket"},
            "require_mention": {"type": "boolean", "default": True, "title": "Require @mention in group chats"},
            "reply_in_thread": {"type": "boolean", "default": False, "title": "Reply in thread (reduce channel noise)"},
            "interactive_table": {
                "type": "boolean",
                "default": True,
                "title": "Render markdown tables as interactive cards",
            },
            "session_mode": {"type": "string", "enum": ["shared", "per_user"], "default": "shared"},
        },
    }

    def __init__(self, config: dict, dispatcher) -> None:
        super().__init__(config, dispatcher)
        self._client: lark.Client | None = None
        self._ws: WSClient | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = asyncio.Event()
        self._bot_open_id: str | None = None
        _media_dir_cfg = config.get("_workspace_media_dir")
        self._media_cache = Path(_media_dir_cfg) if _media_dir_cfg else get_media_cache_dir()
        self._receive_id_store: dict[str, dict] = {}
        self._load_receive_id_store()

    # ── Receive-ID persistence (for proactive / cron sends) ────────────────

    @property
    def _receive_id_store_path(self) -> Path:
        return get_store_dir() / "feishu_receive_ids.json"

    def _load_receive_id_store(self) -> None:
        try:
            if self._receive_id_store_path.exists():
                self._receive_id_store = json.loads(self._receive_id_store_path.read_text(encoding="utf-8"))
        except Exception:
            self._receive_id_store = {}

    def _save_receive_id_store(self) -> None:
        try:
            self._receive_id_store_path.write_text(json.dumps(self._receive_id_store), encoding="utf-8")
        except Exception as e:
            logger.debug("[feishu] receive_id store write failed: %s", e)

    def _record_receive_id(self, key: str, chat_id: str, id_type: str = "chat_id") -> None:
        self._receive_id_store[key] = {"receive_id": chat_id, "type": id_type, "ts": time.time()}
        if len(self._receive_id_store) > 2000:
            cutoff = time.time() - 30 * 86400
            self._receive_id_store = {k: v for k, v in self._receive_id_store.items() if v.get("ts", 0) > cutoff}
        self._save_receive_id_store()

    def resolve_receive_id(self, session_id: str) -> tuple[str, str] | None:
        """Return (receive_id, id_type) for a known session, or None.

        Supports suffix matching: "feishu-g-xxx#1" → tries prefix "feishu-g-xxx".
        Used by cron tasks for proactive sends.
        """
        entry = self._receive_id_store.get(session_id)
        if entry:
            return entry["receive_id"], entry["type"]
        prefix = session_id.split("#")[0]
        candidates = [(sid, v) for sid, v in self._receive_id_store.items() if sid.startswith(prefix)]
        if candidates:
            _, best = max(candidates, key=lambda x: x[1].get("ts", 0))
            return best["receive_id"], best["type"]
        return None

    def resolve_session_id(self, event: MessageEvent) -> str:
        """Keep one session per group chat (threads/topics do not split sessions)."""
        ctx = event.conversation
        if ctx.type in (ConversationType.GROUP, ConversationType.TOPIC):
            mode = self.config.get("session_mode", "shared")
            if mode == "per_user":
                return f"{self.name}-g-{ctx.chat_id}-u-{event.sender_id}"
            return f"{self.name}-g-{ctx.chat_id}"
        return super().resolve_session_id(event)

    # ── Connection ─────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        app_id = self.config["app_id"]
        app_secret = self.config["app_secret"]
        self._client = (
            lark.Client.builder().app_id(app_id).app_secret(app_secret).log_level(lark.LogLevel.WARNING).build()
        )
        # Resolve bot's own open_id for accurate @mention detection
        # Uses GET /open-apis/bot/v3/info via raw BaseRequest
        try:
            from lark_oapi.core.model.base_request import BaseRequest, HttpMethod
            from lark_oapi.core.token import AccessTokenType

            req = (
                BaseRequest.builder()
                .http_method(HttpMethod.GET)
                .uri("/open-apis/bot/v3/info")
                .token_types({AccessTokenType.TENANT})
                .build()
            )
            resp = await asyncio.to_thread(self._client.request, req)
            data = getattr(resp, "data", None) or {}
            if isinstance(data, dict):
                bot_info = data.get("bot", {})
                self._bot_open_id = bot_info.get("open_id")
            else:
                self._bot_open_id = getattr(getattr(data, "bot", None), "open_id", None)
            if self._bot_open_id:
                logger.info("[feishu] bot open_id=%s", self._bot_open_id)
        except Exception as e:
            logger.debug("[feishu] could not resolve bot open_id: %s", e)
        # WSClient is created in _listen_websocket() after the handler is built

    async def _listen(self) -> None:
        mode = self.config.get("mode", "websocket")
        if mode == "websocket":
            await self._listen_websocket()
        else:
            await self._stop_event.wait()

    async def _listen_websocket(self) -> None:
        import threading
        from lark_oapi.api.im.v1.model import P2ImMessageReceiveV1

        app_id = self.config["app_id"]
        app_secret = self.config["app_secret"]
        main_loop = asyncio.get_running_loop()

        def sync_handler(data: P2ImMessageReceiveV1) -> None:
            logger.info("[feishu] sync_handler called — scheduling _on_message")
            asyncio.run_coroutine_threadsafe(self._on_message(data), main_loop)

        handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(sync_handler).build()

        exc_bucket: list[Exception] = []
        ws_done = asyncio.Event()

        def _ws_thread() -> None:
            thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(thread_loop)
            ws = WSClient(
                app_id,
                app_secret,
                log_level=lark.LogLevel.WARNING,
                event_handler=handler,
            )
            self._ws = ws
            self._ws_loop = thread_loop
            try:
                ws.start()
            except Exception as exc:
                exc_bucket.append(exc)
            finally:
                try:
                    thread_loop.close()
                except Exception:
                    pass
                main_loop.call_soon_threadsafe(ws_done.set)

        thread = threading.Thread(target=_ws_thread, daemon=True, name="feishu-ws")
        thread.start()
        await ws_done.wait()
        if exc_bucket:
            raise exc_bucket[0]

    async def _on_message(self, data) -> None:
        logger.info("[feishu] _on_message called")
        try:
            await self._on_message_inner(data)
        except Exception as exc:
            logger.exception("[feishu] _on_message unhandled error: %s", exc)

    async def _on_message_inner(self, data) -> None:
        try:
            msg = data.event.message
            sender = data.event.sender
        except AttributeError as exc:
            logger.warning("[feishu] _on_message: bad data structure: %s", exc)
            return

        msg_type = msg.message_type or "text"
        body = json.loads(msg.content) if msg.content else {}

        text = ""
        media_paths: list[str] = []
        mtype = MessageType.TEXT

        if msg_type == "text":
            text = strip_mentions(body.get("text", ""))
        elif msg_type == MSG_TYPE_IMAGE:
            mtype = MessageType.IMAGE
            text = "[image]"
            image_key = body.get("image_key", "")
            if image_key:
                path = await self._download_resource(msg.message_id, image_key, "image")
                if path:
                    media_paths.append(path)
        elif msg_type == MSG_TYPE_AUDIO:
            mtype = MessageType.VOICE
            text = "[voice]"
            file_key = body.get("file_key", "")
            if file_key:
                path = await self._download_resource(msg.message_id, file_key, "audio")
                if path:
                    media_paths.append(path)
        elif msg_type == MSG_TYPE_FILE:
            mtype = MessageType.FILE
            text = f"[file: {body.get('file_name', 'unknown')}]"
            file_key = body.get("file_key", "")
            if file_key:
                path = await self._download_resource(msg.message_id, file_key, "file")
                if path:
                    media_paths.append(path)
        elif msg_type == MSG_TYPE_POST:
            mtype = MessageType.TEXT
            text = extract_post_text(body)
        else:
            text = f"[{msg_type}]"

        chat_type = msg.chat_type or "p2p"
        open_id = sender.sender_id.open_id if sender.sender_id else ""
        chat_id = msg.chat_id or open_id
        feishu_thread_id = getattr(msg, "thread_id", None) or None

        if chat_type == "p2p":
            conv = ConversationContext(
                type=ConversationType.DM,
                chat_id=chat_id,  # oc_xxx DM chat id, falls back to open_id
                is_dm=True,
            )
        else:
            mentions = msg.mentions or []
            bot_oid = self._bot_open_id
            if bot_oid:
                # Match against the bot's actual open_id (sender_id field inside mention)
                mentioned = any(
                    getattr(getattr(m, "id", None), "open_id", None) == bot_oid or getattr(m, "key", "") == "@all"
                    for m in mentions
                )
            else:
                # Fallback: positional key — only works when bot is first mention
                mentioned = any(getattr(m, "key", "") in ("@_user_1", "@all") for m in mentions)
            if feishu_thread_id:
                conv = ConversationContext(
                    type=ConversationType.TOPIC,
                    chat_id=chat_id,
                    group_id=chat_id,
                    topic_id=feishu_thread_id,
                    mentioned=mentioned,
                )
            else:
                conv = ConversationContext(
                    type=ConversationType.GROUP,
                    chat_id=chat_id,
                    group_id=chat_id,
                    mentioned=mentioned,
                )

        raw: dict = {}
        if feishu_thread_id:
            raw["feishu_thread_id"] = feishu_thread_id

        event = MessageEvent(
            text=text,
            sender_id=open_id,
            sender_name=open_id,
            platform=self.name,
            message_id=msg.message_id or "",
            message_type=mtype,
            conversation=conv,
            media_paths=media_paths,
            raw=raw,
        )
        logger.info(
            "[feishu] enqueue: chat_type=%s conv_type=%s mentioned=%s text=%r",
            chat_type,
            conv.type,
            getattr(conv, "mentioned", None),
            text[:50],
        )
        await self._enqueue(event)

    def make_reply_target(self, event: MessageEvent) -> ReplyTarget:
        ctx = event.conversation
        feishu_thread_id = event.raw.get("feishu_thread_id")

        if feishu_thread_id:
            # For existing Feishu threads, reply by message_id with reply_in_thread=true.
            # thread_id is kept for context only; quote_message_id is the anchor.
            return ReplyTarget(
                chat_id=ctx.chat_id,
                thread_id=feishu_thread_id,
                quote_message_id=event.message_id,
            )
        elif self.config.get("reply_in_thread", False) and ctx.type != ConversationType.DM:
            return ReplyTarget(chat_id=ctx.chat_id, quote_message_id=event.message_id)
        else:
            return ReplyTarget(chat_id=ctx.chat_id)

    async def _download_resource(self, message_id: str, key: str, rtype: str) -> str | None:
        try:
            req = GetMessageResourceRequest.builder().message_id(message_id).file_key(key).type(rtype).build()
            resp = await asyncio.to_thread(self._client.im.v1.message_resource.get, req)
            if not resp.success():
                logger.warning("[feishu] download failed: %s", resp.msg)
                return None
            suffix = {"image": ".jpg", "audio": ".ogg", "file": ""}.get(rtype, "")
            path = self._media_cache / f"feishu_{key}{suffix}"
            path.write_bytes(resp.file.read())
            return str(path)
        except Exception as e:
            logger.error("[feishu] download error: %s", e)
            return None

    # ── Webhook entry point ───────────────────────────────────────────────

    async def _on_webhook(self, payload: dict) -> None:
        encrypt_key = self.config.get("encrypt_key", "")
        if "encrypt" in payload and encrypt_key:
            try:
                payload = decrypt_feishu_payload(payload["encrypt"], encrypt_key)
            except Exception as e:
                logger.warning("[feishu] webhook decryption failed: %s", e)
                return

        if "challenge" in payload:
            return

        header = payload.get("header", {})
        event_type = header.get("event_type", "")
        if event_type != "im.message.receive_v1":
            return

        evt = payload.get("event", {})
        msg_data = evt.get("message", {})
        sender_data = evt.get("sender", {})

        _open_id = sender_data.get("sender_id", {}).get("open_id", "")

        class _SenderID:
            open_id = _open_id

        class _Sender:
            sender_id = _SenderID()

        # Reconstruct mentions so that mention-detection in _on_message works correctly
        class _MentionID:
            def __init__(self, d: dict) -> None:
                self.open_id = d.get("id", {}).get("open_id", "")

        class _Mention:
            def __init__(self, d: dict) -> None:
                self.key = d.get("key", "")
                self.id = _MentionID(d)

        _mentions = [_Mention(m) for m in msg_data.get("mentions", [])]

        _msg_id = msg_data.get("message_id", "")
        _msg_type = msg_data.get("message_type", "text")
        _chat_id = msg_data.get("chat_id", "")
        _chat_type = msg_data.get("chat_type", "p2p")
        _thread_id = msg_data.get("thread_id", None)
        _content = msg_data.get("content", "{}")

        class _Msg:
            message_id = _msg_id
            message_type = _msg_type
            chat_id = _chat_id
            chat_type = _chat_type
            thread_id = _thread_id
            content = _content
            mentions = _mentions

        class _Event:
            message = _Msg()
            sender = _Sender()

        class _Data:
            event = _Event()

        await self._on_message(_Data())

    # ── Sending ───────────────────────────────────────────────────────────

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        token = self.config.get("verification_token", "")
        if not token:
            logger.warning("[feishu] verification_token not configured — rejecting webhook request")
            return False
        return verify_feishu_signature(headers, body, token)

    async def send_typing(self, target: ReplyTarget) -> None:
        pass  # Feishu has no native typing indicator

    def _prefer_interactive_table(self) -> bool:
        """
        Whether markdown tables should be sent as interactive cards.
        """
        raw = self.config.get("interactive_table")
        if raw is None:
            return True
        return bool(raw)

    def _text_payload_candidates(
        self,
        text: str,
        *,
        allow_interactive: bool = True,
    ) -> list[tuple[str, str]]:
        """
        Build outbound `(msg_type, content)` candidates in fallback order.
        """
        body = truncate(text or "", FEISHU_MAX_TEXT_LEN)
        return build_text_payload_candidates(
            body,
            prefer_interactive_table=self._prefer_interactive_table(),
            allow_interactive=allow_interactive,
        )

    async def send(self, target: ReplyTarget, text: str, **kwargs) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="not connected")

        if target.thread_id:
            anchor_id = target.quote_message_id
            if not anchor_id:
                logger.warning("[feishu] thread reply missing anchor message_id, fallback to chat send")
            msg_id, _ = (
                await self._reply_in_thread(
                    anchor_id,
                    text,
                    allow_interactive=True,
                )
                if anchor_id
                else (None, None)
            )
            if msg_id:
                self._record_receive_id(f"feishu-thread-{target.thread_id}", target.chat_id)
                return SendResult(success=True, message_id=msg_id)
            if anchor_id:
                return SendResult(success=False, error="reply_in_thread failed", retryable=True)

        if target.quote_message_id:
            msg_id, thread_id = await self._reply_in_thread(
                target.quote_message_id,
                text,
                allow_interactive=True,
            )
            if msg_id:
                if thread_id:
                    self._record_receive_id(f"feishu-thread-{thread_id}", target.chat_id)
                return SendResult(success=True, message_id=msg_id)

        last_error = "send failed"
        retryable = False
        for msg_type, content in self._text_payload_candidates(text, allow_interactive=True):
            await self._rate_acquire()
            req = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(target.chat_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                )
                .build()
            )
            try:
                resp = await asyncio.to_thread(self._client.im.v1.message.create, req)
                if resp.success():
                    self._record_receive_id(f"feishu-chat-{target.chat_id}", target.chat_id)
                    return SendResult(success=True, message_id=resp.data.message_id)
                last_error = resp.msg or f"{msg_type} send failed"
                retryable = retryable or (resp.code in (99991400, 99991401, 99991663))
                logger.warning("[feishu] send failed with %s: %s", msg_type, last_error)
            except Exception as e:
                last_error = str(e)
                retryable = True
                logger.warning("[feishu] send error with %s: %s", msg_type, e)
        return SendResult(success=False, error=last_error, retryable=retryable)

    async def _reply_in_thread(
        self,
        message_id: str | None,
        text: str,
        *,
        allow_interactive: bool = False,
    ) -> tuple[str | None, str | None]:
        if not message_id:
            return None, None
        for msg_type, content in self._text_payload_candidates(
            text,
            allow_interactive=allow_interactive,
        ):
            await self._rate_acquire()
            req = (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .content(content)
                    .msg_type(msg_type)
                    .reply_in_thread(True)
                    .uuid(str(uuid.uuid4()))
                    .build()
                )
                .build()
            )
            try:
                resp = await asyncio.to_thread(self._client.im.v1.message.reply, req)
                if resp.success():
                    thread_id = getattr(resp.data, "thread_id", None)
                    return resp.data.message_id, thread_id
                logger.warning("[feishu] reply_in_thread failed with %s: %s", msg_type, resp.msg)
            except Exception as e:
                logger.warning("[feishu] reply_in_thread error with %s: %s", msg_type, e)
        return None, None

    async def _create_in_thread(
        self,
        thread_id: str,
        text: str,
        *,
        allow_interactive: bool = True,
    ) -> tuple[str | None, str | None]:
        _ = (text, allow_interactive)
        logger.warning(
            "[feishu] create_in_thread by thread_id is not supported; "
            "use _reply_in_thread(message_id, reply_in_thread=true). "
            "thread_id=%s",
            thread_id,
        )
        return None, None

    async def _edit_message(self, msg_id: str, text: str) -> None:
        if not self._client:
            return
        for msg_type, content in self._text_payload_candidates(
            text,
            allow_interactive=False,
        ):
            await self._rate_acquire()
            req = (
                UpdateMessageRequest.builder()
                .message_id(msg_id)
                .request_body(UpdateMessageRequestBody.builder().content(content).msg_type(msg_type).build())
                .build()
            )
            try:
                resp = await asyncio.to_thread(self._client.im.v1.message.update, req)
                if resp is None or getattr(resp, "success", lambda: True)():
                    return
                logger.warning("[feishu] edit_message failed with %s: %s", msg_type, getattr(resp, "msg", ""))
            except Exception as e:
                logger.warning("[feishu] edit_message error with %s: %s", msg_type, e)

    # ── Reactions ─────────────────────────────────────────────────────────

    async def add_reaction(self, message_id: str, emoji: str) -> str | None:
        if not self._client:
            return None
        # Feishu emoji_type strings; "✅" is intentionally absent — the reply
        # itself is the success signal (hermes-agent design decision).
        _emoji_map = {"⏳": "Typing", "❌": "CrossMark"}
        reaction_type = _emoji_map.get(emoji)
        if not reaction_type:
            return None
        try:
            req = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(
                    CreateMessageReactionRequestBody.builder().reaction_type({"emoji_type": reaction_type}).build()
                )
                .build()
            )
            resp = await asyncio.to_thread(self._client.im.v1.message_reaction.create, req)
            if resp.success():
                return getattr(resp.data, "reaction_id", None)
        except Exception as e:
            logger.debug("[feishu] add_reaction error: %s", e)
        return None

    async def remove_reaction(self, message_id: str, reaction_id: str) -> None:
        if not self._client or not reaction_id:
            return
        try:
            req = DeleteMessageReactionRequest.builder().message_id(message_id).reaction_id(reaction_id).build()
            await asyncio.to_thread(self._client.im.v1.message_reaction.delete, req)
        except Exception as e:
            logger.debug("[feishu] remove_reaction error: %s", e)

    # ── Media sending ─────────────────────────────────────────────────────

    async def send_photo(self, target: ReplyTarget, path: str, caption: str = "") -> SendResult:
        """Upload an image file and send it to the chat."""
        if not self._client:
            return SendResult(success=False, error="not connected")
        try:
            with open(path, "rb") as f:
                req = (
                    CreateImageRequest.builder()
                    .request_body(CreateImageRequestBody.builder().image_type("message").image(f).build())
                    .build()
                )
                resp = await asyncio.to_thread(self._client.im.v1.image.create, req)
            if not resp.success():
                return SendResult(success=False, error=resp.msg, retryable=True)
            image_key = resp.data.image_key
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

        await self._rate_acquire()
        req2 = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(target.chat_id)
                .msg_type("image")
                .content(json.dumps({"image_key": image_key}))
                .build()
            )
            .build()
        )
        try:
            resp2 = await asyncio.to_thread(self._client.im.v1.message.create, req2)
            if resp2.success():
                if caption:
                    await self.send(target, caption)
                return SendResult(success=True, message_id=resp2.data.message_id)
            return SendResult(success=False, error=resp2.msg, retryable=True)
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_document(self, target: ReplyTarget, path: str, filename: str = "", caption: str = "") -> SendResult:
        """Upload a file and send it to the chat."""
        if not self._client:
            return SendResult(success=False, error="not connected")
        import mimetypes
        import os

        fname = filename or os.path.basename(path)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        # Map MIME to Feishu file_type
        _type_map = {
            "application/pdf": "pdf",
            "application/zip": "zip",
            "application/msword": "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.ms-excel": "xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/vnd.ms-powerpoint": "ppt",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        }
        file_type = _type_map.get(mime, "stream")
        try:
            with open(path, "rb") as f:
                req = (
                    CreateFileRequest.builder()
                    .request_body(CreateFileRequestBody.builder().file_type(file_type).file_name(fname).file(f).build())
                    .build()
                )
                resp = await asyncio.to_thread(self._client.im.v1.file.create, req)
            if not resp.success():
                return SendResult(success=False, error=resp.msg, retryable=True)
            file_key = resp.data.file_key
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

        await self._rate_acquire()
        req2 = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(target.chat_id)
                .msg_type("file")
                .content(json.dumps({"file_key": file_key}))
                .build()
            )
            .build()
        )
        try:
            resp2 = await asyncio.to_thread(self._client.im.v1.message.create, req2)
            if resp2.success():
                if caption:
                    await self.send(target, caption)
                return SendResult(success=True, message_id=resp2.data.message_id)
            return SendResult(success=False, error=resp2.msg, retryable=True)
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    # ── Streaming ─────────────────────────────────────────────────────────

    async def send_stream(
        self,
        target: ReplyTarget,
        queue: asyncio.Queue,
        edit_interval: float = 0.2,
        buffer_threshold: int = 10,
    ) -> SendResult:
        """Segment-based streaming: accumulate tokens, send complete segments.

        Feishu's edit API is rate-limited to 5 rps and prone to locking under
        high-frequency token-by-token edits.  Instead we flush a complete
        buffer as a *new* message on each tool_start boundary and at the end.
        """
        buf = ""
        last_sent_id: str | None = None
        feishu_thread_id: str | None = target.thread_id
        thread_anchor_msg_id: str | None = target.quote_message_id

        async def _flush(text: str) -> str | None:
            nonlocal feishu_thread_id
            nonlocal thread_anchor_msg_id
            if not text.strip():
                return None
            if feishu_thread_id:
                msg_id, _ = await self._reply_in_thread(
                    thread_anchor_msg_id,
                    text,
                    allow_interactive=True,
                )
                if msg_id:
                    thread_anchor_msg_id = msg_id
                return msg_id
            if target.quote_message_id:
                msg_id, tid = await self._reply_in_thread(
                    target.quote_message_id,
                    text,
                    allow_interactive=True,
                )
                if tid:
                    feishu_thread_id = tid
                return msg_id
            result = await self.send(ReplyTarget(chat_id=target.chat_id), text)
            return result.message_id if result.success else None

        while True:
            try:
                delta = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if delta is None:
                break

            if isinstance(delta, dict):
                if delta.get("type") == "tool_start":
                    # Flush accumulated text before showing tool indicator
                    if buf.strip():
                        last_sent_id = await _flush(buf)
                        buf = ""
                    await _flush(f"⚙️ {delta.get('name', 'tool')}…")
                continue

            buf += delta

        # Flush any remaining buffer
        if buf.strip():
            last_sent_id = await _flush(buf)

        return SendResult(success=True, message_id=last_sent_id)

    async def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        ws_loop = self._ws_loop
        if ws_loop and not ws_loop.is_closed():
            try:
                ws_loop.call_soon_threadsafe(ws_loop.stop)
            except Exception as e:
                logger.debug("[feishu] ws stop error: %s", e)

    def system_prompt(self) -> str:
        return "Reply format: use Feishu Markdown (supports **bold**, `code`, and headings)."

    async def on_session_reset(self, session_id: str) -> None:
        pass


register_builtin(FeishuChannel)
