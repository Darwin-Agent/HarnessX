# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
import logging
import mimetypes
import os
import shlex
import uuid
from contextvars import ContextVar
from typing import Any

from .base import Tool, ToolResult
from ._dict_registry_mixin import _DictRegistryMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level context: storage dir for large/media outputs.
# RunLoop sets this per-step; falls back to a system temp dir when unset.
# ---------------------------------------------------------------------------

_mcp_results_ctx: ContextVar[str | None] = ContextVar("_mcp_results_ctx", default=None)

_MCP_TEXT_THRESHOLD = 50_000  # chars; single result larger than this is spilled to disk
_MCP_MEDIA_INLINE_LIMIT = 1_048_576  # bytes; blobs larger than this are always saved to disk
_TURN_BUDGET_CHARS = 200_000  # chars; aggregate tool-result budget per step


def _results_dir(override: str | None = None) -> str:
    """Return the active MCP results directory, creating it if necessary."""
    import tempfile

    d = override or _mcp_results_ctx.get(None) or os.path.join(tempfile.gettempdir(), "mcp_results")
    os.makedirs(d, exist_ok=True)
    return d


def _ext_for_mime(mime: str) -> str:
    """Best-effort file extension from a MIME type string."""
    ext = mimetypes.guess_extension(mime.split(";")[0].strip(), strict=False)
    if ext and ext != ".jpe":  # mimetypes maps image/jpeg to .jpe on some platforms
        return ext
    # Fallback table for common types mimetypes may not know
    _FALLBACK = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/wav": ".wav",
        "audio/ogg": ".ogg",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "application/pdf": ".pdf",
    }
    return _FALLBACK.get(mime.split(";")[0].strip(), ".bin")


def _save_bytes(data: bytes, mime: str, tool_name: str, results_dir: str | None = None) -> str:
    """Write *data* to a unique file under the active results dir. Returns the path."""
    ext = _ext_for_mime(mime)
    slug = tool_name.replace("/", "_")[:30]
    name = f"{slug}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(_results_dir(results_dir), name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _save_text(text: str, tool_name: str, results_dir: str | None = None) -> str:
    """Write *text* to a .txt file under the active results dir. Returns the path."""
    slug = tool_name.replace("/", "_")[:30]
    name = f"{slug}_{uuid.uuid4().hex[:8]}.txt"
    path = os.path.join(_results_dir(results_dir), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _modality_of(mime: str) -> str:
    """Return "image", "audio", or "video" for supported types; "other" otherwise."""
    m = mime.split("/")[0].lower()
    return m if m in ("image", "audio", "video") else "other"


def _extract_media_bytes(block) -> tuple[bytes, str] | None:
    """Extract raw bytes and MIME type from an ImageContent or EmbeddedResource blob block."""
    if hasattr(block, "data") and hasattr(block, "mimeType"):
        mime = block.mimeType or "application/octet-stream"
        raw = base64.b64decode(block.data) if isinstance(block.data, str) else block.data
        return raw, mime
    resource = getattr(block, "resource", None)
    if resource is not None and hasattr(resource, "blob") and resource.blob is not None:
        mime = getattr(resource, "mimeType", None) or "application/octet-stream"
        raw = base64.b64decode(resource.blob) if isinstance(resource.blob, str) else resource.blob
        return raw, mime
    return None


def _msg_text_len(content: str | list) -> int:
    """Return the total character count of a message content value.

    For list-form content, counts text characters plus the length of any
    inline base64 payloads (image source.data, input_audio.data) so that
    aggregate-budget enforcement accounts for the full wire size.
    """
    if isinstance(content, str):
        return len(content)
    total = 0
    for b in content:
        if not isinstance(b, dict):
            continue
        btype = b.get("type", "")
        if btype == "text":
            total += len(b.get("text", ""))
        elif btype == "image":
            total += len(b.get("source", {}).get("data", ""))
        elif btype == "input_audio":
            total += len(b.get("input_audio", {}).get("data", ""))
    return total


async def enforce_turn_budget(
    tool_messages: list,
    results_dir: str | None = None,
) -> list:
    """Spill the largest tool-result messages to disk until the aggregate fits within
    ``_TURN_BUDGET_CHARS``.  Already-truncated results (containing ``[truncated``) are
    skipped.  Returns a new list (same objects unless a message was replaced)."""
    import dataclasses

    total = sum(_msg_text_len(m.content) for m in tool_messages)
    if total <= _TURN_BUDGET_CHARS:
        return tool_messages

    candidates = sorted(
        [(i, m) for i, m in enumerate(tool_messages) if "[truncated" not in str(m.content)],
        key=lambda x: _msg_text_len(x[1].content),
        reverse=True,
    )

    result = list(tool_messages)
    for idx, msg in candidates:
        if total <= _TURN_BUDGET_CHARS:
            break
        original_text = (
            msg.content
            if isinstance(msg.content, str)
            else "\n".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")
        )
        media_count = (
            sum(1 for b in msg.content if isinstance(b, dict) and b.get("type") not in ("text",))
            if isinstance(msg.content, list)
            else 0
        )
        old_len = _msg_text_len(msg.content)
        path = _save_text(original_text, tool_name=msg.name or "tool", results_dir=results_dir)
        media_note = f"\n[{media_count} media block(s) removed to fit context budget]" if media_count else ""
        new_text = (
            original_text[:_MCP_TEXT_THRESHOLD] + f"\n\n[truncated — complete output saved to {path}]{media_note}"
        )
        result[idx] = dataclasses.replace(msg, content=new_text)
        total = total - old_len + len(new_text)

    return result


class MCPError(Exception):
    """Raised when an MCP operation fails."""


class MCPClient:
    """
    Persistent MCP client session using the official ``mcp`` SDK.

    Manages the full lifecycle:
      1. Open transport (stdio subprocess or HTTP stream)
      2. Wrap in ``mcp.ClientSession``
      3. Run ``initialize`` handshake
      4. Expose ``list_tools`` / ``call_tool``
      5. Tear down cleanly on ``disconnect()``

    Uses ``contextlib.AsyncExitStack`` to keep the nested async context
    managers alive for the duration of the connection.
    """

    def __init__(
        self,
        transport: str,
        command: str = "",
        url: str = "",
        env: dict | None = None,
        audio_mode: str = "off",
        transcribe_fn: Any = None,
    ) -> None:
        self.transport = transport
        self.command = command.strip()
        self.url = url.rstrip("/")
        self.env = env  # extra environment variables for stdio subprocess
        self._audio_mode = audio_mode  # "off" | "auto" | "native"
        self._transcribe_fn = transcribe_fn  # async (path: str) -> str | None
        self._session: Any = None  # mcp.ClientSession
        self._stack: Any = None  # contextlib.AsyncExitStack
        self._connected = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Open transport + session and run the MCP initialize handshake."""
        try:
            import contextlib
            from mcp import ClientSession
        except ImportError:
            raise ImportError("MCP support requires: pip install 'harnessx'  (i.e. pip install mcp)")

        stack = contextlib.AsyncExitStack()
        try:
            if self.transport == "stdio":
                read, write = await self._enter_stdio(stack)
            elif self.transport == "http":
                read, write = await self._enter_http(stack)
            else:
                raise MCPError(f"Unknown MCP transport: {self.transport!r} (expected 'stdio' or 'http')")

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            self._stack = stack
            self._session = session
            self._connected = True
        except BaseException:
            # Use BaseException (not Exception) so CancelledError is also caught.
            # CancelledError inherits BaseException directly in Python 3.8+, so
            # except Exception would silently skip aclose(), leaving the anyio
            # cancel scope from streamablehttp_client on the task's scope stack and
            # poisoning all subsequent awaits with repeated CancelledErrors.
            await stack.aclose()
            raise

    async def disconnect(self) -> None:
        """Close the session and transport cleanly."""
        self._connected = False
        self._session = None
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except BaseException as exc:
                logger.debug("MCPClient: disconnect error: %s", exc)
            finally:
                self._stack = None

    async def list_tools(self) -> list[dict]:
        """Return raw tool definitions from the MCP server's tools/list response."""
        self._require_connected()
        result = await self._session.list_tools()
        return [t.model_dump() for t in result.tools]

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        supported_modalities: frozenset[str] = frozenset(),
    ) -> tuple[str, list, bool]:
        """Call an MCP tool.

        Returns ``(text_output, content_blocks, is_error)``.

        * ``text_output`` — plain-text result ready for the model.
        * ``content_blocks`` — list of native Anthropic content-block dicts
          (e.g. ``{"type":"image","source":{...}}``) for inline multimodal
          content when the caller declared matching modality support.
          Empty list when no multimodal blocks were produced.
        * ``is_error`` — True when the MCP server signals a tool-level error.

        Large text results (> ``_MCP_TEXT_THRESHOLD`` chars) are spilled to
        the active ``_mcp_results_ctx`` directory so the model sees a short
        preview + file path instead of a context-busting wall of text.

        Media blocks (ImageContent, EmbeddedResource with blob) are handled
        according to ``supported_modalities``:
        - If the matching modality is declared AND the blob is ≤ 1 MB, the
          block is returned as a native content block.
        - Otherwise the raw bytes are saved to disk and a text reference is
          appended to ``text_output``.
        """
        self._require_connected()
        result = await self._session.call_tool(name, arguments)
        return await self._process_content(result.content or [], name, result.isError, supported_modalities)

    async def _process_content(
        self,
        blocks: list,
        tool_name: str,
        is_error: bool,
        supported_modalities: frozenset[str],
    ) -> tuple[str, list, bool]:
        """Convert raw MCP content blocks into (text, content_blocks, is_error)."""
        text_parts: list[str] = []
        content_blocks: list[dict] = []

        for block in blocks:
            # --- TextContent ---
            if hasattr(block, "text"):
                text_parts.append(block.text)
                continue

            # --- EmbeddedResource with text ---
            resource = getattr(block, "resource", None)
            if resource is not None and hasattr(resource, "text") and resource.text is not None:
                text_parts.append(resource.text)
                continue

            # --- Media: ImageContent or EmbeddedResource blob ---
            media = _extract_media_bytes(block)
            if media is None:
                continue
            raw, mime = media
            modality = _modality_of(mime)

            if modality == "audio" and self._audio_mode != "off":
                await self._handle_audio(raw, mime, tool_name, supported_modalities, text_parts, content_blocks)
            else:
                self._handle_media(raw, mime, tool_name, supported_modalities, text_parts, content_blocks)

        text = "\n".join(text_parts)

        # Spill large text results to disk.
        if len(text) > _MCP_TEXT_THRESHOLD:
            path = _save_text(text, tool_name)
            text = text[:_MCP_TEXT_THRESHOLD] + f"\n\n[truncated — complete output saved to {path}]"

        return text, content_blocks, is_error

    async def _handle_audio(
        self,
        raw: bytes,
        mime: str,
        tool_name: str,
        supported_modalities: frozenset[str],
        text_parts: list[str],
        content_blocks: list[dict],
    ) -> None:
        """Handle an audio blob according to self._audio_mode.

        Modes:
          "auto"   — save to disk, attempt transcription; on success replace with
                     "[Audio transcription]: <text>"; on failure keep file reference.
          "native" — send inline if model declared "audio" support and blob ≤ 1 MB;
                     otherwise save to disk like any other media.
        """
        if self._audio_mode == "auto" and self._transcribe_fn is not None:
            # TODO: file is saved unconditionally so transcribe_fn receives a path; on
            # success the file is unused on disk. Could clean up with os.unlink(path)
            # after a successful transcription once callers no longer need the file.
            path = _save_bytes(raw, mime, tool_name)
            try:
                transcript = await self._transcribe_fn(path)
                if transcript:
                    text_parts.append(f"[Audio transcription]: {transcript}")
                    return
            except Exception as exc:
                logger.debug("Audio transcription failed for %s: %s", path, exc)
            text_parts.append(f"[audio saved to {path}]")
        else:
            # native mode or auto without a transcribe_fn: treat like any other media
            self._handle_media(raw, mime, tool_name, supported_modalities, text_parts, content_blocks)

    @staticmethod
    def _handle_media(
        raw: bytes,
        mime: str,
        tool_name: str,
        supported_modalities: frozenset[str],
        text_parts: list[str],
        content_blocks: list[dict],
    ) -> None:
        """Route a media blob to an inline content block or a saved file."""
        modality = _modality_of(mime)
        inline = modality in supported_modalities and len(raw) <= _MCP_MEDIA_INLINE_LIMIT
        if inline:
            content_blocks.append(
                {
                    "type": modality,
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": base64.b64encode(raw).decode(),
                    },
                }
            )
        else:
            path = _save_bytes(raw, mime, tool_name)
            label = modality if modality != "other" else mime
            text_parts.append(f"[{label} saved to {path}]")

    # ── Transport helpers ─────────────────────────────────────────────────────

    async def _enter_stdio(self, stack: Any) -> tuple:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        parts = shlex.split(self.command)
        if not parts:
            raise MCPError("stdio MCP server requires a non-empty command")
        cmd, *args = parts
        # Merge plugin-declared env vars on top of the current environment so
        # the subprocess inherits PATH etc. but also gets any plugin secrets/keys.
        merged_env: dict | None = None
        if self.env:
            import os

            merged_env = {**os.environ, **self.env}
        params = StdioServerParameters(command=cmd, args=args, env=merged_env)
        read, write = await stack.enter_async_context(stdio_client(params))
        return read, write

    async def _enter_http(self, stack: Any) -> tuple:
        from mcp.client.streamable_http import streamablehttp_client

        if not self.url:
            raise MCPError("http MCP server requires a non-empty url")
        read, write, _get_session_id = await stack.enter_async_context(streamablehttp_client(self.url))
        return read, write

    def _require_connected(self) -> None:
        if not self._connected or self._session is None:
            raise MCPError("MCP client is not connected — call connect() first")


class MCPToolRegistry(_DictRegistryMixin):
    """
    Tool registry backed by a live MCP server connection.

    Each discovered tool gets a ``fn`` that closes over the live ``MCPClient``,
    so the tools can be copied into any ``InMemoryToolRegistry`` and will still
    route execution back through the MCP wire protocol.

    Typical usage in a server startup sequence::

        registry = MCPToolRegistry(transport="stdio", command="...", name="fs")
        await registry.connect()

        # Merge into the agent's main tool registry:
        for tool in registry.tools():
            harness_cfg.tool_registry.register(tool)

        # On shutdown:
        await registry.disconnect()
    """

    def __init__(
        self,
        transport: str = "stdio",
        command: str = "",
        url: str = "",
        name: str = "",
        env: dict | None = None,
        supported_modalities: frozenset[str] | set[str] = frozenset(),
        audio_mode: str = "off",
        transcribe_fn: Any = None,
    ) -> None:
        super().__init__()
        self.transport = transport
        self.command = command
        self.url = url
        self.name = name or command or url
        self.env = env  # extra env vars merged into the subprocess environment
        self.supported_modalities: frozenset[str] = frozenset(supported_modalities)
        self.audio_mode = audio_mode  # "off" | "auto" | "native"
        self.transcribe_fn = transcribe_fn  # async (path: str) -> str | None
        self._client: MCPClient | None = None
        self._last_error: str = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the MCP server and populate the tool registry."""
        client = MCPClient(
            transport=self.transport,
            command=self.command,
            url=self.url,
            env=self.env,
            audio_mode=self.audio_mode,
            transcribe_fn=self.transcribe_fn,
        )
        await client.connect()
        self._client = client
        self._last_error = ""

        raw_tools = await client.list_tools()
        self._tools.clear()
        for tool_def in raw_tools:
            t = self._make_tool(tool_def)
            self._tools[t.name] = t

        logger.info(
            "MCPToolRegistry[%s]: connected — %d tool(s): %s",
            self.name,
            len(self._tools),
            ", ".join(self._tools),
        )

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.connected

    @property
    def last_error(self) -> str:
        return self._last_error

    def set_error(self, msg: str) -> None:
        self._last_error = msg

    def tools(self) -> list[Tool]:
        """Return the list of discovered Tool objects (can be registered elsewhere)."""
        return list(self._tools.values())

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute(self, name: str, input: dict) -> ToolResult:
        if self._client is None or not self._client.connected:
            return ToolResult(output="", error=f"MCP server '{self.name}' is not connected")
        try:
            text, blocks, is_error = await self._client.call_tool(name, input, self.supported_modalities)
            if is_error:
                return ToolResult(output="", error=text)
            return ToolResult(output=text, content_blocks=blocks or None)
        except MCPError as exc:
            logger.debug("MCP tool %r raised MCPError: %s", name, exc, exc_info=True)
            return ToolResult(output="", error=str(exc))
        except Exception as exc:
            logger.debug("MCP tool %r raised unexpected error: %s", name, exc, exc_info=True)
            return ToolResult(output="", error=f"MCP call failed: {exc}")

    # ── Tool construction ─────────────────────────────────────────────────────

    def _make_tool(self, tool_def: dict) -> Tool:
        """
        Wrap an MCP tool definition as an harnessx Tool.

        The ``fn`` closes over ``self`` (not the client directly), so if the
        registry reconnects the closure automatically uses the new client.
        """
        tool_name: str = tool_def["name"]
        registry_ref = self  # weak-ish closure — registry keeps client alive

        async def _mcp_call(**kwargs: Any) -> ToolResult:
            if registry_ref._client is None or not registry_ref._client.connected:
                raise RuntimeError(f"MCP server '{registry_ref.name}' is not connected")
            text, blocks, is_error = await registry_ref._client.call_tool(
                tool_name, kwargs, registry_ref.supported_modalities
            )
            if is_error:
                raise RuntimeError(text)
            return ToolResult(output=text, content_blocks=blocks or None)

        _mcp_call.__name__ = tool_name

        # inputSchema from MCP becomes our JSON Schema; fall back to empty object.
        schema = tool_def.get("inputSchema") or {"type": "object", "properties": {}}

        return Tool(
            name=tool_name,
            description=tool_def.get("description", ""),
            input_schema=schema,
            fn=_mcp_call,
            tags=["mcp"],
        )
