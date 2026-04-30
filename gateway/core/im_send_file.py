# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import mimetypes
import os
import unicodedata

from harnessx.tools.base import tool

from .dispatch import _im_channel_var, _im_event_var

logger = logging.getLogger(__name__)

# MIME prefixes that map to image sends
_IMAGE_MIME_PREFIX = "image/"
# MIME prefixes that map to audio — sent as file on Feishu (no dedicated audio message type)
_AUDIO_MIME_PREFIX = "audio/"


def _resolve_path(file_path: str) -> str:
    """Expand ~ and normalise Unicode (NFC) so paths from the LLM resolve correctly."""
    return os.path.abspath(os.path.expanduser(unicodedata.normalize("NFC", file_path)))


@tool(
    description=(
        "Send a local file to the current IM chat. "
        "Supports images (JPEG, PNG, GIF, WebP, …) and documents (PDF, Word, Excel, ZIP, …). "
        "Provide an absolute path or a path relative to the workspace root. "
        "Returns 'File sent successfully' on success or an error message."
    )
)
async def im_send_file(file_path: str) -> str:
    path = _resolve_path(file_path)

    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    if not os.path.isfile(path):
        return f"Error: not a regular file: {path}"

    channel = _im_channel_var.get()
    event = _im_event_var.get()
    if channel is None or event is None:
        return "Error: im_send_file can only be called from within an IM session."

    target = channel.make_reply_target(event)
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"

    if mime.startswith(_IMAGE_MIME_PREFIX):
        if not hasattr(channel, "send_photo"):
            return f"Error: channel '{channel.name}' does not support sending images."
        result = await channel.send_photo(target, path)
    else:
        if not hasattr(channel, "send_document"):
            return f"Error: channel '{channel.name}' does not support sending files."
        result = await channel.send_document(target, path)

    if result.success:
        return f"File sent successfully: {os.path.basename(path)}"
    logger.warning("[im_send_file] failed to send '%s' via %s: %s", os.path.basename(path), channel.name, result.error)
    return f"Error sending file: {result.error}"
