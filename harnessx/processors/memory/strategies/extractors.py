# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from ....core.events import Message, _extract_text


@runtime_checkable
class ContentExtractor(Protocol):
    """Extract indexable content from a Message for storage/retrieval backends."""

    def extract(self, message: Message) -> Any: ...
    def modality(self) -> str: ...


class TextContentExtractor:
    """Default extractor — pulls plain text from str or multimodal block list."""

    def extract(self, message: Message) -> str:
        return _extract_text(message.content)

    def modality(self) -> str:
        return "text"


def extract_blocks_by_type(content: str | list, block_type: str) -> list[dict]:
    """Return all content blocks of the given type. Returns [] for str content."""
    if isinstance(content, str):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == block_type]


def has_modality(message: Message, modality: str) -> bool:
    """Check whether a Message contains content blocks of a given modality."""
    if isinstance(message.content, str):
        return modality == "text"
    return any(isinstance(b, dict) and b.get("type") == modality for b in message.content)


def message_modalities(message: Message) -> set[str]:
    """Return the set of modality types present in a Message's content blocks."""
    if isinstance(message.content, str):
        return {"text"}
    return {b.get("type") for b in message.content if isinstance(b, dict) and b.get("type")}
