# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .....core.events import Message
    from .....core.harness import BaseTask


@runtime_checkable
class BaseUserPromptWrapper(Protocol):
    """Wraps user messages before model context assembly (CoT, XML, bench instructions)."""

    async def wrap(self, message: "Message", task: "BaseTask") -> "Message": ...
