# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .....core.events import Message
    from .....core.harness import BaseTask


class XMLFormatWrapper:
    """Appends structured XML output requirement to user messages."""

    def __init__(self, format_instruction: str = "Respond with structured XML."):
        self.format_instruction = format_instruction

    async def wrap(self, message: "Message", task: "BaseTask") -> "Message":
        return dataclasses.replace(message, content=f"{message.content}\n\n{self.format_instruction}")
