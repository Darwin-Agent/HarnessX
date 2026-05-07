# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .....core.events import Message
    from .....core.harness import BaseTask


class ChainOfThoughtWrapper:
    """Appends chain-of-thought guidance to user messages."""

    def __init__(self, guidance: str = "Think step by step before answering."):
        self.guidance = guidance

    async def wrap(self, message: "Message", task: "BaseTask") -> "Message":
        if isinstance(message.content, str):
            new_content = f"{message.content}\n\n{self.guidance}"
        else:
            new_content = [*message.content, {"type": "text", "text": self.guidance}]
        return dataclasses.replace(message, content=new_content)
