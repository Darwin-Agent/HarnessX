# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseModelProvider
from ..core.events import Message, ModelResponseEvent, ToolSchema

if TYPE_CHECKING:
    pass


class UnConfiguredProvider(BaseModelProvider):
    """
    Placeholder provider for a model slot that has not been wired up yet.

    Raises ``RuntimeError`` on any call with a message indicating which slot
    needs to be configured — instead of silently falling back to a hard-coded
    model string that may not match the user's endpoint.

    Typical use: descriptor defaults that leave the model slot intentionally
    unconfigured so the caller must supply one explicitly::

        UnConfiguredProvider("my_harness.main")

    The caller supplies a real provider via ModelConfig::

        ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"))

    or via ``harnessx lab`` / env vars (ANTHROPIC_API_KEY etc.).
    """

    def __init__(self, slot: str = "model_provider") -> None:
        self.slot = slot

    def _raise(self) -> None:
        raise RuntimeError(
            f"Provider slot '{self.slot}' has not been configured.\n"
            f"Supply a model via ModelConfig before running:\n"
            f'    ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"))\n'
            f"or set ANTHROPIC_API_KEY / OPENAI_API_KEY in the environment."
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback=None,
    ) -> ModelResponseEvent:
        self._raise()

    def count_tokens(self, messages: list[Message]) -> int:
        self._raise()
