# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from fastapi import APIRouter
from harnessx.api.models import ProviderItem

router = APIRouter()

# Curated list of commonly used models; users can also type a custom string.
_PROVIDERS: list[tuple[str, str]] = [
    # Anthropic (via AnthropicProvider)
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    # OpenAI (via LiteLLMProvider)
    ("openai/gpt-4o", "GPT-4o"),
    ("openai/gpt-4o-mini", "GPT-4o mini"),
    ("openai/o3", "OpenAI o3"),
    # Gemini
    ("gemini/gemini-2.0-flash", "Gemini 2.0 Flash"),
    ("gemini/gemini-2.5-pro", "Gemini 2.5 Pro"),
    # DeepSeek
    ("openai/deepseek-chat", "DeepSeek Chat"),
    ("openai/deepseek-reasoner", "DeepSeek Reasoner"),
]


@router.get("/providers", response_model=list[ProviderItem])
def get_providers():
    """Return curated model provider list for the UI model selector."""
    return [ProviderItem(id=mid, label=label) for mid, label in _PROVIDERS]
