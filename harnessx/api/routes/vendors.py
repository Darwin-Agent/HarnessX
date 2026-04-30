# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

from fastapi import APIRouter

from harnessx.api.models import ModelItem, VendorInfo

router = APIRouter()

_VENDORS: list[VendorInfo] = [
    VendorInfo(
        id="anthropic",
        label="Anthropic",
        env_key="ANTHROPIC_API_KEY",
        default_base_url=None,  # SDK uses its own default
        models=[
            ModelItem(id="claude-opus-4-6", label="Claude Opus 4.6"),
            ModelItem(id="claude-sonnet-4-6", label="Claude Sonnet 4.6"),
            ModelItem(id="claude-haiku-4-5-20251001", label="Claude Haiku 4.5"),
        ],
    ),
    VendorInfo(
        id="openai",
        label="OpenAI",
        env_key="OPENAI_API_KEY",
        default_base_url="https://api.openai.com/v1",
        models=[
            ModelItem(id="gpt-4o", label="GPT-4o"),
            ModelItem(id="gpt-4o-mini", label="GPT-4o mini"),
            ModelItem(id="o3", label="OpenAI o3"),
        ],
    ),
    VendorInfo(
        id="litellm",
        label="LiteLLM (Router)",
        env_key="LITELLM_API_KEY",
        default_base_url=None,
        models=[],
    ),
    VendorInfo(
        id="gemini",
        label="Google Gemini",
        env_key="GEMINI_API_KEY",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
        models=[
            ModelItem(id="gemini/gemini-2.0-flash", label="Gemini 2.0 Flash"),
            ModelItem(id="gemini/gemini-2.5-pro", label="Gemini 2.5 Pro"),
        ],
    ),
    VendorInfo(
        id="deepseek",
        label="DeepSeek",
        env_key="DEEPSEEK_API_KEY",
        default_base_url="https://api.deepseek.com/v1",
        models=[
            ModelItem(id="openai/deepseek-chat", label="DeepSeek Chat"),
            ModelItem(id="openai/deepseek-reasoner", label="DeepSeek Reasoner"),
        ],
    ),
    VendorInfo(
        id="custom",
        label="Custom / Self-hosted",
        env_key="OPENAI_API_KEY",
        default_base_url=None,
        models=[],  # user fills in model manually
    ),
]


@router.get("/vendors", response_model=list[VendorInfo])
async def get_vendors():
    """Return vendor catalogue: env var names, default base URLs, curated model lists."""
    result = []
    for v in _VENDORS:
        result.append(v.model_copy(update={"env_key_set": bool(os.environ.get(v.env_key))}))
    return result
