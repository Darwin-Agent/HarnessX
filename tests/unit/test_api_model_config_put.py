# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio

from harnessx.api.routes import model_config as mc_route


class TestApiModelConfigPut:
    def test_build_v2_dict_ensures_main_slot(self) -> None:
        req = mc_route.ModelConfigResponse(
            registry=[
                mc_route.ModelDefItem(
                    id="m1",
                    display_name="Main",
                    vendor="anthropic",
                    model_id="claude-sonnet-4-6",
                    api_key="",
                    base_url="",
                    capabilities=["text"],
                )
            ],
            slots=[],
        )

        data = mc_route._build_v2_dict(req)
        assert data["schema_version"] == 2
        # ID remapped from frontend "m1" to readable "claude-sonnet-4-6"
        assert data["roles"]["main"]["default"] == "claude-sonnet-4-6"

    def test_build_v2_dict_maps_reasoning_fields_by_vendor(self) -> None:
        req = mc_route.ModelConfigResponse(
            registry=[
                mc_route.ModelDefItem(
                    id="m_a",
                    display_name="Anthropic",
                    vendor="anthropic",
                    model_id="claude-sonnet-4-6",
                    api_key="",
                    base_url="",
                    capabilities=["text"],
                    extended_thinking=True,
                    thinking_budget_tokens=12000,
                    reasoning_effort="high",
                    reasoning_summary=True,
                ),
                mc_route.ModelDefItem(
                    id="m_o",
                    display_name="OpenAI",
                    vendor="openai",
                    model_id="o4-mini",
                    api_key="",
                    base_url="",
                    capabilities=["text"],
                    reasoning_effort="medium",
                    reasoning_summary=True,
                ),
            ],
            slots=[
                mc_route.ModelSlotItem(slot_name="main", model_ids=["m_a"], strategy="primary"),
                mc_route.ModelSlotItem(slot_name="judge", model_ids=["m_o"], strategy="primary"),
            ],
        )

        data = mc_route._build_v2_dict(req)
        models = {m["id"]: m for m in data["models"]}

        # IDs remapped to model_id values
        anthropic = models["claude-sonnet-4-6"]
        assert anthropic["extended_thinking"] is True
        assert anthropic["thinking_budget_tokens"] == 12000
        assert "reasoning_effort" not in anthropic
        assert "reasoning_summary" not in anthropic

        non_anthropic = models["o4-mini"]
        assert non_anthropic["reasoning_effort"] == "medium"
        assert non_anthropic["reasoning_summary"] is True

    def test_put_model_config_writes_v2_yaml(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))

        req = mc_route.ModelConfigResponse(
            registry=[
                mc_route.ModelDefItem(
                    id="m1",
                    display_name="Main",
                    vendor="anthropic",
                    model_id="claude-sonnet-4-6",
                    api_key="k",
                    base_url="https://api.anthropic.com",
                    capabilities=["text", "code"],
                    extended_thinking=True,
                    thinking_budget_tokens=8192,
                )
            ],
            slots=[
                mc_route.ModelSlotItem(slot_name="main", model_ids=["m1"], strategy="primary"),
            ],
        )

        saved = asyncio.run(mc_route.put_model_config(req))
        saved_path = home / "model_config.yaml"
        assert saved_path.exists()

        text = saved_path.read_text(encoding="utf-8")
        assert "schema_version: 2" in text
        assert "models:" in text
        assert "roles:" in text
        # ID should be the readable model_id, not the frontend random ID
        assert "id: claude-sonnet-4-6" in text

        assert saved.registry[0].id == "claude-sonnet-4-6"
        assert saved.slots[0].slot_name == "main"

    def test_build_v2_dict_maps_targets_and_headers(self) -> None:
        req = mc_route.ModelConfigResponse(
            registry=[
                mc_route.ModelDefItem(
                    id="m_openai",
                    display_name="OpenAI Main",
                    vendor="openai",
                    model_id="gpt-4o",
                    api_key="k-openai",
                    base_url="https://api.openai.com/v1",
                    extra_headers={"X-Trace": "1"},
                    capabilities=["text"],
                ),
                mc_route.ModelDefItem(
                    id="m_litellm",
                    display_name="LiteLLM Alt",
                    vendor="litellm",
                    model_id="gemini/gemini-2.5-pro",
                    api_key="k-lite",
                    base_url="https://router.example/v1",
                    extra_headers={"X-Router": "yes"},
                    capabilities=["text"],
                ),
                mc_route.ModelDefItem(
                    id="m_anth",
                    display_name="Anthropic Judge",
                    vendor="anthropic",
                    model_id="claude-sonnet-4-6",
                    api_key="k-anth",
                    base_url="https://api.anthropic.com",
                    extra_headers={"anthropic-beta": "extended-thinking-2024-09-10"},
                    capabilities=["text"],
                ),
            ],
            slots=[
                mc_route.ModelSlotItem(slot_name="main", model_ids=["m_openai"], strategy="primary"),
                mc_route.ModelSlotItem(slot_name="compact", model_ids=["m_litellm"], strategy="primary"),
                mc_route.ModelSlotItem(slot_name="judge", model_ids=["m_anth"], strategy="primary"),
            ],
        )

        data = mc_route._build_v2_dict(req)
        models = {m["id"]: m for m in data["models"]}

        # IDs are now model_id values
        openai = models["gpt-4o"]
        assert openai["_target_"] == "harnessx.providers.openai_provider.OpenAIProvider"
        assert openai["extra_headers"] == {"X-Trace": "1"}
        assert "default_headers" not in openai

        litellm = models["gemini/gemini-2.5-pro"]
        assert litellm["_target_"] == "harnessx.providers.litellm_provider.LiteLLMProvider"
        assert litellm["extra_headers"] == {"X-Router": "yes"}
        assert "default_headers" not in litellm

        anthropic = models["claude-sonnet-4-6"]
        assert anthropic["_target_"] == "harnessx.providers.anthropic_provider.AnthropicProvider"
        assert anthropic["default_headers"] == {"anthropic-beta": "extended-thinking-2024-09-10"}
        assert "extra_headers" not in anthropic

        parsed = mc_route._parse_v2(data)
        parsed_map = {m.id: m for m in parsed.registry}
        assert parsed_map["gpt-4o"].vendor == "openai"
        assert parsed_map["gpt-4o"].extra_headers == {"X-Trace": "1"}
        assert parsed_map["gemini/gemini-2.5-pro"].vendor == "litellm"
        assert parsed_map["gemini/gemini-2.5-pro"].extra_headers == {"X-Router": "yes"}
        assert parsed_map["claude-sonnet-4-6"].vendor == "anthropic"
        assert parsed_map["claude-sonnet-4-6"].extra_headers == {"anthropic-beta": "extended-thinking-2024-09-10"}
