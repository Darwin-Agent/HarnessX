# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator
from harnessx.home import default_agent_id, default_project


# ── Harness config ────────────────────────────────────────────────────────────


class HarnessConfigPayload(BaseModel):
    """Flat processor-list harness configuration.

    ``processors`` is a list of ``_target_`` dicts describing the processor pipeline.
    Model is intentionally absent — it is always supplied separately via
    ``provider_config`` and combined with the harness via ``model_config.agentic(config)``.
    """

    processors: list[dict[str, Any]] = Field(default_factory=list)
    plugins: list[Any] | None = None
    mcp_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "source": "agent_home",
        }
    )


# ── Slot configuration ────────────────────────────────────────────────────────


class SlotConfig(BaseModel):
    """Runtime slot overrides — sandbox, tool selection, skill filtering."""

    enabled_tools: list[str] | None = None  # None → all built-in tools
    enabled_skills: list[str] | None = None  # None → all skills; [] → none
    sandbox_type: str = "local"  # "local" | "remote"
    sandbox_url: str | None = None  # remote sandbox endpoint


# ── Run ───────────────────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    harness_config: HarnessConfigPayload = Field(default_factory=HarnessConfigPayload)
    task: Any = Field(...)  # str | list[dict] (Anthropic content blocks)
    success_criteria: str = ""
    max_steps: int = 30
    token_budget: int | None = None
    session_id: str | None = None  # multi-turn: resume existing session
    slot_config: SlotConfig = Field(default_factory=SlotConfig)
    provider_config: dict[str, Any] = Field(...)  # full ModelConfig dict (_target_ format)
    agent_id: str = Field(default_factory=default_agent_id)  # AGENT_HOME workspace routing
    project: str = Field(default_factory=default_project)  # AGENT_HOME project routing

    @field_validator("task")
    @classmethod
    def task_not_empty(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            raise ValueError("task must not be empty")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("task must not be empty")
        return v


class RunResponse(BaseModel):
    run_id: str
    session_id: str  # always returned; new uuid if not provided in request


# ── SSE event shapes (serialised as JSON in `data:` field) ────────────────────


class TokenEvent(BaseModel):
    type: str = "token"
    content: str


class StepEndEvent(BaseModel):
    type: str = "step_end"
    step: int
    cost_usd: float


class DoneEvent(BaseModel):
    type: str = "done"
    exit_reason: str
    steps: int
    total_cost: float
    passed: bool | None = None


class ErrorEvent(BaseModel):
    type: str = "error"
    message: str


# ── Example ───────────────────────────────────────────────────────────────────


class ExampleItem(BaseModel):
    """A runnable example loaded from examples/*/harness_config.yaml."""

    key: str
    label: str
    description: str
    harness_config: dict[str, Any]
    workspace: dict[str, str] | None = None


# ── Provider / vendor ─────────────────────────────────────────────────────────


class ProviderItem(BaseModel):
    id: str
    label: str


class ModelItem(BaseModel):
    id: str
    label: str


class VendorInfo(BaseModel):
    id: str
    label: str
    env_key: str
    env_key_set: bool = False  # True when the env var is present on the server
    default_base_url: str | None
    models: list[ModelItem]


# ── Tool info ─────────────────────────────────────────────────────────────────


class ToolInfo(BaseModel):
    name: str
    group: str  # "filesystem" | "web"
    description: str


# ── Skill info ────────────────────────────────────────────────────────────────


class SkillInfo(BaseModel):
    name: str
    description: str
