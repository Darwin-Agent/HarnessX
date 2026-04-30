# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import Callable

from ...core.events import BeforeModelEvent, ToolSchema
from ...core.processor import MultiHookProcessor


# ---------------------------------------------------------------------------
# Schema transformation helpers
# ---------------------------------------------------------------------------


def _flatten_schema(schema: dict, depth: int = 1) -> dict:
    """Heuristically flatten nested object schemas up to *depth* levels.

    Nested ``object`` properties whose sub-properties all have primitive types
    are inlined into the parent with dot-notation keys
    (e.g. ``options.verbose`` → ``options__verbose``).
    """
    if depth <= 0:
        return schema
    props = schema.get("properties")
    if not props:
        return schema
    flat_props: dict = {}
    flat_required: list = list(schema.get("required", []))
    for key, sub in props.items():
        if sub.get("type") == "object" and "properties" in sub:
            inner = sub["properties"]
            all_primitive = all(
                v.get("type") in ("string", "number", "integer", "boolean", "null") for v in inner.values()
            )
            if all_primitive:
                for inner_key, inner_val in inner.items():
                    flat_key = f"{key}__{inner_key}"
                    flat_props[flat_key] = inner_val
                # Update required: if outer key was required, inner keys become required
                if key in flat_required:
                    flat_required.remove(key)
                    flat_required.extend(f"{key}__{ik}" for ik in sub.get("required", []))
            else:
                flat_props[key] = sub
        else:
            flat_props[key] = sub
    result = {**schema, "properties": flat_props}
    if flat_required:
        result["required"] = flat_required
    else:
        result.pop("required", None)
    return result


def _sort_fields(schema: dict) -> dict:
    """Reorder schema properties: required fields first, then optional."""
    props = schema.get("properties")
    if not props:
        return schema
    required = set(schema.get("required", []))
    ordered = {k: v for k, v in props.items() if k in required}
    ordered.update({k: v for k, v in props.items() if k not in required})
    return {**schema, "properties": ordered}


# ---------------------------------------------------------------------------
# Built-in model profiles
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelProfile:
    """Transformation rules for one model family."""

    flatten_schemas: bool = False  # flatten nested object properties
    sort_fields: bool = False  # required fields before optional
    truncation_reminder: str = ""  # injected into tool description suffix
    custom_transforms: list[Callable[[dict], dict]] = dataclasses.field(default_factory=list)


DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "gpt-5": ModelProfile(
        flatten_schemas=True,
        sort_fields=True,
        truncation_reminder=" (Note: provide complete values — truncation causes errors.)",
    ),
    "gpt-4": ModelProfile(
        sort_fields=True,
    ),
    "claude": ModelProfile(),
}


def _match_profile(model: str, profiles: dict[str, ModelProfile]) -> ModelProfile | None:
    """Return the best-matching profile for *model* by prefix/substring match."""
    lower = model.lower()
    if lower in profiles:
        return profiles[lower]
    candidates = [(k, v) for k, v in profiles.items() if lower.startswith(k) or k in lower]
    if candidates:
        candidates.sort(key=lambda x: len(x[0]), reverse=True)
        return candidates[0][1]
    return None


def _apply_profile(schema: dict, profile: ModelProfile) -> dict:
    result = schema
    if profile.flatten_schemas:
        result = _flatten_schema(result)
    if profile.sort_fields:
        result = _sort_fields(result)
    for transform in profile.custom_transforms:
        result = transform(result)
    return result


def _adapt_tool(tool: ToolSchema, profile: ModelProfile) -> ToolSchema:
    new_schema = _apply_profile(tool.input_schema, profile)
    new_desc = tool.description
    if profile.truncation_reminder and not new_desc.endswith(profile.truncation_reminder):
        new_desc = new_desc + profile.truncation_reminder
    if new_schema is tool.input_schema and new_desc == tool.description:
        return tool
    return dataclasses.replace(tool, input_schema=new_schema, description=new_desc)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class ModelSpecificSchemaAdapter(MultiHookProcessor):
    """Adapt tool schemas to per-model quirks at the ``before_model`` boundary.

    Configured with a target model identifier and a profile table.  The profile
    specifies which schema transformations to apply before the model sees the
    schemas.  The built-in table covers ``gpt-5``, ``gpt-4``, and ``claude``
    families; pass ``profiles`` to extend or override.

    Args:
        model:    Model identifier (e.g. ``"gpt-5.4"``).  Matched against
                  profile keys by prefix/substring.
        profiles: Override or extend the built-in :data:`_DEFAULT_PROFILES` table.
    """

    _singleton_group = "tools.schema_adapter"
    _order = 15  # after correction layer (5), before whitelist (10)

    def __init__(
        self,
        model: str,
        profiles: dict[str, ModelProfile] | None = None,
    ) -> None:
        self._model = model
        self._profile: ModelProfile | None = _match_profile(model, {**DEFAULT_PROFILES, **(profiles or {})})

    async def on_before_model(self, event: BeforeModelEvent):
        if not self._profile:
            yield event
            return
        adapted = []
        changed = False
        for t in event.tools:
            new_t = _adapt_tool(t, self._profile)
            if new_t is not t:
                changed = True
            adapted.append(new_t)
        if changed:
            yield dataclasses.replace(event, tools=tuple(adapted))
        else:
            yield event
