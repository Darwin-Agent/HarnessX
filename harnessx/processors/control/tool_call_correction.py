# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable

from ...core.events import ToolCallEvent
from ...core.processor import MultiHookProcessor


# ---------------------------------------------------------------------------
# Built-in correction rules
# Each rule: (param_name, value, param_schema) → corrected_value
# ---------------------------------------------------------------------------


def _coerce_booleans(name: str, value: Any, schema: dict) -> Any:
    """String "true"/"false" → bool when the schema declares type=boolean."""
    if schema.get("type") == "boolean" and isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return value


def _coerce_integers(name: str, value: Any, schema: dict) -> Any:
    """Numeric string → int when the schema declares type=integer."""
    if schema.get("type") == "integer" and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    return value


def _normalize_path(name: str, value: Any, schema: dict) -> Any:
    """Expand ``~`` and resolve relative paths in parameters named *path/file/dir*."""
    if not isinstance(value, str):
        return value
    lower = name.lower()
    if any(k in lower for k in ("path", "file", "dir", "directory")):
        value = os.path.expanduser(value)
        if not os.path.isabs(value):
            value = os.path.abspath(value)
    return value


def _fold_enum_case(name: str, value: Any, schema: dict) -> Any:
    """Case-fold enum values: if schema enums are lowercase and value is not, lowercase it."""
    enums = schema.get("enum")
    if enums and isinstance(value, str) and value not in enums:
        lower = value.lower()
        if lower in enums:
            return lower
    return value


_DEFAULT_RULES: list[Callable[[str, Any, dict], Any]] = [
    _coerce_booleans,
    _coerce_integers,
    _normalize_path,
    _fold_enum_case,
]


class ToolCallCorrectionLayer(MultiHookProcessor):
    """Apply heuristic corrections to tool call parameters before dispatch.

    Each correction rule is a callable ``(param_name, value, param_schema) → value``.
    Rules are applied in order; each receives the value already processed by
    prior rules.

    Built-in rules:
    - boolean coercion  — ``"true"`` / ``"false"`` strings → ``bool``
    - integer coercion  — numeric strings → ``int`` when schema says integer
    - path normalisation — expand ``~``, resolve relative paths for path/file/dir params
    - enum case-folding  — lowercase value when enum values are lowercase

    Args:
        rules:        Full replacement rule list (overrides all defaults).
        extra_rules:  Additional rules appended after the defaults.
        tool_schemas: Optional mapping of tool name → JSON Schema ``properties``
                      dict for schema-aware corrections.  When omitted, all
                      rules receive an empty schema dict.
    """

    _singleton_group = "tool_call_correction"
    _order = 5  # run early so other before_tool processors see corrected values

    def __init__(
        self,
        rules: list[Callable] | None = None,
        extra_rules: list[Callable] | None = None,
        tool_schemas: dict[str, dict] | None = None,
    ) -> None:
        self._rules = list(rules if rules is not None else _DEFAULT_RULES)
        if extra_rules:
            self._rules.extend(extra_rules)
        self._tool_schemas: dict[str, dict] = tool_schemas or {}
        # lowercase → canonical name; seeded from tool_schemas, updated by _bind_tool_registry
        self._name_ci_map: dict[str, str] = {k.lower(): k for k in self._tool_schemas}

    def _bind_tool_registry(self, tool_registry: Any) -> None:
        """Build case-insensitive tool name map from the live registry."""
        if tool_registry is None:
            return
        try:
            names = tool_registry.list_names()
        except Exception:
            return
        self._name_ci_map = {n.lower(): n for n in names}

    def _normalize_tool_name(self, name: str) -> str:
        """Return canonical tool name, correcting case mismatches (e.g. 'bash' → 'Bash')."""
        if not self._name_ci_map or name in self._name_ci_map.values():
            return name
        return self._name_ci_map.get(name.lower(), name)

    def _param_schema(self, tool_name: str, param: str) -> dict:
        tool_def = self._tool_schemas.get(tool_name, {})
        props = tool_def.get("properties") or {}
        return props.get(param, {})

    def _correct(self, tool_name: str, input_dict: dict) -> dict:
        corrected: dict = {}
        changed = False
        for key, val in input_dict.items():
            param_schema = self._param_schema(tool_name, key)
            new_val = val
            for rule in self._rules:
                new_val = rule(key, new_val, param_schema)
            if new_val is not val:
                changed = True
            corrected[key] = new_val
        # Return original dict reference if nothing changed (avoids spurious copies)
        return corrected if changed else input_dict

    async def on_before_tool(self, event: ToolCallEvent):
        canonical_name = self._normalize_tool_name(event.tool_name)
        corrected = self._correct(canonical_name, event.tool_input)
        if canonical_name != event.tool_name or corrected is not event.tool_input:
            yield dataclasses.replace(event, tool_name=canonical_name, tool_input=corrected)
        else:
            yield event
