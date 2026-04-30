# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import logging
import re

from harnessx.core.events import ToolResultEvent
from harnessx.core.processor import MultiHookProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redaction patterns
# ---------------------------------------------------------------------------

_REDACTED = "***REDACTED***"

# Matches known high-entropy API key formats inline (value capture group 1)
_KEY_VALUE_PATTERNS: list[re.Pattern[str]] = [
    # Anthropic:  sk-ant-api03-<80+ chars>
    re.compile(r"(sk-ant-[A-Za-z0-9_\-]{20,})", re.ASCII),
    # OpenAI / generic sk- keys
    re.compile(r"(sk-[A-Za-z0-9]{32,})", re.ASCII),
    # YAML / TOML api_key field:  api_key: <value>   or   api_key = <value>
    re.compile(
        r"""((?:api[_\-]key|secret[_\-]key|access[_\-]token)\s*[:=]\s*["']?)([A-Za-z0-9_\-\.]{20,})""",
        re.IGNORECASE,
    ),
    # Environment variable assignment:  ANTHROPIC_API_KEY=sk-...
    re.compile(
        r"""(\b[A-Z][A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)\s*=\s*["']?)([A-Za-z0-9_\-\.]{16,})""",
        re.ASCII,
    ),
]

# Paths whose Read output should always be redacted for secrets
_SENSITIVE_READ_PATHS: list[re.Pattern[str]] = [
    re.compile(r"model_config\.yaml", re.IGNORECASE),
    re.compile(r"\.env\b"),
    re.compile(r"gateway\.yaml", re.IGNORECASE),
    re.compile(r"\.harnessx[/\\]", re.IGNORECASE),
]


def _is_sensitive_read(tool_name: str, tool_input: dict) -> bool:
    if tool_name not in ("Read", "Bash"):
        return False
    path = tool_input.get("file_path", "") or tool_input.get("command", "")
    return any(p.search(path) for p in _SENSITIVE_READ_PATHS)


def _redact(text: str) -> tuple[str, int]:
    """Return (redacted_text, count_of_replacements)."""
    count = 0
    for pattern in _KEY_VALUE_PATTERNS:
        groups = pattern.groups
        if groups >= 2:
            # Pattern has a prefix group + value group — redact only the value
            def _replace_value(m: re.Match[str]) -> str:
                nonlocal count
                count += 1
                return m.group(1) + _REDACTED

            text = pattern.sub(_replace_value, text)
        else:
            # Single capture group — redact the whole match
            def _replace_all(m: re.Match[str]) -> str:
                nonlocal count
                count += 1
                return _REDACTED

            text = pattern.sub(_replace_all, text)
    return text, count


class IMSecretRedactProcessor(MultiHookProcessor):
    """Redact API keys and secrets from tool results before they reach the model.

    Two modes:
    - Always-on regex scan for known key formats (sk-ant-*, sk-*, etc.)
    - Targeted scan when the tool reads a sensitive config path (model_config.yaml,
      .env, gateway.yaml, .harnessx/) — these files commonly contain plaintext keys.
    """

    _singleton_group = "im_secret_redact"
    _order = 5  # after tool execution, before model sees the result

    async def on_after_tool(self, event: ToolResultEvent):
        result = event.result
        if not result:
            yield event
            return

        redacted, count = _redact(result)

        if count > 0:
            logger.warning(
                "[im_secret_redact] redacted %d secret(s) from tool=%r result",
                count,
                event.tool_name,
            )
            yield dataclasses.replace(event, result=redacted)
            return

        yield event
