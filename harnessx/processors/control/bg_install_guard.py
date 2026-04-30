# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import re

from ...core.events import ToolCallEvent
from ...core.processor import MultiHookProcessor

# Matches installation/build commands that end up backgrounded:
#   apt-get install ... &
#   pip install ... &
#   npm install ... &
#   make ... &  /  cargo build ... &
# Looks for the pattern anywhere in the command (multiline shell scripts too).
#
# The trailing & must be a *standalone* background operator, not:
#   - &&  (logical AND)  — excluded by (?!&) lookahead
#   - >&  or  2>&1       — excluded by (?<![>0-9]) lookbehind
_BG_INSTALL_RE = re.compile(
    r"(?:apt(?:-get)?|pip3?|npm|yarn|pnpm|make|cargo|mvn|gradle|cmake)\b"
    r"[^\n]*(?<![&>0-9])&(?!&)\s*(?:#[^\n]*)?\n?",
)

_WARNING = (
    "[BgInstallGuard] This command has an unpredictable runtime — do not run it in the background. "
    "Either (a) remove the '&' and run it synchronously, or "
    "(b) if you must background it, wrap it with an explicit timeout: `timeout <N> <cmd> &`. "
    "Backgrounding commands without a timeout can leave them running after your session ends, "
    "which will interfere with the verifier."
)


class BgInstallGuard(MultiHookProcessor):
    """Intercept Bash calls that run package installation/build in the background.

    When detected, the tool call is blocked and a corrective message is injected
    as the tool result so the model retries without the ``&``.
    """

    _singleton_group = "bg_install_guard"
    _order = 15  # run early, before tools execute

    async def on_before_tool(self, event: ToolCallEvent):
        if event.tool_name != "Bash":
            yield event
            return

        command = event.tool_input.get("command", "")
        if _BG_INSTALL_RE.search(command):
            yield dataclasses.replace(
                event,
                approved=False,
                synthetic_result=_WARNING,
            )
        else:
            yield event
