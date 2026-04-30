from __future__ import annotations

import dataclasses
import platform
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from harnessx.core.events import TaskStartEvent
from harnessx.core.processor import MultiHookProcessor

if TYPE_CHECKING:
    pass


class IMSystemProcessor(MultiHookProcessor):
    """
    Inject static IM system prompt at on_task_start.
    Reads im_channel from _im_channel_var ContextVar set by ChannelDispatcher.

    Content (frozen for task lifetime):
    - Persona / role description
    - Platform name and message constraints
    - Response format rules (use platform-native markdown)
    - channel.system_prompt() platform-specific supplement
    """

    _singleton_group = "im_system"
    _order = 5  # run before SystemPromptProcessor (_order=1 already ran; we append)

    async def on_task_start(self, event: TaskStartEvent) -> AsyncIterator[TaskStartEvent]:
        from ..dispatch import _im_channel_var

        channel = _im_channel_var.get(None)
        if channel is None:
            yield event
            return

        workspace_root = ""
        agent_home = ""
        agent_id = ""
        skills_dir = ""
        if event.workspace is not None:
            try:
                workspace_root = str(Path(event.workspace.root).expanduser().resolve())
            except Exception:
                workspace_root = str(getattr(event.workspace, "root", ""))
            try:
                ws_agent_id = getattr(event.workspace, "agent_id", None)
                if ws_agent_id:
                    agent_id = str(ws_agent_id)
            except Exception:
                agent_id = ""
            try:
                ws_home = getattr(event.workspace, "home", None)
                if ws_home:
                    agent_home = str(Path(ws_home).expanduser().resolve())
                elif workspace_root:
                    # Fallback for roots like ".../im-workspaces/{agent_id}"
                    agent_home = str(Path(workspace_root).resolve().parent.parent)
            except Exception:
                agent_home = ""
            if agent_home:
                skills_dir = str(Path(agent_home) / "skills")
        tz_str = ""
        try:
            now = datetime.now().astimezone()
            tz = now.tzinfo
            if hasattr(tz, "key"):
                tz_str = str(getattr(tz, "key"))
            elif now.tzname():
                tz_str = str(now.tzname())
            elif tz is not None:
                tz_str = str(tz)
        except Exception:
            tz_str = ""
        os_str = f"{platform.system()} {platform.release()} ({platform.machine()})".strip()

        platform_specific = channel.system_prompt()
        parts = [
            f"You are a personal AI assistant on {channel.display_name}.",
            f"Platform: {channel.display_name} ({channel.name})",
        ]
        runtime_lines = []
        if agent_id:
            runtime_lines.append(f"- Agent ID: {agent_id}")
        if event.session_id:
            runtime_lines.append(f"- Session ID: {event.session_id}")
        if event.model:
            runtime_lines.append(f"- Model: {event.model}")
        if workspace_root:
            runtime_lines.append(f"- Agent workspace path: {workspace_root}")
        if skills_dir:
            runtime_lines.append(f"- Skills path: {skills_dir}")
        if os_str:
            runtime_lines.append(f"- OS: {os_str}")
        if tz_str:
            runtime_lines.append(f"- Timezone: {tz_str}")
        if runtime_lines:
            parts.append("Runtime context:")
            parts.extend(runtime_lines)
        if platform_specific:
            parts.append(platform_specific)
        parts += [
            "Reply guidelines:",
            "- Language: match the user's language automatically.",
            f"- Formatting: use {channel.display_name} native Markdown; no HTML tags.",
            "- Length: be concise and direct; avoid unnecessary preambles or filler phrases.",
        ]
        im_section = "\n".join(parts)

        existing = event.system_prompt or ""
        combined = f"{existing}\n\n{im_section}".strip() if existing else im_section

        yield dataclasses.replace(event, system_prompt=combined)
