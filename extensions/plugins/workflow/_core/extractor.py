# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harnessx.core.events import Message

_logger = logging.getLogger(__name__)

_EXTRACTOR_SYSTEM_PROMPT = """\
You are a workflow extraction assistant.  Your job is to read a conversation
between a user and an AI assistant, and extract reusable workflow procedures
from it.

A "workflow" is a sequence of deterministic shell steps that together accomplish
a well-defined task — something like "deploy app to Kubernetes", "run full test
suite", "scaffold a new service", or "clean up stale branches".  Workflows are
only useful when the same procedure is likely to be repeated.

Instructions
------------
1. Read the conversation below.
2. Identify one or more completed procedures that are worth capturing.
3. For each procedure, write a YAML file to the workflow directory.
4. If a very similar workflow already exists in the directory, UPDATE it instead
   of creating a duplicate — read existing files first.
5. If nothing reusable was found (e.g. the task was one-off research), respond
   with "No reusable workflows identified." and stop.

Each workflow YAML must follow this schema exactly:

```yaml
name: short-kebab-case-name
description: One sentence describing what this workflow does.
tags: [tag1, tag2]
trigger_patterns:
  - "natural language pattern that would trigger this workflow"
params:
  - name: param_name
    description: What this parameter controls.
    default: optional_default_value   # omit if no default
steps:
  - id: step_id
    description: What this step does.
    shell: "shell command; may reference $param_name or $other_step_id"
    condition: "$prev_step.success"   # optional
    approval: true                    # optional, for destructive steps
created: "YYYY-MM-DD"
```

Rules:
- Use real shell commands from the conversation, not pseudocode.
- Parameterize variable parts (app names, namespaces, paths) as $param_name.
- Mark destructive steps (delete, deploy to prod, overwrite) with approval: true.
- Keep trigger_patterns as concrete phrases a user might type.
- The YAML file name must match the workflow name: <name>.yaml
- Save files to: {workflow_dir}/

Write each workflow file using your Write tool, then confirm what you wrote.
"""

_EXTRACTOR_USER_PROMPT = """\
Here is the conversation to analyse. Extract any reusable workflows from it.

<conversation>
{conversation_text}
</conversation>

Workflow directory: {workflow_dir}

List the existing workflow files first (use Read or Bash ls), then extract and write workflows.
"""


async def spawn_extractor(
    messages: "list[Message]",
    workflow_dir: str,
    extractor_model: str,
    on_done: "asyncio.Future[bool] | None" = None,
) -> None:
    """Spawn a sub-harness to extract workflow YAMLs from *messages*.

    Fire-and-forget: launched via ``asyncio.create_task()``.  Resolves
    ``on_done`` (if provided) with True on success, False on any error.
    """
    try:
        result = await _run_extractor(messages, workflow_dir, extractor_model)
        if on_done is not None and not on_done.done():
            on_done.set_result(result)
    except Exception as exc:
        _logger.warning("workflow extractor failed: %s", exc, exc_info=True)
        if on_done is not None and not on_done.done():
            on_done.set_result(False)


async def _run_extractor(
    messages: "list[Message]",
    workflow_dir: str,
    extractor_model: str,
) -> bool:
    """Build and run the extractor sub-harness."""
    try:
        from harnessx.core.builder import HarnessBuilder
        from harnessx.core.state import State  # noqa: F401
        from harnessx.tools.builtin import Bash, Read, Write
    except ImportError as e:
        _logger.warning("workflow extractor: import error: %s", e)
        return False

    # Ensure workflow dir exists
    Path(workflow_dir).mkdir(parents=True, exist_ok=True)

    # Build extractor harness: minimal — just file tools, no processors
    try:
        config = HarnessBuilder(model=extractor_model).add_tool(Read).add_tool(Write).add_tool(Bash).build()
    except Exception as e:
        _logger.warning("workflow extractor: failed to build harness: %s", e)
        return False

    # Format conversation
    conversation_text = _format_conversation(messages)

    system = _EXTRACTOR_SYSTEM_PROMPT.format(workflow_dir=workflow_dir)
    user_prompt = _EXTRACTOR_USER_PROMPT.format(
        conversation_text=conversation_text,
        workflow_dir=workflow_dir,
    )

    try:
        from harnessx.core.harness import Harness

        harness = Harness(config)
        await harness.run(user_prompt, system_prompt=system)
        return True
    except Exception as e:
        _logger.warning("workflow extractor: run failed: %s", e)
        return False


def _format_conversation(messages: "list[Message]") -> str:
    """Convert messages to a readable transcript for the extractor."""
    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", "?")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        parts.append(f"[tool: {block.get('name', '?')}({block.get('input', {})})]")
                    elif block.get("type") == "tool_result":
                        r = block.get("content", "")
                        if isinstance(r, list):
                            r = " ".join(b.get("text", "") for b in r if isinstance(b, dict))
                        r = str(r)[:200]
                        parts.append(f"[result: {r}]")
            content = " ".join(parts)
        else:
            content = str(content or "")

        tool_calls = getattr(msg, "tool_calls", ())
        if tool_calls:
            for tc in tool_calls:
                tc_input = str(getattr(tc, "input", {}))[:150]
                lines.append(f"[TOOL CALL] {tc.name}({tc_input})")

        if content:
            role_label = role.upper()
            # Truncate very long messages
            if len(content) > 1000:
                content = content[:1000] + f"\n…[{len(content) - 1000} chars truncated]"
            lines.append(f"{role_label}: {content}")

    return "\n\n".join(lines)
