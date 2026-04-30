# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are evaluating whether an AI assistant completed a task during a conversation.

Task description: {task_description}

Last few messages of the conversation (most recent last):
{tail_messages}

Was the task successfully completed? Answer with exactly one word: YES or NO.
- YES: the assistant delivered a clear result or solution addressing the task
- NO: the conversation ended without resolution, the user flagged problems,
  or the assistant was still working on it

Answer:"""

_TAIL_N = 8  # last N messages to include in judgment context


async def judge_task_complete(
    messages: "list",
    task_description: str,
    model: str,
) -> bool:
    """Call *model* to judge if the task in *messages* was completed.

    Returns True if the model says YES, False otherwise.  On any error returns
    False (conservative — better to skip internalization than spam duplicates).
    """
    try:
        import anthropic

        client = anthropic.AsyncAnthropic()

        tail = messages[-_TAIL_N:]
        tail_text = _format_tail(tail)

        prompt = _JUDGE_PROMPT.format(
            task_description=task_description[:500],
            tail_messages=tail_text,
        )

        response = await client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = ""
        for block in response.content:
            if hasattr(block, "text"):
                answer = block.text.strip().upper()
                break

        return answer.startswith("YES")

    except Exception as e:
        _logger.debug("judge_task_complete error: %s", e)
        return False


def _format_tail(messages: list) -> str:
    """Format the tail messages as a readable transcript."""
    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", "?")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # Multimodal — extract text
            text = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        else:
            text = str(content or "")
        tool_calls = getattr(msg, "tool_calls", ())
        if tool_calls:
            tc_names = ", ".join(tc.name for tc in tool_calls)
            text = f"[calls: {tc_names}] {text}"
        text = text[:300]
        lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines)
