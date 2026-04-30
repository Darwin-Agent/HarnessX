# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harnessx.core.events import Message, ToolSchema

try:
    from jinja2 import Template as _JinjaTemplate

    _JINJA_AVAILABLE = True
except ImportError:
    _JINJA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Retool-compatible Jinja2 template (matches the retool SFT training format)
# ---------------------------------------------------------------------------

RETOOL_TOOL_TEMPLATE = """\
<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

# Pre-compiled template — avoids re-parsing the Jinja2 AST on every episode.
_RETOOL_COMPILED_TEMPLATE = _JinjaTemplate(RETOOL_TOOL_TEMPLATE) if _JINJA_AVAILABLE else None


# ---------------------------------------------------------------------------
# Core conversion: HarnessX Messages → retool token IDs
# ---------------------------------------------------------------------------


def _messages_to_retool_ids(
    messages: list["Message"],
    tools: list["ToolSchema"],
    tokenizer: Any,
) -> list[int]:
    """Tokenize using retool's Jinja2 RETOOL_TOOL_TEMPLATE.

    Folds role="tool" messages back into the preceding assistant message content
    as "\\n\\n<interpreter>\\n{result}\\n</interpreter>\\n\\n" — matching the SFT
    checkpoint's training format exactly.
    """
    if not _JINJA_AVAILABLE:
        raise ImportError("jinja2 is required for retool compat: pip install jinja2")

    # Extract system message content
    system_content = next(
        (m.content for m in messages if m.role == "system" and isinstance(m.content, str)),
        "You are a helpful assistant.",
    )

    # Build retool-format message list: user/assistant only, tool results folded in
    retool_msgs: list[dict] = [{"role": "system", "content": system_content}]
    non_sys = [m for m in messages if m.role != "system"]

    i = 0
    while i < len(non_sys):
        m = non_sys[i]
        if m.role == "user":
            retool_msgs.append({"role": "user", "content": m.content or ""})
            i += 1
        elif m.role == "assistant":
            asst_content = m.content or ""
            i += 1
            # Absorb following tool result messages into the assistant turn
            while i < len(non_sys) and non_sys[i].role == "tool":
                tool_result = non_sys[i].content or ""
                asst_content += f"\n\n<interpreter>\n{tool_result}\n</interpreter>\n\n"
                i += 1
            retool_msgs.append({"role": "assistant", "content": asst_content})
        else:
            i += 1

    tool_specs = (
        [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        if tools
        else []
    )

    text = _RETOOL_COMPILED_TEMPLATE.render(messages=retool_msgs, tools=tool_specs)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------------------
# Factory: full formatter (step 0 only)
# ---------------------------------------------------------------------------


def make_retool_formatter(tokenizer: Any):
    """Return a full message_formatter callable bound to the given tokenizer.

    Used for step 0 only — tokenizes the entire conversation from scratch.
    The returned function matches the SGLangProvider.message_formatter
    protocol: (messages, tools) -> list[int].
    """

    def _formatter(
        messages: list["Message"],
        tools: list["ToolSchema"],
    ) -> list[int]:
        return _messages_to_retool_ids(messages, tools, tokenizer)

    return _formatter


# ---------------------------------------------------------------------------
# Factory: incremental inter-turn formatter (step t+1)
# ---------------------------------------------------------------------------

# The retool template closes each assistant turn with <|im_end|> then opens
# the next with <|im_start|>assistant\n.  After step t's model output has been
# captured as output_token_ids, the only text that needs to be tokenized to
# advance to step t+1 is this inter-turn suffix:
#
#   \n\n<interpreter>\n{tool_result}\n</interpreter>\n\n   (one per tool message)
#   <|im_end|>\n<|im_start|>assistant\n                    (closes turn, opens next)
#
# Tokenizing this suffix independently (never decoding+retokenizing the full
# conversation) eliminates BPE boundary mismatches at turn boundaries.
_RETOOL_GEN_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def make_retool_inter_turn_formatter(tokenizer: Any):
    """Return an inter-turn formatter for incremental retool tokenization.

    The returned function matches the SGLangProvider.inter_turn_formatter
    protocol: (new_messages: list[Message]) -> list[int]

    new_messages = messages[prev_msg_count:] — the messages added to the
    conversation since the last complete() call.  Typically one assistant
    message (skipped — already in output_token_ids) + one or more tool
    messages (their contents are formatted as <interpreter>...</interpreter>).

    Returns token IDs for the inter-turn suffix only.  The caller prepends
    known_prefix = prev_input_ids + prev_output_token_ids.
    """

    def _inter_turn(new_messages: list["Message"]) -> list[int]:
        text = ""
        for m in new_messages:
            if m.role == "tool":
                text += f"\n\n<interpreter>\n{m.content or ''}\n</interpreter>\n\n"
        text += _RETOOL_GEN_PROMPT_SUFFIX
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    return _inter_turn
