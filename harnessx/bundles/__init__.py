# harnessx/bundles/
#
# Reusable agentic loop designs expressed as HarnessBuilder bundles.
#
# Each bundle defines a processor strategy, leaving
# model_provider / memory / tool_registry slots open for the caller.
# Combine with | to compose, then .build() to get a HarnessConfig.
#
# Which bundle should I use?
# ─────────────────────────────────────────────────────────────────────────────
# Building blocks (compose freely):
#   context       — context assembly: system prompt, history, user wrapper
#   window_mgmt   — window health: token budget, compaction, end nudge,
#                   reasoning budget, tool failure guard
#   reliability   — behavior guardrails: loop detection, parse retry,
#                   tool call correction, repeated call/edit guard
#
# Pre-composed agent bundles:
#   coding        — reliability + env injector + skill loader + window_mgmt
#                   (batteries-included for coding tasks)
#   control       — context + window_mgmt + reliability + coding guards
#                   (full assistant control stack, no skill loader)
#   contrarian    — sycophancy detection + devil's-advocate persona overlay
#
# Optional dimension factories:
#   make_tools()      — tool filtering and whitelisting
#   make_execution()  — sandbox / code execution environment
#
# Typical patterns:
#   Minimal:    context | window_mgmt
#   Assistant:  context | window_mgmt | reliability
#   Coding:     context | coding          # coding already includes window_mgmt + reliability
#   Research:   context | window_mgmt | make_tools(whitelist=[...])
# ─────────────────────────────────────────────────────────────────────────────
#
# Example:
#   from harnessx.bundles import context, window_mgmt, coding
#   config = (
#       HarnessBuilder().slot(model_provider=my_provider)
#       | context | window_mgmt | coding
#   ).build()

from .reliability import reliability, make_reliability
from .coding import coding, make_coding
from .contrarian import contrarian, make_contrarian

# Context: assembly + window management (single module)
from .context import context, make_context, window_mgmt, make_window_mgmt

# Dimension bundles
from .tools import make_tools
from .execution import make_execution
from .control import control, make_control

__all__ = [
    "reliability",
    "make_reliability",
    "coding",
    "make_coding",
    "contrarian",
    "make_contrarian",
    # Context (assembly + window management)
    "context",
    "make_context",
    "window_mgmt",
    "make_window_mgmt",
    # Dimension bundles
    "make_tools",
    "make_execution",
    "control",
    "make_control",
]
