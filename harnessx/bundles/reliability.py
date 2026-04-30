# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from ..core.builder import HarnessBuilder
from ..processors.control.loop_detection import LoopDetectionProcessor
from ..processors.control.parse_retry import ParseRetryProcessor
from ..processors.control.self_verify import SelfVerifyProcessor
from ..processors.control.repeated_edit_detector import RepeatedFileEditDetector
from ..processors.control.todo_check import TodoCheck, make_todo_tool
from ..processors.control.tool_call_correction import ToolCallCorrectionLayer


def make_reliability(*, self_verify: bool = True) -> HarnessBuilder:
    """Build the reliability bundle with optional self-verification."""
    b = (
        HarnessBuilder()
        .add(ToolCallCorrectionLayer())
        .add(ParseRetryProcessor())
        .add(TodoCheck())
        .add(LoopDetectionProcessor())
        .add(RepeatedFileEditDetector())
        .add_tool(make_todo_tool())
    )
    if self_verify:
        b = b.add(SelfVerifyProcessor())
    return b


reliability: HarnessBuilder = make_reliability(self_verify=True)
"""Reliability harness bundle.

Addresses the six most common agentic failure modes identified by ForgeCode
Services.  Plug into any ``HarnessBuilder`` via ``| reliability``.
"""
