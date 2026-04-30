# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..core.builder import HarnessBuilder
from .reliability import reliability
from .context import make_window_mgmt
from ..processors.control.cost_guard import CostGuardProcessor
from ..processors.control.loop_detection import LoopDetectionProcessor
from ..processors.control.parse_retry import ParseRetryProcessor
from ..processors.control.self_verify import SelfVerifyProcessor
from ..processors.control.repeated_edit_detector import RepeatedFileEditDetector
from ..processors.control.todo_check import TodoCheck, make_todo_tool
from ..processors.control.tool_call_correction import ToolCallCorrectionLayer
from ..processors.control.bg_install_guard import BgInstallGuard


def make_control(
    # Reliability sub-group (bulk toggle)
    include_reliability: bool = True,
    loop_threshold: int = 5,
    loop_warn_threshold: int = 3,
    # Individual processor overrides (None = defer to include_reliability)
    loop_detection: bool | None = None,
    tool_call_correction: bool | None = None,
    parse_retry: bool | None = None,
    self_verify: bool | None = None,
    repeated_edit_detector: bool | None = None,
    # Budget sub-group
    include_budget: bool = False,
    token_threshold: int = 140_000,
    message_threshold: int = 100,
    max_tool_failures: int = 3,
    skill_tool_names: list[str] | None = None,
    # Cost / step guards
    max_cost_usd: float | None = None,
    bg_install_guard: bool = False,
) -> HarnessBuilder:
    """Return a control bundle with the specified sub-groups enabled.

    Args:
        include_reliability:      Add all reliability processors as a group (default True).
                                  Ignored when any individual processor flag is set.
        loop_threshold:           Identical-fingerprint count before loop halt (default 5).
        loop_warn_threshold:      Fingerprint repeat count before a warning is emitted (default 3).
        loop_detection:           Individually enable/disable LoopDetectionProcessor.
        tool_call_correction:     Individually enable/disable ToolCallCorrectionLayer.
        parse_retry:              Individually enable/disable ParseRetryProcessor.
        self_verify:              Individually enable/disable SelfVerifyProcessor.
        repeated_edit_detector:   Individually enable/disable RepeatedFileEditDetector.
        include_budget:           Add context budget processors (default False).
        token_threshold:          Compaction triggers above this token count.
        message_threshold:        Compaction triggers above this message count.
        max_tool_failures:        ToolFailureGuard threshold per turn.
        skill_tool_names:         Tool names passed to CompactionProcessor.
        max_cost_usd:             When set, add ``CostGuardProcessor`` with this limit.
        bg_install_guard:         When True, add ``BgInstallGuard``.
    """
    builder = HarnessBuilder()

    _individual = any(
        f is not None
        for f in [
            loop_detection,
            tool_call_correction,
            parse_retry,
            self_verify,
            repeated_edit_detector,
        ]
    )

    if _individual:
        # Individual mode: each flag defaults to include_reliability if not explicitly set
        _r = include_reliability  # default for unset flags
        _tcc = tool_call_correction if tool_call_correction is not None else _r
        _pr = parse_retry if parse_retry is not None else _r
        _sv = self_verify if self_verify is not None else False
        _ld = loop_detection if loop_detection is not None else _r
        _red = repeated_edit_detector if repeated_edit_detector is not None else _r

        if _tcc:
            builder = builder.add(ToolCallCorrectionLayer())
        if _pr:
            builder = builder.add(ParseRetryProcessor())
        if any([_tcc, _pr, _sv, _ld, _red]):
            builder = builder.add(TodoCheck()).add_tool(make_todo_tool())
        if _sv:
            builder = builder.add(SelfVerifyProcessor())
        if _ld:
            builder = builder.add(
                LoopDetectionProcessor(
                    threshold=loop_threshold,
                    warn_threshold=loop_warn_threshold,
                )
            )
        if _red:
            builder = builder.add(RepeatedFileEditDetector())
    elif include_reliability:
        # Bulk mode: use pre-built singleton when defaults are unchanged
        if loop_threshold == 5 and loop_warn_threshold == 3:
            builder = builder | reliability
        else:
            builder = (
                builder.add(ToolCallCorrectionLayer())
                .add(ParseRetryProcessor())
                .add(TodoCheck())
                .add(SelfVerifyProcessor())
                .add(
                    LoopDetectionProcessor(
                        threshold=loop_threshold,
                        warn_threshold=loop_warn_threshold,
                    )
                )
                .add(RepeatedFileEditDetector())
                .add_tool(make_todo_tool())
            )

    if include_budget:
        builder = builder | make_window_mgmt(
            token_threshold=token_threshold,
            message_threshold=message_threshold,
            max_tool_failures=max_tool_failures,
            skill_tool_names=skill_tool_names,
        )
    if max_cost_usd is not None:
        builder = builder.add(CostGuardProcessor(max_usd=max_cost_usd))
    if bg_install_guard:
        builder = builder.add(BgInstallGuard())
    return builder


control: HarnessBuilder = make_control()
"""Full control bundle with default parameters.

Includes reliability enforcement and context budget management.
Plug into any ``HarnessBuilder`` via ``| control``.
"""
