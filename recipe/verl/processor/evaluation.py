# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Evaluation processor for verl backend."""
from harnessx.processors.evaluation.evaluation import EvaluationProcessor


class VerlEvaluationProcessor(EvaluationProcessor):
    __hx_runtime_only__ = True
