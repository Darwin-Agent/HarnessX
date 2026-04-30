# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..base import HarnessPlugin
from ...processors.control.rl_signal import RLSignalCollectorProcessor
from ...processors.observability.episode_metrics import EpisodeMetricsProcessor


class RLControlPlugin(HarnessPlugin):
    """RL training control: loop detection via LoopDetectedError + episode metrics.

    Replaces ``make_rl_control()`` from the (now deleted) ``bundles/rl.py``.

    Processors:
        ``RLSignalCollectorProcessor``
            Tracks per-step tool-call fingerprints in a sliding window of 5.
            When the same pattern repeats ≥2 times, raises ``LoopDetectedError``
            on the next ``before_model`` hook → run_loop exits with
            ``exit_reason="loop_detected"``.

        ``EpisodeMetricsProcessor``
            Counts tool successes/errors (``after_tool``), records cumulative
            tokens/cost (``step_end``), and builds ``episode_summary`` dict on
            ``task_end``.

    Note: This plugin does NOT set ``step_snapshots=False``.  Callers that
    want to disable step snapshots (e.g. RL training) must set the slot explicitly::

        builder.slot(step_snapshots=False)
    """

    name = "rl_control"
    version = "0.1.0"
    description = "RL training control: loop detection + episode metrics"

    def __init__(self) -> None:
        super().__init__()
        self.processors = [
            RLSignalCollectorProcessor(),
            EpisodeMetricsProcessor(),
        ]
