# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.events import Message, ModelResponseEvent, ToolSchema

if TYPE_CHECKING:
    from ..core.harness import HarnessConfig, Harness
    from ..core.trajectory import StatefulTrajectory


@runtime_checkable
class BaseModelProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback: "object | None" = None,
    ) -> ModelResponseEvent: ...

    def count_tokens(self, messages: list[Message]) -> int: ...

    def annotate_trajectory(self, trajectory: "StatefulTrajectory") -> None:
        """Optional hook called by Harness.run() after backfill_rewards().

        Override in RL provider subclasses (e.g. SlimeSGLangProvider) to
        populate TrajectoryStep.token_annotation from captured token data.

        The default implementation is a no-op, so standard providers that
        don't capture token-level data are unaffected.
        """
        ...

    def agentic(self, config: "HarnessConfig") -> "Harness":
        """Combine this provider with a HarnessConfig to produce a runnable Harness.

        Implemented by :class:`~harnessx.providers.agentic.AgenticMixin`.
        """
        ...
