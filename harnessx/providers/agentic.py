# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.harness import HarnessConfig, Harness


class AgenticMixin:
    """Mixin that adds ``agentic(harness_config) -> Harness`` to any provider.

    The separation keeps HarnessConfig as a pure behaviour pipeline while
    the model provider is independently configured and composed at the last
    step::

        agent = LLM + harness   →   provider.agentic(config)
    """

    def agentic(self, config: "HarnessConfig") -> "Harness":
        """Shorthand for ``ModelConfig(main=self).agentic(config)``.

        Args:
            config: A fully built HarnessConfig (tools, processors, tracer, …).

        Returns:
            A :class:`~harnessx.core.harness.Harness` ready for ``await harness.run(task)``.
        """
        from ..core.model_config import ModelConfig

        return ModelConfig(main=self).agentic(config)
