"""LightMemory — file-based markdown memory dimension plugin.

Quick start::

    from harnessx.plugins.dimensions.light_memory import LightMemoryPlugin

    config = (
        HarnessBuilder()
        .slot(model_provider=provider)
        .plugin(LightMemoryPlugin())        # uses AGENT_HOME/memory/ by default
    ).build()

With custom root and LLM-driven extraction::

    config = (
        HarnessBuilder()
        .slot(model_provider=provider)
        .plugin(LightMemoryPlugin(
            memory_root="/path/to/memory",
            write_mode="llm",
            organization_enabled=True,
        ))
    ).build()
"""

from .plugin import LightMemoryPlugin
from .processors import LightMemoryCaptureProcessor, LightMemoryRetrievalProcessor

__all__ = [
    "LightMemoryPlugin",
    "LightMemoryRetrievalProcessor",
    "LightMemoryCaptureProcessor",
]
