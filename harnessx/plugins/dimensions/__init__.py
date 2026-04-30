"""Dimension plugins — optional behavioral dimensions as HarnessPlugin classes.

Each plugin corresponds to one of the eight harness dimensions and encapsulates
all the processors needed for that dimension.  Use them in place of the legacy
``make_xxx()`` bundle functions when you want lifecycle management (setup/stop).

Quick reference::

    from harnessx.plugins.dimensions import LightMemoryPlugin

    config = (
        HarnessBuilder()
        .slot(model_provider=provider)
        .plugin(LightMemoryPlugin())
    ).build()
"""

from .light_memory import LightMemoryPlugin

__all__ = [
    "LightMemoryPlugin",
]
