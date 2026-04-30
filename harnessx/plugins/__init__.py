"""HarnessX plugin system.

A plugin bundles processors, tools, and slash commands into a single
installable unit that integrates with the HarnessBuilder pipeline.

Quick start::

    from harnessx.plugins import HarnessPlugin, plugin_registry

    class MyPlugin(HarnessPlugin):
        name = "my-plugin"
        processors = [MyProcessor()]
        slash_commands = {"/mycommand": "_my_slot"}

    plugin_registry.register(MyPlugin())

For manifest-based plugins (``plugin.json``)::

    from harnessx.plugins.loader import load_plugin
    plugin = load_plugin("./path/to/plugin_dir")
    plugin_registry.register(plugin)
"""

from .base import HarnessPlugin
from .registry import PluginRegistry, plugin_registry
from .loader import load_plugin

__all__ = [
    "HarnessPlugin",
    "PluginRegistry",
    "plugin_registry",
    "load_plugin",
]
