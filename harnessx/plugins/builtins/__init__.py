"""Built-in HarnessX plugins."""

from .session import SessionPlugin
from .command_injection import CommandInjectionProcessor
from .shell_hook import ShellHookProcessor, build_shell_hook_processor

__all__ = [
    "SessionPlugin",
    "CommandInjectionProcessor",
    "ShellHookProcessor",
    "build_shell_hook_processor",
]
