# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import os
import sys

from loguru import logger as _loguru_logger

# ── Default format: colored, human-readable ───────────────────────────────────
_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)

_COMPACT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> <level>[{level: <5}]</level> <cyan>{name}</cyan> — <level>{message}</level>"
)

# ── Stdlib → loguru intercept ─────────────────────────────────────────────────


class _InterceptHandler(logging.Handler):
    """Route all stdlib logging calls through loguru.

    All modules using logging.getLogger(__name__) automatically get loguru's
    formatting, level control, and exception rendering without any code changes.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk the call stack to find the true caller (skip logging internals)
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# ── Initialize ────────────────────────────────────────────────────────────────


def configure_logging(
    level: str | None = None,
    format: str | None = None,
    compact: bool = True,
    sink=None,
) -> None:
    """Configure HarnessX logging and install stdlib→loguru interception.

    After this call every ``logging.getLogger(name).xxx()`` call in any module
    (harnessx core, gateway, providers, …) is routed through loguru, giving
    unified formatting, level control, and full exception tracebacks.

    Args:
        level: Log level (DEBUG/INFO/WARNING/ERROR). Falls back to
               HARNESSX_LOG_LEVEL env var, then INFO.
        format: Custom loguru format string.
        compact: Use compact single-line format (default True). Set False for
                 verbose format with file/function/line info.
        sink: Output sink (default: sys.stderr).
    """
    _loguru_logger.remove()

    effective_level = level or os.environ.get("HARNESSX_LOG_LEVEL", "INFO")
    effective_format = format or (_COMPACT_FORMAT if compact else _DEFAULT_FORMAT)
    effective_sink = sink or sys.stderr

    _loguru_logger.add(
        effective_sink,
        level=effective_level,
        format=effective_format,
        colorize=True,
        backtrace=True,  # full variable trace on exceptions
        diagnose=True,  # show local variables in tracebacks
        enqueue=False,
    )

    # Intercept all stdlib logging (including third-party libs) into loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Suppress noisy third-party SDK logs unless DEBUG is requested.
    # slack_sdk emits high-volume INFO logs for routine reconnects and rate-limit
    # retries that are not actionable; only surface them in verbose mode.
    _noisy_loggers = ("slack_sdk", "slack_bolt", "slack.web", "httpcore", "httpx")
    if effective_level.upper() != "DEBUG":
        for name in _noisy_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)
    else:
        for name in _noisy_loggers:
            logging.getLogger(name).setLevel(logging.NOTSET)


# ── Apply defaults on import ──────────────────────────────────────────────────
configure_logging()

# Re-export the configured logger
logger = _loguru_logger


__all__ = ["logger", "configure_logging"]
