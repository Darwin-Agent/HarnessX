from __future__ import annotations

import importlib
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.base_channel import BaseChannel

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type["BaseChannel"]] = {}
_DISCOVERED = False


def register_builtin(cls: type["BaseChannel"]) -> None:
    _REGISTRY[cls.name] = cls


def _load_enabled_channels() -> set[str]:
    """Return the set of platform types that are enabled in gateway.yaml.

    Uses channel_type when present (e.g. instance "discord_hx" with channel_type "discord"
    → adds "discord"), falling back to the YAML key itself for the common case where the
    instance name equals the platform type (e.g. "telegram").
    """
    try:
        from ..core.config import load_channels_config

        cfg = load_channels_config()
        return {
            ch_cfg.get("channel_type", inst_name)
            for inst_name, ch_cfg in cfg.get("channels", {}).items()
            if ch_cfg.get("enabled", False)
        }
    except Exception:
        return set()  # If no config, allow all builtins to auto-discover


def _auto_discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return

    enabled = _load_enabled_channels()
    channels_dir = Path(__file__).parent

    for path in sorted(channels_dir.iterdir()):
        if not path.is_dir() or not (path / "channel.py").exists():
            continue
        name = path.name
        # Strip trailing "_" so "discord_" matches platform type "discord"
        platform_type = name.rstrip("_")
        if enabled and platform_type not in enabled and name not in enabled:
            continue
        try:
            importlib.import_module(f"gateway.channels.{name}.channel")
        except ImportError as e:
            warnings.warn(
                f"Channel '{name}' skipped (optional dependency missing): {e}",
                ImportWarning,
                stacklevel=2,
            )
        except Exception as e:
            logger.error("Failed to load channel '%s': %s", name, e)

    _DISCOVERED = True


def get_registry() -> dict[str, type["BaseChannel"]]:
    _auto_discover()
    return dict(_REGISTRY)


def get_channel_class(name: str) -> type["BaseChannel"] | None:
    _auto_discover()
    return _REGISTRY.get(name)
