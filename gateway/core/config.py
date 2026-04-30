from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_BASE_DIR = Path.home() / ".harnessx"
_DEFAULT_CONFIG_PATH = _BASE_DIR / "gateway.yaml"
_MEDIA_CACHE_DIR = _BASE_DIR / "media_cache"
_DEDUP_DIR = _BASE_DIR / "dedup"
_STORE_DIR = _BASE_DIR / "store"


def load_channels_config(path: Path | None = None) -> dict:
    cfg_path = path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to load channels config from %s: %s", cfg_path, e)
        return {}


def get_media_cache_dir() -> Path:
    _MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _MEDIA_CACHE_DIR


def get_dedup_dir() -> Path:
    """Return path to dedup state directory (~/.harnessx/dedup/)."""
    _DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    return _DEDUP_DIR


def get_store_dir() -> Path:
    """Return path to persistent key-value store directory (~/.harnessx/store/)."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR


def get_channel_config(name: str, full_config: dict | None = None) -> dict:
    cfg = full_config or load_channels_config()
    return cfg.get("channels", {}).get(name, {})
