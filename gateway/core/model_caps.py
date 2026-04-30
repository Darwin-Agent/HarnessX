# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

# Known audio-input-capable model name patterns (case-insensitive).
# Gemini 1.5+ and 2.x all support audio; GPT-4o audio/realtime variants do too.
_AUDIO_PATTERNS = re.compile(
    r"gemini-(?:1\.[5-9]|[2-9]\d*)[.\-]|"
    r"gemini-2[.\-]|"
    r"gpt-4o-audio|"
    r"gpt-4o-realtime|"
    r"gpt-4o-mini-realtime",
    re.IGNORECASE,
)

# Known video-input-capable model name patterns (case-insensitive).
# Gemini models support video natively.
_VIDEO_PATTERNS = re.compile(
    r"gemini-(?:1\.[5-9]|[2-9]\d*)[.\-]|"
    r"gemini-2[.\-]",
    re.IGNORECASE,
)

# Maximum file size to embed inline as base64.
# Larger files are passed as file paths (for tool use).
AUDIO_EMBED_MAX_BYTES = 1 * 1024 * 1024  # 1 MB
VIDEO_EMBED_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

# _capabilities values → input modalities mapping
# asr  = Automatic Speech Recognition (audio input)
# omni = all-modality model (audio + video + image input)
# vl   = Vision-Language (image input only)
_CAP_TO_MODS: dict[str, set[str]] = {
    "asr": {"audio"},
    "omni": {"audio", "video"},
    "vl": set(),  # image only; IMAGE is always handled separately
}

# Simple TTL cache: avoids re-reading YAML on every message
_caps_cache: dict[str, list[str]] = {}  # model_string → capabilities list
_caps_cache_ts: float = 0.0
_CAPS_CACHE_TTL = 30.0  # seconds


def _model_config_paths() -> list[Path]:
    candidates: list[Path] = []
    try:
        from harnessx.home import agent_home

        candidates.append(agent_home() / "model_config.yaml")
    except Exception:
        pass
    candidates.append(Path.home() / ".harnessx" / "model_config.yaml")
    return candidates


def _load_all_caps() -> dict[str, list[str]]:
    """Read model_config.yaml and return {model_string: [capabilities]} for all entries."""
    import yaml

    result: dict[str, list[str]] = {}
    for cfg_path in _model_config_paths():
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for m in data.get("models", []):
                if not isinstance(m, dict):
                    continue
                model_str = m.get("model") or m.get("id", "")
                caps = m.get("_capabilities")
                if model_str and caps:
                    if isinstance(caps, str):
                        caps = [c.strip() for c in caps.split(",")]
                    result[model_str] = list(caps)
                    # Also index by id alias in case provider uses it
                    id_alias = m.get("id", "")
                    if id_alias and id_alias != model_str:
                        result.setdefault(id_alias, list(caps))
        except Exception:
            pass
        break  # Use the first file found
    return result


def _get_capabilities(model_name: str) -> Optional[list[str]]:
    """Return _capabilities list for a model from model_config.yaml, or None if not found."""
    global _caps_cache, _caps_cache_ts

    now = time.monotonic()
    if now - _caps_cache_ts > _CAPS_CACHE_TTL:
        _caps_cache = _load_all_caps()
        _caps_cache_ts = now

    return _caps_cache.get(model_name)


def get_input_modalities(model_name: str) -> frozenset[str]:
    """Return input modalities for a model.

    Priority:
    1. _capabilities field from model_config.yaml (set by the user in the console)
    2. Regex heuristics based on model name (fallback for unconfigured models)

    Always includes "text". Maps capabilities:
      asr  → audio
      omni → audio + video
      vl   → (image; handled separately by IMAGE message type)
    """
    mods: set[str] = {"text"}

    caps = _get_capabilities(model_name)
    if caps is not None:
        for cap in caps:
            extra = _CAP_TO_MODS.get(cap)
            if extra:
                mods.update(extra)
        return frozenset(mods)

    # Fallback: regex heuristics
    if _AUDIO_PATTERNS.search(model_name):
        mods.add("audio")
    if _VIDEO_PATTERNS.search(model_name):
        mods.add("video")
    return frozenset(mods)
