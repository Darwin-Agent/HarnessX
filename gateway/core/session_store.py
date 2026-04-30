from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".harnessx" / "gateway_state.json"


class SessionStore:
    """
    Lightweight KV persistence for gateway routing state.
    Distinct from HarnessJournal: Journal stores agent trajectories;
    Store stores gateway-level routing state (epochs, allowlists).

    Format: JSON, atomic write via temp-file rename.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._data: dict = {
            "session_epochs": {},
            "authorized_users": {},
            "channel_states": {},
        }
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = self._path.read_text(encoding="utf-8")
                loaded = json.loads(raw)
                self._data.update(loaded)
            except Exception as e:
                logger.warning("Failed to load gateway state from %s: %s", self._path, e)

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception as e:
            logger.error("Failed to save gateway state to %s: %s", self._path, e)
            tmp.unlink(missing_ok=True)

    # ── Session epoch API ────────────────────────────────────────────────────

    def load_epochs(self) -> dict[str, int]:
        return dict(self._data.get("session_epochs", {}))

    def save_epoch(self, base_session_id: str, epoch: int) -> None:
        with self._lock:
            self._data.setdefault("session_epochs", {})[base_session_id] = epoch
            self._save()

    # ── Authorized users API ─────────────────────────────────────────────────

    def load_authorized_users(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._data.get("authorized_users", {}).items()}

    def add_authorized_user(self, channel_name: str, sender_id: str) -> None:
        with self._lock:
            users = self._data.setdefault("authorized_users", {})
            if sender_id not in users.setdefault(channel_name, []):
                users[channel_name].append(sender_id)
            self._save()

    def remove_authorized_user(self, channel_name: str, sender_id: str) -> None:
        with self._lock:
            users = self._data.get("authorized_users", {}).get(channel_name, [])
            if sender_id in users:
                users.remove(sender_id)
            self._save()

    def is_authorized(self, channel_name: str, sender_id: str) -> bool:
        return sender_id in self._data.get("authorized_users", {}).get(channel_name, [])

    # ── Channel state API ────────────────────────────────────────────────────

    def save_channel_state(self, channel_name: str, state: str) -> None:
        with self._lock:
            self._data.setdefault("channel_states", {})[channel_name] = state
            self._save()

    def load_channel_states(self) -> dict[str, str]:
        return dict(self._data.get("channel_states", {}))
