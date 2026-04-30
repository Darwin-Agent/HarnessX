from __future__ import annotations

import logging
import random
import time
from threading import Lock

logger = logging.getLogger(__name__)

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 32 chars, excludes 0/O/1/I
_CODE_LEN = 8
_CODE_TTL = 3600  # 1 hour
_RATE_LIMIT_WINDOW = 600  # 10 minutes, 1 attempt
_MAX_FAILURES = 5
_LOCK_DURATION = 3600  # 1 hour after 5 failures
_MAX_PENDING = 3  # at most 3 active codes at once


class PairingAuth:
    """
    Pairing code authorization (inspired by Hermes pairing.py).

    Flow:
      1. Admin generates a code via generate_code()
      2. New user sends /pair <code> to the bot
      3. verify() validates and calls store.add_authorized_user() on success

    Security:
      - 8-char code from 32-char alphabet (excludes ambiguous 0/O/1/I)
      - 1-hour TTL, 10-minute rate limit per sender_id
      - 5 failures → 1-hour lockout for that sender_id
      - Max 3 pending codes at once
    """

    def __init__(self, channel_name: str, session_store) -> None:
        self._channel = channel_name
        self._store = session_store
        self._lock = Lock()
        self._pending: dict[str, float] = {}  # code → expiry_ts
        self._attempts: dict[str, float] = {}  # sender_id → last_attempt_ts
        self._failures: dict[str, int] = {}  # sender_id → failure_count
        self._locked: dict[str, float] = {}  # sender_id → unlock_ts

    def generate_code(self) -> str:
        with self._lock:
            # Prune expired codes
            now = time.time()
            self._pending = {c: exp for c, exp in self._pending.items() if exp > now}
            if len(self._pending) >= _MAX_PENDING:
                raise RuntimeError(f"Too many pending codes ({_MAX_PENDING} max)")
            code = "".join(random.choices(_ALPHABET, k=_CODE_LEN))
            self._pending[code] = now + _CODE_TTL
            logger.info("[%s] Generated pairing code (expires in %ds)", self._channel, _CODE_TTL)
            return code

    def is_authorized(self, sender_id: str) -> bool:
        return self._store.is_authorized(self._channel, sender_id)

    async def verify(self, sender_id: str, code: str) -> tuple[bool, str]:
        """Returns (success, message)."""
        now = time.time()
        with self._lock:
            # Check lockout
            unlock_ts = self._locked.get(sender_id, 0)
            if now < unlock_ts:
                remaining = int(unlock_ts - now)
                return False, f"Account is locked. Please try again in {remaining // 60} minutes."

            # Rate limit
            last = self._attempts.get(sender_id, 0)
            if now - last < _RATE_LIMIT_WINDOW:
                return (
                    False,
                    f"Too many verification attempts. Please retry in {int(_RATE_LIMIT_WINDOW - (now - last))} seconds.",
                )

            self._attempts[sender_id] = now

            # Prune expired
            self._pending = {c: exp for c, exp in self._pending.items() if exp > now}

            if code not in self._pending:
                self._failures[sender_id] = self._failures.get(sender_id, 0) + 1
                if self._failures[sender_id] >= _MAX_FAILURES:
                    self._locked[sender_id] = now + _LOCK_DURATION
                    return False, "Too many invalid codes. Account locked for 1 hour."
                return (
                    False,
                    f"Invalid or expired code (remaining attempts: {_MAX_FAILURES - self._failures[sender_id]}).",
                )

            # Success
            del self._pending[code]
            self._failures.pop(sender_id, None)
            self._locked.pop(sender_id, None)

        self._store.add_authorized_user(self._channel, sender_id)
        logger.info("[%s] sender=%s paired successfully", self._channel, sender_id)
        return True, "Authorization successful. You can use the bot now."

    def revoke(self, sender_id: str) -> None:
        self._store.remove_authorized_user(self._channel, sender_id)
        logger.info("[%s] sender=%s revoked", self._channel, sender_id)
