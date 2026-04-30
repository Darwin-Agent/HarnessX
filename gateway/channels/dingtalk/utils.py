from __future__ import annotations

import base64
import hashlib
import hmac
import time


def verify_dingtalk_signature(headers: dict, body: bytes, secret: str) -> bool:
    """Verify DingTalk outgoing webhook HMAC signature."""
    ts = headers.get("timestamp", "")
    sig = headers.get("sign", "")
    if not ts or not sig:
        return False
    msg = f"{ts}\n{secret}"
    digest = base64.b64encode(hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()
    return hmac.compare_digest(digest, sig)


def is_webhook_expired(expired_time_ms: int | None, safety_margin_ms: int = 300_000) -> bool:
    """Check if a session_webhook has expired (with 5-minute safety margin).

    Returns False when expired_time_ms is 0 or None (unknown expiry — assume valid).
    DingTalk's sessionWebhookExpiredTime is a Unix timestamp in milliseconds.
    """
    if not expired_time_ms:
        return False  # Unknown expiry: try it and let the HTTP call fail if stale
    return time.time() * 1000 + safety_margin_ms >= expired_time_ms
