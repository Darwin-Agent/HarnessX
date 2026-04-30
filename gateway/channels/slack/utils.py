from __future__ import annotations

import hashlib
import hmac
import logging
import time

logger = logging.getLogger(__name__)


def verify_slack_signature(headers: dict, body: bytes, signing_secret: str) -> bool:
    """Verify X-Slack-Signature v0=HMAC-SHA256."""
    ts = headers.get("x-slack-request-timestamp", "")
    sig = headers.get("x-slack-signature", "")
    if not ts or not sig:
        return False
    # Replay attack window: 5 minutes
    if abs(time.time() - float(ts)) > 300:
        return False
    base = f"v0:{ts}:{body.decode('utf-8', errors='replace')}"
    digest = "v0=" + hmac.new(signing_secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, sig)


async def download_slack_file(url: str, bot_token: str) -> bytes | None:
    """Download a Slack private file with Bearer token auth."""
    try:
        import httpx

        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as hc:
            resp = await hc.get(url, headers={"Authorization": f"Bearer {bot_token}"})
            if resp.status_code != 200:
                return None
            ct = resp.headers.get("content-type", "")
            if "text/html" in ct:
                # Slack returns login page when token is invalid
                return None
            return resp.content
    except Exception as e:
        logger.debug("[slack] file download failed %s: %s", url, e)
        return None
