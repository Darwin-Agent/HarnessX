from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def verify_discord_interaction(headers: dict, body: bytes, public_key: str) -> bool:
    """Verify Discord Ed25519 interaction signature (for slash commands)."""
    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignatureError  # noqa: F401
    except ImportError:
        logger.warning(
            "[discord] PyNaCl not installed — rejecting interaction request (install with: pip install pynacl)"
        )
        return False
    sig = headers.get("x-signature-ed25519", "")
    ts = headers.get("x-signature-timestamp", "")
    if not sig or not ts:
        return False
    try:
        vk = VerifyKey(bytes.fromhex(public_key))
        vk.verify((ts.encode() + body), bytes.fromhex(sig))
        return True
    except Exception:
        return False


async def download_attachment(url: str) -> bytes | None:
    """Download a Discord CDN attachment (no auth required)."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30) as hc:
            resp = await hc.get(url)
            if resp.status_code == 200:
                return resp.content
    except Exception as e:
        logger.debug("[discord] attachment download failed %s: %s", url, e)
    return None
