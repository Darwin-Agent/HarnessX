from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re

from .constants import FEISHU_MAX_TEXT_LEN


def verify_feishu_signature(headers: dict, body: bytes, verification_token: str) -> bool:
    """Verify X-Lark-Signature HMAC-SHA256."""
    ts = headers.get("x-lark-request-timestamp", "")
    nonce = headers.get("x-lark-request-nonce", "")
    sig = headers.get("x-lark-signature", "")
    if not sig:
        return False
    key = verification_token.encode()
    msg = f"{ts}{nonce}".encode() + body
    expected = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def decrypt_feishu_payload(encrypt: str, encrypt_key: str) -> dict:
    """Decrypt Feishu AES-CBC encrypted event payload.

    Feishu key derivation: SHA256(encrypt_key)[:32]
    Ciphertext format: base64(iv[16] + ciphertext)
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
    except ImportError as e:
        raise ImportError(
            "Feishu webhook encryption requires 'cryptography'. Install with: pip install cryptography"
        ) from e

    aes_key = hashlib.sha256(encrypt_key.encode()).digest()[:32]
    raw = base64.b64decode(encrypt)
    iv, ciphertext = raw[:16], raw[16:]
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    # Remove PKCS7 padding
    pad_len = plaintext[-1]
    plaintext = plaintext[:-pad_len]
    return json.loads(plaintext.decode("utf-8"))


_MENTION_RE = re.compile(r"@_user_\d+|@_all")
_WHITESPACE_RE = re.compile(r"\s+")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def strip_mentions(text: str) -> str:
    """Remove @mention placeholders and normalize whitespace."""
    cleaned = _MENTION_RE.sub(" ", text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n"))
    cleaned = "\n".join(line for line in cleaned.split("\n") if line)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def extract_post_text(rich_text_content: dict) -> str:
    """Extract plain text from feishu post (rich text) message."""
    lines: list[str] = []
    for elements in rich_text_content.get("content", []):
        parts: list[str] = []
        for elem in elements:
            tag = elem.get("tag", "")
            if tag in {"text", "md", "code_block"}:
                parts.append(elem.get("text", ""))
            elif tag == "a":
                parts.append(elem.get("text", "") or elem.get("href", ""))
            elif tag == "at":
                parts.append(f"@{elem.get('user_name') or elem.get('user_id', '')}")
            elif tag == "img":
                parts.append("[image]")
        lines.append("".join(parts))
    return "\n".join(lines)


def truncate(text: str, max_len: int = FEISHU_MAX_TEXT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "…"
