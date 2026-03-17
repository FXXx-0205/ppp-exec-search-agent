from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(\+?\d[\d\s().-]{7,}\d)\b")


def mask_pii(text: str) -> str:
    if not text:
        return text
    text = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text

