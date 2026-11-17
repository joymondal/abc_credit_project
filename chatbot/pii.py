"""PII protection: detect, mask, scrub and pseudonymise personal data.

Rules of the design
  * Data minimisation first: the bot never asks for name, Aadhaar, PAN or gender.
  * Anything personal that a user types anyway is scrubbed BEFORE it is logged or sent to an LLM.
  * The full pincode is never stored: the model only needs its first three digits.
  * A contact number (optional, after the decision) is stored only as a keyed HMAC token + mask.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re

# Secret key for pseudonymisation. In production this comes from a secrets manager,
# never from source code; the fallback below exists only so the demo runs out of the box.
def _secret() -> bytes:
    return (os.environ.get("ABC_PII_KEY") or "demo-key-change-me").encode()

# Order matters: longer / more specific patterns first.
PATTERNS = {
    "EMAIL":   re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "AADHAAR": re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
    "PHONE":   re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    "PAN":     re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b"),
    "PINCODE": re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)"),
}
_NAME_HINT = re.compile(r"\b(my name is|i am|i'm|this is|name[:\s]+)\s+([A-Za-z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
                        re.IGNORECASE)
_NOT_NAMES = {"salaried", "self", "a", "an", "the", "student", "farmer", "pensioner", "not", "looking",
              "interested", "buying", "working", "employed", "unemployed", "retired", "sure", "ok",
              "from", "in", "years", "graduate", "here", "applying", "paying", "earning", "getting"}


def token(value: str, kind: str = "pii") -> str:
    """Deterministic keyed pseudonym: same input -> same token; irreversible without the key."""
    digest = hmac.new(_secret(), f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"tok_{digest[:16]}"


def mask_pincode(pin) -> str:
    s = re.sub(r"\D", "", str(pin))
    return s[:3] + "XXX" if len(s) == 6 else "XXXXXX"


def coarsen_pincode(pin) -> int:
    """560037 -> 560000. Keeps the sorting district (all the model uses), drops the post office."""
    return int(str(int(pin))[:3]) * 1000


def mask_phone(phone) -> str:
    d = re.sub(r"\D", "", str(phone))[-10:]
    return d[:2] + "X" * 6 + d[-2:] if len(d) == 10 else "XXXXXXXXXX"


def scrub_text(text) -> str:
    """Replace PII in free text with typed placeholders (safe to log / send to an LLM)."""
    if text is None:
        return text
    out = str(text)
    for kind, pat in PATTERNS.items():
        if kind == "PINCODE":
            out = pat.sub(lambda m: f"<PINCODE:{mask_pincode(m.group())}>", out)
        else:
            out = pat.sub(f"<{kind}>", out)

    def _name(m):
        return m.group(0) if m.group(2).split()[0].lower() in _NOT_NAMES else f"{m.group(1)} <NAME>"
    return _NAME_HINT.sub(_name, out)


def find_pii(text) -> list[str]:
    """Kinds of PII present in a text (used by tests and by the log completeness check)."""
    return [k for k, pat in PATTERNS.items() if pat.search(str(text or ""))]
