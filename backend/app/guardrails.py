from __future__ import annotations

_BAD = [
    "build a bomb",
    "self-harm",
    "explosive",
    "make a weapon",
    "harm others",
    "ddos",
    "malware",
]

def is_safe(text: str) -> bool:
    t = (text or "").lower()
    return not any(b in t for b in _BAD)
