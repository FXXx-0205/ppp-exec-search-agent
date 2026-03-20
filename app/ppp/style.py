from __future__ import annotations

import re
from typing import Sequence


def stable_variant_index(*parts: str, modulo: int) -> int:
    joined = "|".join(part for part in parts if part)
    return sum(ord(char) for char in joined) % modulo if joined else 0


def choose_variant(variants: Sequence[str], *parts: str) -> str:
    if not variants:
        return ""
    return variants[stable_variant_index(*parts, modulo=len(variants))]


def polish_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = re.sub(r"\ba ([aeiouAEIOU])", r"an \1", cleaned)
    cleaned = re.sub(r"\ban ([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])", r"a \1", cleaned)
    cleaned = cleaned.replace(" ;", ";")
    cleaned = cleaned.replace(" ,", ",")
    cleaned = cleaned.replace(" .", ".")
    cleaned = cleaned.replace("  ", " ")
    cleaned = cleaned.replace("a institutional", "an institutional")
    cleaned = cleaned.replace("a ANZ", "an ANZ")
    cleaned = cleaned.replace("a adjacent", "an adjacent")
    cleaned = cleaned.replace("an useful", "a useful")
    cleaned = cleaned.replace("and and", "and")
    return cleaned.strip()


def polish_join(*sentences: str) -> str:
    return polish_text(" ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip()))
