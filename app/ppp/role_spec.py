from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.llm.anthropic_client import ClaudeClient
from app.llm.prompts import ROLE_PARSER_SYSTEM_PROMPT


def load_role_spec_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Role spec file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Role spec JSON is invalid: {exc}") from exc

    return normalize_role_spec(data)


def load_role_spec_json_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Role spec JSON is invalid: {exc}") from exc
    return normalize_role_spec(data)


def parse_role_spec_text(*, text: str, client: ClaudeClient, model: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Role specification text is empty.")

    raw = client.generate_text(
        system_prompt=ROLE_PARSER_SYSTEM_PROMPT,
        user_prompt=cleaned,
        model=model,
        max_tokens=900,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Role specification parser returned invalid JSON: {exc}") from exc
    return normalize_role_spec(parsed)


def normalize_role_spec(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Role spec JSON must be an object.")

    role = _clean_string(payload.get("role")) or _clean_string(payload.get("title"))
    if not role:
        raise ValueError("Role spec must include a role or title.")

    normalized: dict[str, Any] = {"role": role}

    scalar_aliases = {
        "firm": ("firm", "company", "target_firm"),
        "aum_range": ("aum_range", "aum", "aum_band"),
        "ideal_background": ("ideal_background", "ideal_profile"),
        "seniority": ("seniority",),
        "sector": ("sector",),
        "location": ("location",),
    }
    list_aliases = {
        "focus": ("focus", "channel_focus"),
        "requirements": ("requirements", "required_skills"),
        "preferred_skills": ("preferred_skills",),
        "search_keywords": ("search_keywords",),
        "disqualifiers": ("disqualifiers",),
    }

    for output_key, aliases in scalar_aliases.items():
        scalar_value = _first_clean_scalar(payload, aliases)
        if scalar_value:
            normalized[output_key] = scalar_value

    for output_key, aliases in list_aliases.items():
        list_value = _first_clean_list(payload, aliases)
        if list_value:
            normalized[output_key] = list_value

    # Preserve any additional structured keys so the prompt can still use them.
    for key, value in payload.items():
        if key in normalized or key == "title":
            continue
        if isinstance(value, str):
            cleaned = _clean_string(value)
            if cleaned:
                normalized[key] = cleaned
        elif isinstance(value, list):
            cleaned_list = [_clean_string(item) for item in value if _clean_string(item)]
            if cleaned_list:
                normalized[key] = cleaned_list
        elif isinstance(value, (int, float, bool)):
            normalized[key] = value

    return normalized


def dump_role_spec_json(role_spec: dict[str, Any]) -> str:
    return json.dumps(role_spec, ensure_ascii=False, indent=2)


def _first_clean_scalar(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _clean_string(payload.get(key))
        if value:
            return value
    return None


def _first_clean_list(payload: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            cleaned = [cleaned_item for item in value if (cleaned_item := _clean_string(item))]
            if cleaned:
                return cleaned
    return []


def _clean_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
