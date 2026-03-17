from __future__ import annotations

from typing import Any


def normalize_text_field(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [normalize_text_field(item) for item in value]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        preferred_keys = ("title", "name", "label", "value", "country", "region")
        for key in preferred_keys:
            normalized = normalize_text_field(value.get(key))
            if normalized:
                return normalized
        primary = value.get("primary")
        if primary:
            return normalize_text_field(primary)
    return ""


def normalize_location_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        primary = value.get("primary") or []
        country = normalize_text_field(value.get("country"))
        remote_flexibility = normalize_text_field(value.get("remote_flexibility"))

        parts: list[str] = []
        primary_text = normalize_text_field(primary)
        if primary_text:
            parts.append(primary_text)
        if country:
            parts.append(country)
        if remote_flexibility and remote_flexibility.lower() not in {"not specified", "unknown"}:
            parts.append(remote_flexibility)
        return " / ".join(dict.fromkeys(parts))
    return normalize_text_field(value)


def normalize_search_keywords(role_spec: dict[str, Any]) -> list[str]:
    raw_keywords = role_spec.get("search_keywords") or []
    if isinstance(raw_keywords, list):
        keywords = [normalize_text_field(item) for item in raw_keywords]
        return [keyword for keyword in keywords if keyword]

    fallback = normalize_text_field(raw_keywords)
    return [fallback] if fallback else []


def format_role_spec_for_prompt(role_spec: dict[str, Any]) -> str:
    lines = [
        f"Title: {normalize_text_field(role_spec.get('title')) or 'Unspecified'}",
        f"Seniority: {normalize_text_field(role_spec.get('seniority')) or 'Unspecified'}",
        f"Sector: {normalize_text_field(role_spec.get('sector')) or 'Unspecified'}",
        f"Location: {normalize_location_text(role_spec.get('location')) or 'Unspecified'}",
    ]

    required_skills = ", ".join(normalize_search_keywords({"search_keywords": role_spec.get("required_skills")}))
    preferred_skills = ", ".join(normalize_search_keywords({"search_keywords": role_spec.get("preferred_skills")}))
    search_keywords = ", ".join(normalize_search_keywords(role_spec))

    lines.append(f"Required skills: {required_skills or 'None specified'}")
    lines.append(f"Preferred skills: {preferred_skills or 'None specified'}")
    lines.append(f"Search keywords: {search_keywords or 'None specified'}")
    return "\n".join(lines)
