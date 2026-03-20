from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.ppp.paths import DEFAULT_PATHS


@dataclass(frozen=True)
class LocalAPIKeyState:
    anthropic_api_key: str = ""
    tavily_api_key: str = ""


def _normalize_key(value: str | None) -> str:
    return value.strip() if value else ""


def load_local_api_state(path: Path | None = None) -> LocalAPIKeyState:
    target = path or DEFAULT_PATHS.local_state_file
    if not target.exists():
        return LocalAPIKeyState()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return LocalAPIKeyState()
    if not isinstance(payload, dict):
        return LocalAPIKeyState()
    return LocalAPIKeyState(
        anthropic_api_key=_normalize_key(str(payload.get("anthropic_api_key", ""))),
        tavily_api_key=_normalize_key(str(payload.get("tavily_api_key", ""))),
    )


def save_local_api_state(state: LocalAPIKeyState, path: Path | None = None) -> Path:
    target = path or DEFAULT_PATHS.local_state_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def clear_local_api_state(path: Path | None = None) -> None:
    target = path or DEFAULT_PATHS.local_state_file
    if target.exists():
        target.unlink()
