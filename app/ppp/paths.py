from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_path(raw: str, *, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else repo_root / path


def _path_from_env(env_var: str, default: Path, *, repo_root: Path) -> Path:
    raw = os.getenv(env_var)
    if raw and raw.strip():
        return _resolve_path(raw.strip(), repo_root=repo_root)
    return default


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class PPPPaths:
    repo_root: Path
    data_dir: Path
    candidates_csv: Path
    role_spec: Path
    role_spec_ui_override: Path
    research_fixtures: Path
    output_json: Path
    intermediate_dir: Path
    uploaded_candidates_csv: Path
    local_state_dir: Path
    local_state_file: Path

    def display(self, path: Path) -> str:
        return _display_path(path, repo_root=self.repo_root)


def load_ppp_paths() -> PPPPaths:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = _path_from_env("PPP_DATA_DIR", repo_root / "data" / "ppp", repo_root=repo_root)

    role_spec = _path_from_env("PPP_ROLE_SPEC_PATH", data_dir / "role_spec.json", repo_root=repo_root)
    role_spec_ui_override = _path_from_env(
        "PPP_ROLE_SPEC_UI_PATH",
        data_dir / "role_spec_ui.json",
        repo_root=repo_root,
    )
    research_fixtures = _path_from_env(
        "PPP_RESEARCH_FIXTURES_PATH",
        data_dir / "research_fixtures.json",
        repo_root=repo_root,
    )
    output_json = _path_from_env("PPP_OUTPUT_PATH", data_dir / "output.json", repo_root=repo_root)
    intermediate_dir = _path_from_env("PPP_INTERMEDIATE_DIR", data_dir / "intermediate", repo_root=repo_root)
    uploaded_candidates_csv = _path_from_env(
        "PPP_UPLOADED_CSV_PATH",
        data_dir / "uploaded_candidates.csv",
        repo_root=repo_root,
    )
    local_state_dir = _path_from_env("PPP_LOCAL_STATE_DIR", repo_root / ".streamlit", repo_root=repo_root)
    local_state_file = _path_from_env(
        "PPP_LOCAL_STATE_FILE",
        local_state_dir / "ppp_local_state.json",
        repo_root=repo_root,
    )
    candidates_csv = _path_from_env("PPP_CANDIDATES_PATH", data_dir / "candidates.csv", repo_root=repo_root)

    return PPPPaths(
        repo_root=repo_root,
        data_dir=data_dir,
        candidates_csv=candidates_csv,
        role_spec=role_spec,
        role_spec_ui_override=role_spec_ui_override,
        research_fixtures=research_fixtures,
        output_json=output_json,
        intermediate_dir=intermediate_dir,
        uploaded_candidates_csv=uploaded_candidates_csv,
        local_state_dir=local_state_dir,
        local_state_file=local_state_file,
    )


DEFAULT_PATHS = load_ppp_paths()

