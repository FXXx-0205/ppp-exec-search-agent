from app.ppp.paths import DEFAULT_PATHS


def test_streamlit_entrypoint_exists_at_repo_root() -> None:
    entrypoint = DEFAULT_PATHS.repo_root / "streamlit_app.py"
    assert entrypoint.exists()


def test_streamlit_defaults_point_to_ppp_submission_paths() -> None:
    assert DEFAULT_PATHS.output_json == DEFAULT_PATHS.data_dir / "output.json"
    assert DEFAULT_PATHS.intermediate_dir == DEFAULT_PATHS.data_dir / "intermediate"
    assert DEFAULT_PATHS.uploaded_candidates_csv == DEFAULT_PATHS.data_dir / "uploaded_candidates.csv"


def test_readme_documents_root_streamlit_command() -> None:
    readme_text = (DEFAULT_PATHS.repo_root / "README.md").read_text(encoding="utf-8")
    assert "streamlit run streamlit_app.py" in readme_text
