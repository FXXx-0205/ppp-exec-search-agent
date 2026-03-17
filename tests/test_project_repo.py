from __future__ import annotations

from app.repositories.interfaces import StoredSearchProject
from app.repositories.project_repo import ProjectRepo


def test_project_repo_create_get_list_and_tenant_isolation(tmp_path) -> None:
    repo = ProjectRepo(root_dir=str(tmp_path / "briefs"))
    repo.create_project(
        StoredSearchProject(
            project_id="proj_a",
            tenant_id="tenant_a",
            project_name="Infra Search",
            client_name="Client A",
            role_title="PM",
            status="draft",
            created_by="user_a",
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            metadata={"priority": "high"},
        )
    )
    repo.create_project(
        StoredSearchProject(
            project_id="proj_b",
            tenant_id="tenant_b",
            project_name="Other Search",
            client_name="Client B",
            role_title="Analyst",
            status="draft",
            created_by="user_b",
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            metadata=None,
        )
    )

    loaded = repo.get_project("proj_a")
    tenant_projects = repo.list_projects(tenant_id="tenant_a")

    assert loaded is not None
    assert loaded.project_name == "Infra Search"
    assert [project.project_id for project in tenant_projects] == ["proj_a"]
