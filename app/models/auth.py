from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class UserRole(StrEnum):
    ADMIN = "admin"
    CONSULTANT = "consultant"
    RESEARCHER = "researcher"
    COMPLIANCE = "compliance"


class ApprovalStatus(StrEnum):
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class UserIdentity(BaseModel):
    user_id: str = Field(min_length=1)
    email: str
    display_name: str
    role: UserRole


class ApprovalDecision(BaseModel):
    brief_id: str
    status: ApprovalStatus
    decided_by: UserIdentity
    decided_at: datetime
    comment: str | None = None


class AccessContext(BaseModel):
    tenant_id: str
    project_id: str | None = None
    actor: UserIdentity


ROLE_PERMISSIONS: dict[UserRole, set[str]] = {
    UserRole.ADMIN: {
        "project:create",
        "project:view",
        "search:run",
        "brief:generate",
        "brief:submit",
        "brief:approve",
        "brief:export",
        "audit:view",
    },
    UserRole.CONSULTANT: {
        "project:create",
        "project:view",
        "search:run",
        "brief:generate",
        "brief:submit",
        "brief:approve",
        "brief:export",
        "audit:view",
    },
    UserRole.RESEARCHER: {"project:create", "project:view", "search:run", "brief:generate", "brief:submit"},
    UserRole.COMPLIANCE: {"project:view", "search:run", "brief:approve", "brief:export", "audit:view"},
}


MANAGER_ROLES: set[UserRole] = {UserRole.ADMIN, UserRole.CONSULTANT}
