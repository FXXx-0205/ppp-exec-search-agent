from __future__ import annotations

from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.middleware import register_middleware
from app.logging_config import configure_logging
from app.api.routes_audit import router as audit_router
from app.api.routes_health import router as health_router
from app.api.routes_search import router as search_router
from app.api.routes_brief import router as brief_router
from app.api.routes_projects import router as projects_router


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="ppp-ai-search-copilot", version="0.1.0")
    register_middleware(app)
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(projects_router, prefix="/projects", tags=["projects"])
    app.include_router(search_router, prefix="/search", tags=["search"])
    app.include_router(brief_router, prefix="/briefs", tags=["briefs"])
    app.include_router(brief_router, prefix="/brief", tags=["brief-legacy"])
    app.include_router(audit_router, prefix="/audit", tags=["audit"])
    return app


app = create_app()
