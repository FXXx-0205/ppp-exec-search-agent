from __future__ import annotations

import argparse
import uuid

from app.adapters.greenhouse import GreenhouseATSAdapter
from app.repositories.sqlite_repo import SqliteCandidateRepository
from app.services.candidate_sync_service import CandidateSyncService


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync candidates from Greenhouse into local SQLite storage.")
    parser.add_argument("--database-url", default="sqlite:///./data/app.db")
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--base-url", default="https://harvest.greenhouse.io/v1")
    parser.add_argument("--updated-after", default=None)
    parser.add_argument("--created-after", default=None)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    adapter = GreenhouseATSAdapter(
        api_token=args.api_token,
        base_url=args.base_url,
        per_page=min(args.limit, 100),
    )
    repository = SqliteCandidateRepository(args.database_url)
    service = CandidateSyncService(ats_adapter=adapter, candidate_repository=repository)
    synced = service.sync_candidates(
        tenant_id=args.tenant_id,
        provider_filters={
            "updated_after": args.updated_after,
            "created_after": args.created_after,
            "limit": args.limit,
        },
        request_id=f"sync_{uuid.uuid4().hex[:8]}",
    )
    print(f"Synced {len(synced)} candidates for tenant {args.tenant_id}")
