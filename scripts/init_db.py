from __future__ import annotations

import argparse

from app.repositories.sqlite_repo import SqliteAuditRepository, SqliteBriefRepository, SqliteCandidateRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize SQLite storage for AI Search Copilot.")
    parser.add_argument("--database-url", default="sqlite:///./data/app.db")
    args = parser.parse_args()

    SqliteBriefRepository(args.database_url)
    SqliteAuditRepository(args.database_url)
    SqliteCandidateRepository(args.database_url)
    print(f"Initialized SQLite storage at {args.database_url}")


if __name__ == "__main__":
    main()
