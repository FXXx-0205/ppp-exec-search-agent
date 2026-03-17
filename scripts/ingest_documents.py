from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.retrieval.vector_store import VectorStore


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    raw = ROOT / "data" / "raw"
    store = VectorStore(persist_dir=settings.chroma_persist_dir)

    docs: list[dict[str, Any]] = []
    firm_path = raw / "sample_firm_profiles" / "firm_profiles.json"
    if firm_path.exists():
        items = _load_json(firm_path)
        for it in items:
            docs.append(
                {
                    "doc_id": it["doc_id"],
                    "text": it["text"],
                    "metadata": it.get("metadata") or {},
                }
            )

    if not docs:
        print("No docs found to ingest. Run scripts/seed_demo_data.py first.")
        return

    store.upsert(docs)
    print(f"Ingested {len(docs)} docs into vector store (mode={store.mode}).")


if __name__ == "__main__":
    main()

