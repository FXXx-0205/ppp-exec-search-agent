## Architecture (MVP)

### Components

- **FastAPI**: exposes `intake → candidates → rank → brief` endpoints
- **LLM layer** (`app/llm/`): wraps Anthropic Claude; auto-fallback to a deterministic mock if `ANTHROPIC_API_KEY` is missing
- **Retrieval layer** (`app/retrieval/`): ChromaDB persistent store when available; in-memory fallback otherwise
- **Services layer** (`app/services/`): job parsing, candidate filtering, deterministic scoring, brief generation
- **Streamlit UI** (`app/ui/`): 4 pages to run the demo end-to-end

### Data flow

1. **Role intake**: raw input → `role_spec` (JSON)
2. **RAG retrieval**: query built from `role_spec.search_keywords` → top-k context chunks
3. **Candidate pool**: load demo candidates → filter by keywords/skills
4. **Ranking**: deterministic weighted scoring → reasons + risks
5. **Brief**: generate markdown grounded on retrieved context + shortlist summary

### Auditability (planned)

For regulated environments, the intended audit record includes:

- request_id, timestamps
- prompt version(s), model used
- retrieved doc ids / sources
- human approval status for client-facing exports

