from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.llm.anthropic_client import ClaudeClient
from app.llm.prompts import (
    BRIEF_GENERATOR_PROMPT_ID,
    BRIEF_GENERATOR_PROMPT_VERSION,
    BRIEF_GENERATOR_SYSTEM_PROMPT,
)
from app.services.role_spec_utils import format_role_spec_for_prompt


class BriefService:
    def __init__(self, llm: ClaudeClient):
        self.llm = llm

    def generate_markdown(
        self,
        role_spec: dict[str, Any],
        ranked_candidates: list[dict[str, Any]],
        retrieval_context: list[dict[str, Any]] | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx_lines: list[str] = []
        if retrieval_context:
            ctx_lines.append("## Retrieval Context (grounding)")
            for c in retrieval_context[:8]:
                title = c.get("title") or c.get("doc_id") or "context"
                ctx_lines.append(f"- {title}: {c.get('text','')[:280]}")

        top = ranked_candidates[:8]
        shortlist_lines = ["## Ranked Shortlist (top)"]
        for r in top:
            shortlist_lines.append(
                f"- {r.get('candidate_id')}: fit={r.get('fit_score')} reasons={'; '.join(r.get('reasons') or r.get('reasoning') or [])}"
            )

        user_prompt = (
            "Role spec:\n"
            f"{format_role_spec_for_prompt(role_spec)}\n\n"
            + "\n".join(ctx_lines)
            + "\n\n"
            + "\n".join(shortlist_lines)
        ).strip()

        md = self.llm.generate_text(
            system_prompt=BRIEF_GENERATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        now = datetime.now(timezone.utc).isoformat()
        return {
            "brief_id": f"brief_{uuid4().hex[:12]}",
            "markdown": md,
            "generated_at": now,
            "citations": [c.get("source") for c in (retrieval_context or []) if c.get("source")],
            "prompt": {"id": BRIEF_GENERATOR_PROMPT_ID, "version": BRIEF_GENERATOR_PROMPT_VERSION},
            "generation_context": generation_context or {},
        }
