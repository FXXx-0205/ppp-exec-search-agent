from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.core.metrics import observe_llm_call

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str


class ClaudeClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else settings.anthropic_api_key
        self._client = None

        if self.api_key:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1200,
        extra: dict[str, Any] | None = None,
    ) -> str:
        started_at = time.perf_counter()
        estimated_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        if not self._client:
            observe_llm_call(
                round((time.perf_counter() - started_at) * 1000, 2),
                used_fallback=True,
                estimated_tokens=estimated_tokens,
            )
            return self._mock(system_prompt=system_prompt, user_prompt=user_prompt)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if extra:
            kwargs.update(extra)

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            logger.warning("Claude API failed, falling back to mock response: %s", exc)
            observe_llm_call(
                round((time.perf_counter() - started_at) * 1000, 2),
                used_fallback=True,
                estimated_tokens=estimated_tokens,
            )
            return self._mock(system_prompt=system_prompt, user_prompt=user_prompt)
        parts: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        observe_llm_call(
            round((time.perf_counter() - started_at) * 1000, 2),
            used_fallback=False,
            estimated_tokens=estimated_tokens,
        )
        return "\n".join(parts).strip()

    def generate_with_tools(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1200,
        tools: list[dict[str, Any]],
        tool_handler,
        extra: dict[str, Any] | None = None,
        max_rounds: int = 4,
    ) -> str:
        started_at = time.perf_counter()
        estimated_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)

        if not self._client:
            observe_llm_call(
                round((time.perf_counter() - started_at) * 1000, 2),
                used_fallback=True,
                estimated_tokens=estimated_tokens,
            )
            return self._mock(system_prompt=system_prompt, user_prompt=user_prompt)

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "tools": tools,
        }
        if extra:
            kwargs.update(extra)

        try:
            for _ in range(max_rounds):
                resp = self._client.messages.create(messages=messages, **kwargs)
                assistant_content: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []
                text_parts: list[str] = []

                for block in resp.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                        text_parts.append(block.text)
                    elif block_type == "tool_use":
                        tool_input = dict(getattr(block, "input", {}) or {})
                        assistant_content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": tool_input,
                            }
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_handler(block.name, tool_input),
                            }
                        )

                if not tool_results:
                    observe_llm_call(
                        round((time.perf_counter() - started_at) * 1000, 2),
                        used_fallback=False,
                        estimated_tokens=estimated_tokens,
                    )
                    return "\n".join(text_parts).strip()

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})
        except Exception as exc:
            logger.warning("Claude tool API failed, falling back to mock response: %s", exc)
            observe_llm_call(
                round((time.perf_counter() - started_at) * 1000, 2),
                used_fallback=True,
                estimated_tokens=estimated_tokens,
            )
            return self._mock(system_prompt=system_prompt, user_prompt=user_prompt)

        observe_llm_call(
            round((time.perf_counter() - started_at) * 1000, 2),
            used_fallback=True,
            estimated_tokens=estimated_tokens,
        )
        return self._mock(system_prompt=system_prompt, user_prompt=user_prompt)

    def _mock(self, system_prompt: str, user_prompt: str) -> str:
        # 无 key 时用于 demo：只做极简规则解析/生成，保证端到端能跑通
        if "structured role specification" in system_prompt:
            lower = user_prompt.lower()
            if "real assets" in lower:
                title = "Head of Real Assets"
                sector = "Real Assets"
                keywords = ["real assets", "infrastructure", "portfolio", "institutional"]
                required_skills = ["real assets", "portfolio management", "manager selection"]
            elif "infrastructure" in lower:
                title = "Infrastructure Portfolio Manager"
                sector = "Infrastructure"
                keywords = ["infrastructure", "portfolio manager", "real assets", "institutional"]
                required_skills = ["infrastructure", "portfolio management", "institutional"]
            else:
                title = "Executive Role"
                sector = "Funds Management" if "fund" in lower or "asset" in lower else "Unspecified"
                keywords = ["institutional", "portfolio"]
                required_skills = ["Institutional portfolio management"]
            role = {
                "title": title,
                "seniority": "Senior" if "senior" in lower else "Unspecified",
                "sector": sector,
                "location": "Australia" if "australia" in lower else "Unspecified",
                "required_skills": required_skills,
                "preferred_skills": [],
                "search_keywords": keywords,
                "disqualifiers": [],
            }
            return json.dumps(role, ensure_ascii=False)

        if "briefing note" in system_prompt:
            shortlist = []
            for line in user_prompt.splitlines():
                if line.startswith("- cand_") and "fit=" in line:
                    shortlist.append(line[2:])
            top_three = shortlist[:3]
            landscape = "Top candidates show a clear infrastructure and portfolio-management core."
            if not shortlist:
                landscape = "Search returned limited candidate evidence in demo mode."
            return (
                "## 1. Role Summary\n"
                "Search targets a senior infrastructure or real assets leader with institutional portfolio ownership.\n\n"
                "## 2. Market Overview\n"
                "The demo pool includes strong matches, adjacent oversight or strategy profiles, and deliberate noise candidates for ranking separation.\n\n"
                "## 3. Candidate Landscape\n"
                f"Shortlist depth: {len(shortlist)} candidates reviewed.\n"
                + ("\n".join(f"- {line}" for line in top_three) + "\n" if top_three else "")
                + f"{landscape}\n\n"
                "## 4. Recommended Search Strategy\n"
                "- Prioritize directly evidenced infrastructure portfolio managers in Australia\n"
                "- Keep adjacent real assets and governance talent as backup depth\n\n"
                "## 5. Risks / Open Questions\n"
                "- Validate mandate size, mobility, and direct decision-right ownership\n"
            )

        return user_prompt
