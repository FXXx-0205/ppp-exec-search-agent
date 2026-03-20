from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, cast

from app.config import settings
from app.core.metrics import observe_llm_call

logger = logging.getLogger(__name__)

ToolRunner = Callable[[str, dict[str, Any]], dict[str, Any]]

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
        extra_payload = dict(extra or {})
        tool_definitions = cast(list[dict[str, Any]] | None, extra_payload.pop("tools", None))
        tool_runner = cast(ToolRunner | None, extra_payload.pop("tool_runner", None))
        extra_payload.pop("tool_choice", None)
        extra_payload.pop("max_tool_rounds", None)
        allow_fallback = bool(extra_payload.pop("allow_fallback", True))
        if not self._client:
            if not allow_fallback:
                raise RuntimeError("Claude client is unavailable and fallback is disabled.")
            observe_llm_call(
                round((time.perf_counter() - started_at) * 1000, 2),
                used_fallback=True,
                estimated_tokens=estimated_tokens,
            )
            if tool_definitions:
                raise RuntimeError("Tool-enabled generation must use run_tool_phase_once, not generate_text.")
            return self._mock(system_prompt=system_prompt, user_prompt=user_prompt)

        try:
            resp = self._create_message(
                model=model,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                extra_payload=extra_payload,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            logger.warning("Claude API failed, falling back to mock response: %s", exc)
            observe_llm_call(
                round((time.perf_counter() - started_at) * 1000, 2),
                used_fallback=True,
                estimated_tokens=estimated_tokens,
            )
            return self._mock(system_prompt=system_prompt, user_prompt=user_prompt)
        parts: list[str] = []
        for block in resp.content:
            block_obj = cast(Any, block)
            if getattr(block_obj, "type", None) == "text":
                parts.append(str(getattr(block_obj, "text", "")))
        observe_llm_call(
            round((time.perf_counter() - started_at) * 1000, 2),
            used_fallback=False,
            estimated_tokens=estimated_tokens,
        )
        return "\n".join(parts).strip()

    def run_tool_phase_once(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        user_prompt: str,
        extra_payload: dict[str, Any],
        tool_definitions: list[dict[str, Any]],
        tool_choice: Any,
        tool_runner: ToolRunner,
    ) -> str:
        logger.info("Entering research phase with tool calling enabled.")
        allow_fallback = bool(extra_payload.pop("allow_fallback", True))
        if not self._client:
            if not allow_fallback:
                raise RuntimeError("Claude client is unavailable and fallback is disabled for research phase.")
            logger.info("Research phase running without live Claude client; executing local tool fallback once.")
            tool_result = self._run_mock_tool_round(tool_definitions=tool_definitions, tool_runner=tool_runner, user_prompt=user_prompt)
            return json.dumps(tool_result, ensure_ascii=False)

        initial_messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": initial_messages,
            "tools": tool_definitions,
        }
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if extra_payload:
            kwargs.update(extra_payload)

        first_response = self._create_message_from_kwargs(kwargs)
        tool_uses = self._extract_tool_uses(first_response)
        if not tool_uses:
            logger.info("Research phase completed without tool request.")
            return self._extract_text(first_response)
        if len(tool_uses) != 1:
            raise RuntimeError(f"Claude requested {len(tool_uses)} tools in a single research round; expected exactly 1.")

        logger.info("Research phase tool requested: %s", ", ".join(tool_use["name"] for tool_use in tool_uses))
        initial_messages.append({"role": "assistant", "content": self._serialize_response_content(first_response)})
        tool_results = []
        executed_results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            logger.info("Executing tool: %s", tool_use["name"])
            result = tool_runner(tool_use["name"], tool_use["input"])
            executed_results.append(result)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        initial_messages.append({"role": "user", "content": tool_results})

        follow_up_response = self._create_message(
            model=model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            messages=initial_messages,
            extra_payload=extra_payload,
        )
        if self._extract_tool_uses(follow_up_response):
            raise RuntimeError("Claude requested another tool after the allowed single research-tool round.")
        logger.info("Tool phase completed.")
        return self._finalize_tool_phase_output(
            follow_up_response=follow_up_response,
            executed_results=executed_results,
        )

    def _create_message(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        messages: list[dict[str, Any]],
        extra_payload: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if extra_payload:
            kwargs.update(extra_payload)
        return self._create_message_from_kwargs(kwargs)

    def _create_message_from_kwargs(self, kwargs: dict[str, Any]) -> Any:
        delay_seconds = 1.0
        for attempt in range(3):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt == 2:
                    raise
                logger.warning("Rate limit retry %s/3 after %.1fs", attempt + 1, delay_seconds)
                time.sleep(delay_seconds)
                delay_seconds *= 2
        raise RuntimeError("Claude request failed after rate-limit retries.")

    def _extract_tool_uses(self, resp: Any) -> list[dict[str, Any]]:
        tool_uses: list[dict[str, Any]] = []
        for block in resp.content:
            block_obj = cast(Any, block)
            if getattr(block_obj, "type", None) != "tool_use":
                continue
            tool_uses.append(
                {
                    "id": str(getattr(block_obj, "id", "")),
                    "name": str(getattr(block_obj, "name", "")),
                    "input": cast(dict[str, Any], getattr(block_obj, "input", {}) or {}),
                }
            )
        return tool_uses

    def _serialize_response_content(self, resp: Any) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for block in resp.content:
            block_obj = cast(Any, block)
            block_type = getattr(block_obj, "type", None)
            if block_type == "text":
                content.append({"type": "text", "text": str(getattr(block_obj, "text", ""))})
            elif block_type == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": str(getattr(block_obj, "id", "")),
                        "name": str(getattr(block_obj, "name", "")),
                        "input": cast(dict[str, Any], getattr(block_obj, "input", {}) or {}),
                    }
                )
        return content

    def _run_mock_tool_round(
        self,
        *,
        tool_definitions: list[dict[str, Any]],
        tool_runner: ToolRunner,
        user_prompt: str,
    ) -> dict[str, Any]:
        if not tool_definitions:
            return {}
        tool_name = str(tool_definitions[0].get("name", "")).strip()
        if not tool_name:
            return {}
        try:
            payload = json.loads(user_prompt)
        except json.JSONDecodeError:
            payload = {}
        tool_input = {
            "candidate_identity": payload.get("candidate_identity", {}),
            "research_package": payload.get("research_package", {}),
            "role_spec": payload.get("role_spec", {}),
        }
        return tool_runner(tool_name, tool_input)

    def _extract_text(self, resp: Any) -> str:
        parts: list[str] = []
        for block in resp.content:
            block_obj = cast(Any, block)
            if getattr(block_obj, "type", None) == "text":
                parts.append(str(getattr(block_obj, "text", "")))
        return "\n".join(parts).strip()

    def _finalize_tool_phase_output(
        self,
        *,
        follow_up_response: Any,
        executed_results: list[dict[str, Any]],
    ) -> str:
        text = self._extract_text(follow_up_response)
        if text:
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                if len(executed_results) == 1:
                    logger.warning("Research follow-up returned malformed JSON; using executed tool result directly.")
                    return json.dumps(executed_results[0], ensure_ascii=False)
                raise
        if len(executed_results) == 1:
            logger.warning("Research follow-up returned empty text; using executed tool result directly.")
            return json.dumps(executed_results[0], ensure_ascii=False)
        return text

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


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate_limit" in text or "rate limit" in text
