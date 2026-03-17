from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.llm.anthropic_client import ClaudeClient
from app.llm.prompts import ROLE_PARSER_PROMPT_ID, ROLE_PARSER_PROMPT_VERSION, ROLE_PARSER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _safe_json_extract(text: str) -> dict[str, Any]:
    """
    Best-effort 从 LLM 输出中提取 JSON：
    1) 直接解析整段
    2) 处理 ```json ... ``` 或 ``` ... ``` 包起来的内容
    3) 用正则提取第一个大括号对象
    失败时返回空 dict，由上层 fallback。
    """
    if not text:
        return {}

    stripped = text.strip()

    # 1) 直接解析整段
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        logger.debug("Direct JSON parse failed; trying fenced block extraction.")

    # 2) 去掉 markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from fenced block, falling back to brace search.")

    # 3) 匹配第一个 JSON 对象
    brace_match = re.search(r"\{[\s\S]*\}", stripped)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from first brace object, returning empty dict.")

    logger.error("LLM output did not contain parseable JSON; returning empty role_spec.")
    return {}


class JobParserService:
    def __init__(self, llm: ClaudeClient):
        self.llm = llm

    def parse_role(self, raw_input: str) -> dict[str, Any]:
        out = self.llm.generate_text(
            system_prompt=ROLE_PARSER_SYSTEM_PROMPT,
            user_prompt=raw_input,
        )
        role = _safe_json_extract(out)

        # Fallback：即便解析失败，也返回一个最小结构，避免 500
        if not role:
            role = {
                "title": "Unparsed Role",
                "seniority": "Unspecified",
                "sector": "Unspecified",
                "location": "Unspecified",
                "required_skills": [],
                "preferred_skills": [],
                "search_keywords": [],
                "disqualifiers": [],
                "_parse_error": True,
            }

        role.setdefault("search_keywords", [])
        role.setdefault("disqualifiers", [])
        role["_prompt"] = {"id": ROLE_PARSER_PROMPT_ID, "version": ROLE_PARSER_PROMPT_VERSION}
        return role
