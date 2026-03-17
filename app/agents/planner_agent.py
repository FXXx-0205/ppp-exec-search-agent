from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Plan:
    steps: list[str]
    mode: str


class PlannerAgent:
    """
    MVP 版 planner：先用规则决定是否跑 full pipeline。
    后续可替换为 LLM planner + 工具/成本约束。
    """

    def plan(self, role_spec: dict[str, Any]) -> Plan:
        title = (role_spec.get("title") or "").lower()
        if "map" in title:
            return Plan(steps=["retrieve_context", "build_market_map"], mode="market_map_only")
        return Plan(
            steps=["retrieve_context", "collect_candidates", "score_candidates", "generate_brief", "critique_output"],
            mode="full",
        )

