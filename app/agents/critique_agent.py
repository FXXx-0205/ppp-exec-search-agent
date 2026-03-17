from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Critique:
    issues: list[str]
    ok: bool


class CritiqueAgent:
    """
    MVP 版 critique：做最小的“证据不足/PII 风险/格式”检查。
    """

    def critique(self, brief_markdown: str) -> Critique:
        issues: list[str] = []
        if "Evidence is limited" in brief_markdown:
            issues.append("Brief 处于 demo 模式，市场结论需要补充证据或人工复核。")
        if "[EMAIL_REDACTED]" in brief_markdown or "[PHONE_REDACTED]" in brief_markdown:
            issues.append("检测到潜在 PII（已遮罩），请确认是否允许在输出中出现。")
        required_sections = ["Role Summary", "Market Overview", "Candidate Landscape", "Recommended Search Strategy", "Risks / Open Questions"]
        for s in required_sections:
            if s not in brief_markdown:
                issues.append(f"缺少章节：{s}")
        return Critique(issues=issues, ok=len(issues) == 0)

