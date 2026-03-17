from __future__ import annotations

from typing import Any

from app.config import settings
from app.models.workflow import MatchType
from app.services.role_spec_utils import normalize_location_text, normalize_text_field


class RankingService:
    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        strategy_version: str | None = None,
    ):
        self.weights = weights or settings.ranking_weights()
        self.strategy_version = strategy_version or settings.ranking_strategy_version

    def score_candidates(self, role_spec: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        required = [s.lower() for s in (role_spec.get("required_skills") or []) if isinstance(s, str)]
        preferred = [s.lower() for s in (role_spec.get("preferred_skills") or []) if isinstance(s, str)]
        search_keywords = [s.lower() for s in (role_spec.get("search_keywords") or []) if isinstance(s, str)]
        location = self._normalize_location_target(role_spec.get("location"))
        seniority_target = normalize_text_field(role_spec.get("seniority")).lower()
        sector_target = normalize_text_field(role_spec.get("sector")).lower()
        title_target = normalize_text_field(role_spec.get("title")).lower()

        def has_any(blob: str, items: list[str]) -> int:
            return sum(1 for it in items if it and it in blob)

        ranked: list[dict[str, Any]] = []
        for c in candidates:
            summary = c.get("summary") or ""
            sectors = c.get("sectors") or []
            functions = c.get("functions") or []
            skills = c.get("skills") or []
            employment = c.get("employment_history") or []
            full_blob = " ".join(
                [
                    str(c.get("full_name") or "").lower(),
                    str(c.get("current_title") or "").lower(),
                    str(c.get("current_company") or "").lower(),
                    summary.lower(),
                    " ".join(sectors).lower(),
                    " ".join(functions).lower(),
                    " ".join(skills).lower(),
                    " ".join(str(item) for item in employment).lower(),
                ]
            )
            sector_blob = " ".join(sectors).lower()
            required_hit = has_any(full_blob, required)
            preferred_hit = has_any(full_blob, preferred)
            keyword_hit = has_any(full_blob, search_keywords)
            loc_hit = 1 if self._location_matches(location, c.get("location")) else 0
            title_blob = " ".join(
                value.lower()
                for value in [c.get("current_title") or "", c.get("headline") or "", summary]
                if isinstance(value, str)
            )
            years_experience = c.get("years_experience")

            skill_denominator = len(required) + len(preferred) + max(1, min(3, len(search_keywords)))
            skill_points = (required_hit * 2.0) + preferred_hit + min(keyword_hit, 3)
            skill_match = min(1.0, skill_points / max(1.0, skill_denominator))
            sector_relevance = self._sector_relevance(sector_target, sector_blob, summary.lower(), title_blob)
            location_alignment = 0.95 if loc_hit else (0.6 if not location else 0.25)
            seniority_match = self._seniority_match(seniority_target, title_blob, years_experience)
            functional_similarity = self._functional_similarity(title_target, search_keywords, functions, title_blob, full_blob)
            stability_signal = self._stability_signal(years_experience, employment)

            final = (
                self.weights["skill_match"] * skill_match
                + self.weights["seniority_match"] * seniority_match
                + self.weights["sector_relevance"] * sector_relevance
                + self.weights["functional_similarity"] * functional_similarity
                + self.weights["location_alignment"] * location_alignment
                + self.weights["stability_signal"] * stability_signal
            )

            current_title = c.get("current_title")
            location_value = c.get("location")
            dimension_scores = {
                "skill_match": self._dimension(
                    score=skill_match,
                    match_type=MatchType.EXPLICIT if required_hit else (MatchType.INFERRED if summary and required else MatchType.UNKNOWN),
                    reasons=(
                        ["Direct evidence of required skill signals in structured profile content."]
                        if required_hit
                        else (["Skill alignment inferred from summary context."] if summary and required else ["Required skill evidence is missing."])
                    ),
                    evidence=(
                        [{"field": "summary", "source": "candidate.summary", "value": summary[:280]}]
                        if summary
                        else []
                    ),
                    missing_information=([] if summary else ["summary"]),
                ),
                "seniority_match": self._dimension(
                    score=seniority_match,
                    match_type=MatchType.EXPLICIT if current_title and seniority_target and seniority_target in title_blob else (MatchType.INFERRED if current_title else MatchType.UNKNOWN),
                    reasons=(
                        ["Current title suggests aligned seniority."]
                        if current_title and seniority_target and seniority_target in title_blob
                        else (["Seniority is partially inferred from title context."] if current_title else ["Current title is missing."])
                    ),
                    evidence=(
                        [{"field": "experience", "source": "candidate.current_title", "value": current_title}]
                        if current_title
                        else []
                    ),
                    missing_information=([] if current_title else ["current_title"]),
                ),
                "sector_relevance": self._dimension(
                    score=sector_relevance,
                    match_type=MatchType.EXPLICIT if sector_target and sector_target in " ".join(sectors).lower() else (MatchType.INFERRED if sectors or summary else MatchType.UNKNOWN),
                    reasons=(
                        ["Sector history directly aligns with the role sector."]
                        if sector_target and sector_target in " ".join(sectors).lower()
                        else (["Sector relevance is inferred from profile context."] if sectors or summary else ["Sector data is missing."])
                    ),
                    evidence=(
                        [{"field": "sector", "source": "candidate.sectors[0]", "value": sectors[0]}]
                        if sectors
                        else []
                    ),
                    missing_information=([] if sectors else ["sector"]),
                ),
                "functional_similarity": self._dimension(
                    score=functional_similarity,
                    match_type=MatchType.INFERRED if summary else MatchType.UNKNOWN,
                    reasons=["Functional alignment is inferred from summary context."] if summary else ["Functional scope is missing."],
                    evidence=([{"field": "summary", "source": "candidate.summary", "value": summary[:280]}] if summary else []),
                    missing_information=([] if summary else ["summary"]),
                ),
                "location_alignment": self._dimension(
                    score=location_alignment,
                    match_type=MatchType.EXPLICIT if location_value else MatchType.UNKNOWN,
                    reasons=(["Location aligns with role requirement."] if loc_hit else ["Location alignment unclear."]),
                    evidence=([{"field": "location", "source": "candidate.location", "value": location_value}] if location_value else []),
                    missing_information=([] if location_value or not location else ["location"]),
                ),
                "stability_signal": self._dimension(
                    score=stability_signal,
                    match_type=MatchType.UNKNOWN,
                    reasons=["Stability signal remains heuristic with current profile data."],
                    evidence=[],
                    missing_information=["tenure_history"],
                ),
            }

            reasons = self._headline_reasons(
                role_spec=role_spec,
                candidate=c,
                dimension_scores=dimension_scores,
                loc_hit=loc_hit,
            )

            risks: list[str] = []
            if not required_hit and not keyword_hit and required:
                risks.append("Limited direct evidence for required skills.")
            if not loc_hit and location:
                risks.append("Location alignment unclear.")
            if sector_target and sector_target not in sector_blob:
                risks.append("Sector relevance is inferred rather than directly evidenced.")

            evidence = [item for dimension in dimension_scores.values() for item in dimension["evidence"]]
            missing_information = sorted({item for dimension in dimension_scores.values() for item in dimension["missing_information"]})
            dimension_match_types = [dimension["match_type"] for dimension in dimension_scores.values()]
            if MatchType.EXPLICIT in dimension_match_types:
                overall_match_type = MatchType.EXPLICIT
            elif MatchType.INFERRED in dimension_match_types:
                overall_match_type = MatchType.INFERRED
            else:
                overall_match_type = MatchType.UNKNOWN

            ranked.append(
                {
                    "candidate_id": c.get("candidate_id"),
                    "raw_fit_score": round(final * 100, 1),
                    "dimension_scores": dimension_scores,
                    "skill_match_score": round(skill_match * 100, 1),
                    "seniority_score": round(seniority_match * 100, 1),
                    "sector_relevance_score": round(sector_relevance * 100, 1),
                    "location_score": round(location_alignment * 100, 1),
                    "ranking_version": self.strategy_version,
                    "ranking_weights": self.weights,
                    "reasons": reasons or ["General alignment based on available signals."],
                    "summary_reasons": reasons or ["General alignment based on available signals."],
                    "headline_reason": (reasons or ["General alignment based on available signals."])[0],
                    "reasoning": reasons or ["General alignment based on available signals."],
                    "risks": risks,
                    "missing_information": sorted(set(missing_information)),
                    "evidence": evidence,
                    "match_type": overall_match_type,
                    "_raw_final": final,
                }
            )

        ranked.sort(key=lambda x: x["_raw_final"], reverse=True)

        if ranked:
            raw_scores = [item["_raw_final"] for item in ranked]
            min_raw = min(raw_scores)
            max_raw = max(raw_scores)
            for item in ranked:
                item["fit_score"] = self._calibrated_fit_score(item["_raw_final"], min_raw=min_raw, max_raw=max_raw)
                item.pop("_raw_final", None)
        return ranked

    def _sector_relevance(self, sector_target: str, sector_blob: str, summary_blob: str, title_blob: str) -> float:
        if sector_target:
            if sector_target in sector_blob:
                return 1.0
            if sector_target in summary_blob or sector_target in title_blob:
                return 0.8
            return 0.35
        if any(term in sector_blob for term in ("infrastructure", "real assets", "funds management", "superannuation")):
            return 0.8
        return 0.45 if sector_blob else 0.25

    def _seniority_match(self, seniority_target: str, title_blob: str, years_experience: Any) -> float:
        years = int(years_experience) if isinstance(years_experience, int) else None
        if seniority_target:
            if seniority_target in title_blob:
                return 0.95
            if years is not None:
                if years >= 15:
                    return 0.9
                if years >= 10:
                    return 0.72
            return 0.45 if title_blob else 0.25
        if years is not None:
            if years >= 15:
                return 0.85
            if years >= 10:
                return 0.72
        return 0.55 if title_blob else 0.3

    def _functional_similarity(
        self,
        title_target: str,
        search_keywords: list[str],
        functions: list[str],
        title_blob: str,
        full_blob: str,
    ) -> float:
        function_blob = " ".join(functions).lower()
        direct_terms = [
            term
            for term in [
                "portfolio management",
                "investment",
                "strategy",
                "governance",
                "operations",
                "distribution",
            ]
            if term in (title_target + " " + " ".join(search_keywords))
        ]
        direct_hits = sum(1 for term in direct_terms if term in function_blob or term in full_blob)
        title_hits = sum(1 for term in search_keywords[:4] if term and term in title_blob)
        score = 0.35 + (0.2 * min(direct_hits, 2)) + (0.1 * min(title_hits, 3))
        if "portfolio manager" in title_blob or "portfolio management" in function_blob:
            score += 0.15
        return min(1.0, score)

    def _stability_signal(self, years_experience: Any, employment_history: list[dict[str, Any]]) -> float:
        years = int(years_experience) if isinstance(years_experience, int) else None
        role_count = len(employment_history or [])
        score = 0.35
        if years is not None:
            score += min(0.3, years / 50)
        if role_count >= 3:
            score += 0.2
        elif role_count == 2:
            score += 0.1
        return min(0.95, score)

    def _location_matches(self, location_target: str, candidate_location: Any) -> bool:
        if not location_target or not candidate_location:
            return False
        location_value = str(candidate_location).lower()
        if location_target in location_value:
            return True
        australian_markers = ("sydney", "melbourne", "brisbane", "perth", "australia")
        if "australia" in location_target and any(marker in location_value for marker in australian_markers):
            return True
        return False

    def _normalize_location_target(self, location: Any) -> str:
        return normalize_location_text(location).lower()

    def _dimension(
        self,
        *,
        score: float,
        match_type: MatchType,
        reasons: list[str],
        evidence: list[dict[str, Any]],
        missing_information: list[str],
    ) -> dict[str, Any]:
        return {
            "score": round(score, 3),
            "match_type": match_type,
            "reasons": reasons,
            "evidence": evidence,
            "missing_information": missing_information,
        }

    def _calibrated_fit_score(self, raw_final: float, *, min_raw: float, max_raw: float) -> float:
        if max_raw - min_raw < 0.03:
            # When candidates are similarly relevant, use a gentler uplift so the shortlist is still readable.
            return round((0.15 + (0.85 * raw_final)) * 100, 1)
        normalized = (raw_final - min_raw) / (max_raw - min_raw)
        return round(52 + (normalized * 43), 1)

    def _headline_reasons(
        self,
        *,
        role_spec: dict[str, Any],
        candidate: dict[str, Any],
        dimension_scores: dict[str, dict[str, Any]],
        loc_hit: int,
    ) -> list[str]:
        reasons: list[str] = []
        current_title = candidate.get("current_title") or "Current title unavailable"
        location_value = candidate.get("location")
        sectors = candidate.get("sectors") or []
        title = normalize_text_field(role_spec.get("title")) or "the mandate"

        skill_score = dimension_scores["skill_match"]["score"]
        if skill_score >= 0.7:
            reasons.append(f"{current_title} shows strong direct overlap with {title} requirements.")
        elif skill_score >= 0.45:
            reasons.append(f"{current_title} has partial keyword and experience overlap with {title}.")

        seniority_score = dimension_scores["seniority_match"]["score"]
        if seniority_score >= 0.8:
            reasons.append(f"Seniority looks aligned based on title: {current_title}.")

        sector_score = dimension_scores["sector_relevance"]["score"]
        if sector_score >= 0.7 and sectors:
            reasons.append(f"Sector background includes {', '.join(sectors[:2])}.")
        elif sector_score >= 0.5 and sectors:
            reasons.append(f"Sector relevance is directionally useful through {', '.join(sectors[:2])}.")

        if loc_hit and location_value:
            reasons.append(f"Location matches the mandate: {location_value}.")

        function_score = dimension_scores["functional_similarity"]["score"]
        functions = candidate.get("functions") or []
        if function_score >= 0.7 and functions:
            reasons.append(f"Functional scope covers {', '.join(functions[:2])}.")

        deduped: list[str] = []
        for reason in reasons:
            if reason not in deduped:
                deduped.append(reason)
        return deduped[:3] or ["Broad partial alignment; manual review is still needed."]
