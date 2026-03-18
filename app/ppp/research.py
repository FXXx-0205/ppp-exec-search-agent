from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, Sequence
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from app.ppp.enrichment import CandidatePublicProfileLookupInput


class ResearchClientError(Exception):
    pass


@dataclass(frozen=True)
class ResearchPayload:
    mode: str
    fixture: dict[str, object]


class PublicResearchClient(Protocol):
    def lookup_candidate(self, tool_input: CandidatePublicProfileLookupInput) -> ResearchPayload:
        ...


class TavilyResearchClient:
    SOURCE_PRIORITY = {
        "company_site": 1.0,
        "linkedin_public": 0.95,
        "regulatory_public": 0.95,
        "industry_biography": 0.7,
        "public_web": 0.45,
    }
    PREFERRED_HOST_PATTERNS = (
        "linkedin.com",
        "asx.com.au",
        ".gov.au",
        "apra.gov.au",
        "asic.gov.au",
        "morningstar.com.au",
        "financialstandard.com.au",
        "investmentmagazine.com.au",
        "professionalplanner.com.au",
        "ifa.com.au",
        "company",
        "leadership",
        "team",
        "about",
    )
    EXCLUDED_HOST_PATTERNS = (
        "facebook.com",
        "instagram.com",
        "x.com",
        "twitter.com",
        "tiktok.com",
        "youtube.com",
        "seek.com.au",
        "glassdoor.com",
        "wikipedia.org",
    )

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 20.0,
        max_results: int = 8,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def lookup_candidate(self, tool_input: CandidatePublicProfileLookupInput) -> ResearchPayload:
        query = self._build_query(tool_input)
        try:
            response = self.client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max(self.max_results * 2, 10),
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ResearchClientError(f"Tavily lookup failed: {exc}") from exc

        payload = response.json()
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list) or not raw_results:
            raise ResearchClientError("Tavily returned no usable public-web results.")

        filtered_results = self._filter_results(tool_input, raw_results)
        if not filtered_results:
            raise ResearchClientError("Tavily returned results, but none passed relevance and source-quality filters.")

        fixture = self._results_to_fixture(tool_input, filtered_results)
        return ResearchPayload(mode="live_web_tavily", fixture=fixture)

    def _build_query(self, tool_input: CandidatePublicProfileLookupInput) -> str:
        return " ".join(
            [
                f'"{tool_input.full_name}"',
                f'"{tool_input.current_employer}"',
                f'"{tool_input.current_title}"',
                "Australia asset management wealth distribution executive biography leadership profile",
            ]
        )

    def _filter_results(
        self,
        tool_input: CandidatePublicProfileLookupInput,
        results: list[object],
    ) -> list[dict[str, object]]:
        scored_results: list[tuple[int, dict[str, object]]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            score = self._score_result(tool_input, item)
            if score < 4:
                continue
            scored_results.append((score, item))

        scored_results.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored_results[: self.max_results]]

    def _score_result(self, tool_input: CandidatePublicProfileLookupInput, result: dict[str, object]) -> int:
        title = str(result.get("title", "")).strip()
        content = str(result.get("content", "")).strip()
        url = str(result.get("url", "")).strip()
        if not url:
            return -10

        host = self._hostname(url)
        if any(pattern in host for pattern in self.EXCLUDED_HOST_PATTERNS):
            return -10

        combined_text = f"{title} {content}".strip()
        combined = combined_text.lower()
        exact_name = tool_input.full_name.lower() in combined
        employer_match = tool_input.current_employer.lower() in combined
        title_match = any(token in combined for token in self._significant_title_tokens(tool_input.current_title))
        linkedin_match = self._linkedin_identity_match(tool_input.linkedin_url, url)

        if not exact_name:
            return -5
        if not employer_match and not title_match and not linkedin_match:
            return -1
        if not employer_match and not linkedin_match and self._mentions_other_employer(combined_text, tool_input.current_employer):
            return -5

        score = 0
        if exact_name:
            score += 4
        if employer_match:
            score += 4
        if title_match:
            score += 2
        if linkedin_match:
            score += 4
        if "australia" in combined:
            score += 1
        if any(pattern in host for pattern in self.PREFERRED_HOST_PATTERNS):
            score += 3
        if any(term in combined for term in ("distribution", "wholesale", "institutional", "funds management", "asset management")):
            score += 1
        return score

    def _results_to_fixture(
        self,
        tool_input: CandidatePublicProfileLookupInput,
        results: Sequence[object],
    ) -> dict[str, object]:
        snippets: list[str] = []
        sources: list[dict[str, object]] = []
        evidence: list[dict[str, object]] = []
        seen_urls: set[str] = set()

        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            snippet = self._compose_snippet(title=title, content=content)
            if snippet:
                snippets.append(snippet)
            sources.append(
                {
                    "label": title or url,
                    "source_type": self._source_type(url),
                    "url": url,
                    "confidence": self._source_confidence(url),
                    "trust_score": self.SOURCE_PRIORITY.get(self._source_type(url), 0.4),
                }
            )
            if snippet:
                evidence.append(
                    {
                        "category": self._classify_category(snippet),
                        "signal": "public_web_search",
                        "snippet": snippet,
                        "source_labels": [title or url],
                    }
                )

        if not snippets:
            raise ResearchClientError("Public-web results did not contain usable snippets.")

        inferred_tenure_years = self._infer_tenure_years(snippets)
        firm_aum_context = self._extract_firm_aum_context(tool_input.current_employer, snippets)
        likely_channel_evidence = self._collect_channel_evidence(tool_input.current_title, snippets)
        likely_experience_evidence = self._collect_experience_evidence(tool_input.current_employer, snippets)
        uncertain_fields = self._derive_uncertain_fields(snippets, inferred_tenure_years, firm_aum_context)
        confidence_notes = [
            f"Live public-web research was attempted through Tavily for {tool_input.full_name}.",
            "Only publicly available snippets were captured, so chronology and coverage scope may still require manual verification.",
        ]

        return {
            "verified_public_snippets": snippets[:3],
            "tenure_years": inferred_tenure_years,
            "tenure_rationale": self._build_tenure_rationale(inferred_tenure_years),
            "firm_aum_context": firm_aum_context,
            "likely_channel_evidence": likely_channel_evidence,
            "likely_experience_evidence": likely_experience_evidence,
            "mobility_evidence": [
                "Recent public-web snippets should be reviewed manually to confirm whether the candidate is in an authentic transition window."
            ],
            "missing_fields": [
                "verified current-role start date",
                "verified reporting line / team size",
                "direct evidence of channel revenue ownership",
            ],
            "uncertain_fields": uncertain_fields,
            "confidence_notes": confidence_notes,
            "sources": sources,
            "evidence": evidence[:5],
        }

    def _compose_snippet(self, *, title: str, content: str) -> str:
        text = " ".join(part for part in [title, content] if part).strip()
        text = re.sub(r"\s+", " ", text)
        return text[:320].strip()

    def _source_type(self, url: str) -> str:
        host = self._hostname(url)
        lowered = url.lower()
        if "linkedin.com" in host:
            return "linkedin_public"
        if ".gov" in host or "asx.com.au" in host:
            return "regulatory_public"
        if any(token in lowered for token in ("company", "about", "leadership", "team")):
            return "company_site"
        if any(token in host for token in ("morningstar", "financialstandard", "investmentmagazine", "professionalplanner", "ifa.com.au")):
            return "industry_biography"
        return "public_web"

    def _source_confidence(self, url: str) -> str:
        source_type = self._source_type(url)
        if source_type in {"linkedin_public", "regulatory_public", "company_site"}:
            return "high"
        if source_type == "industry_biography":
            return "medium"
        return "low"

    def _linkedin_identity_match(self, candidate_linkedin_url: str, result_url: str) -> bool:
        if "linkedin.com" not in candidate_linkedin_url.lower() or "linkedin.com" not in result_url.lower():
            return False
        expected_path = urlparse(candidate_linkedin_url).path.rstrip("/").lower()
        actual_path = urlparse(result_url).path.rstrip("/").lower()
        return bool(expected_path and actual_path and expected_path == actual_path)

    def _mentions_other_employer(self, text: str, expected_employer: str) -> bool:
        employer_tokens = {token for token in re.findall(r"[a-zA-Z]{4,}", expected_employer.lower())}
        capitalized_phrases = re.findall(r"\b[A-Z][A-Za-z&.-]+(?:\s+[A-Z][A-Za-z&.-]+){0,3}\b", text)
        for phrase in capitalized_phrases:
            phrase_lower = phrase.lower()
            phrase_tokens = {token for token in re.findall(r"[a-zA-Z]{4,}", phrase_lower)}
            if phrase_tokens and phrase_tokens.isdisjoint(employer_tokens) and phrase_lower not in {"australia"}:
                return True
        return False

    def _classify_category(self, snippet: str) -> str:
        lowered = snippet.lower()
        if any(token in lowered for token in ("aum", "assets under management", "$", "aud")):
            return "firm_context"
        if any(token in lowered for token in ("joined", "since", "appointed", "promoted")):
            return "tenure"
        return "current_role"

    def _infer_tenure_years(self, snippets: list[str]) -> float | None:
        current_year_match = re.compile(r"\b(20\d{2})\b")
        current_year = datetime.now(UTC).year
        for snippet in snippets:
            lowered = snippet.lower()
            if not any(token in lowered for token in ("joined", "since", "appointed", "promoted")):
                continue
            match = current_year_match.search(snippet)
            if not match:
                continue
            start_year = int(match.group(1))
            if 2010 <= start_year <= 2030:
                # Keep this conservative because public snippets often omit exact months.
                return round(max(0.5, current_year - start_year), 1)
        return None

    def _extract_firm_aum_context(self, employer: str, snippets: list[str]) -> str:
        for snippet in snippets:
            lowered = snippet.lower()
            if "aum" in lowered or "assets under management" in lowered or re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", lowered):
                return f"{employer} appears in public-web snippets with firm-scale references, but the exact AUM figure should still be verified manually."
        return f"{employer} appears in public-web results as an established investment or wealth platform, but exact AUM still requires direct verification."

    def _collect_channel_evidence(self, current_title: str, snippets: list[str]) -> list[str]:
        channel_terms = ("distribution", "wholesale", "institutional", "adviser", "consultant", "sales")
        matches = [snippet for snippet in snippets if any(term in snippet.lower() for term in channel_terms)]
        if matches:
            return matches[:2]
        return [f"The title '{current_title}' suggests client-channel responsibility, but public-web confirmation remains limited."]

    def _collect_experience_evidence(self, employer: str, snippets: list[str]) -> list[str]:
        matches = [snippet for snippet in snippets if employer.lower() in snippet.lower()]
        if matches:
            return matches[:2]
        return [f"Public-web search found directional evidence tied to {employer}, but exact remit details still need manual confirmation."]

    def _derive_uncertain_fields(
        self,
        snippets: list[str],
        inferred_tenure_years: float | None,
        firm_aum_context: str,
    ) -> list[str]:
        uncertain_fields: list[str] = []
        if inferred_tenure_years is None:
            uncertain_fields.append("precise current-role tenure")
        if "exact AUM" in firm_aum_context:
            uncertain_fields.append("precise firm AUM")
        if not any("team" in snippet.lower() or "report" in snippet.lower() for snippet in snippets):
            uncertain_fields.append("team leadership scope")
        return uncertain_fields or ["current mobility intent"]

    def _build_tenure_rationale(self, inferred_tenure_years: float | None) -> str:
        if inferred_tenure_years is None:
            return "Exact current-role tenure could not be verified from public-web snippets and remains a manual follow-up item."
        return f"Estimated at approximately {inferred_tenure_years:.1f} years based on dated public-web snippets; this should still be checked manually."

    def _hostname(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _significant_title_tokens(self, current_title: str) -> list[str]:
        ignored_tokens = {"head", "senior", "director"}
        return [token for token in re.findall(r"[a-zA-Z]{4,}", current_title.lower()) if token not in ignored_tokens]
