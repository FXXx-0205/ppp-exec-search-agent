from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, Sequence
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from app.ppp.enrichment import CandidatePublicProfileLookupInput

logger = logging.getLogger(__name__)

AUM_SCALED_VALUE_PATTERN = re.compile(
    r"(?i)(?:A\$|AUD|USD|\$|£|€)?\s*\d+(?:\.\d+)?\s*(?:B|M|bn|mn|billion|million|trillion)\b"
)
AUM_LARGE_AMOUNT_PATTERN = re.compile(
    r"(?i)(?:A\$|AUD|USD|\$|£|€)\s*\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?\b"
)
COMPANY_SUFFIX_PATTERN = re.compile(
    r"(?i)\b(?:group|company|co|ltd|limited|pty|inc|llc|holdings?)\b|&\s*(?:co|company)\b"
)
NEGATIVE_CONTEXT_WORDS = [
    "former",
    "previously",
    "prior",
    "worked at",
    "acquired",
    "subsidiary",
    "merger",
    "competitor",
]
GLOBAL_AUM_ALLOWLIST = {"blackrock", "vanguard"}
YEAR_CONTEXT_WORDS = ("founded", "since", "established", "in", "year")


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
        try:
            candidate_query = self._build_candidate_query(tool_input)
            firm_query = self._build_firm_query(tool_input)
            logger.info("Tavily candidate query for %s: %s", tool_input.full_name, candidate_query)
            logger.info("Tavily firm query for %s: %s", tool_input.current_employer, firm_query)

            candidate_payload = self._run_search(candidate_query)
        except httpx.HTTPError as exc:
            raise ResearchClientError(f"Tavily lookup failed: {exc}") from exc

        raw_results = candidate_payload.get("results", [])
        if not isinstance(raw_results, list):
            raise ResearchClientError("Tavily returned no usable public-web results.")

        candidate_results = self._filter_candidate_results(tool_input, raw_results)
        identity_resolution = self._assess_identity_resolution(tool_input, raw_results)

        firm_results: list[dict[str, object]] = []
        try:
            firm_payload = self._run_search(firm_query)
            raw_firm_results = firm_payload.get("results", [])
            if isinstance(raw_firm_results, list) and raw_firm_results:
                firm_results = self._filter_firm_results(tool_input.current_employer, raw_firm_results)
        except ResearchClientError as exc:
            logger.warning("Tavily firm-context query failed for %s: %s", tool_input.current_employer, exc)

        fixture = self._results_to_fixture(
            tool_input,
            candidate_results,
            firm_results=firm_results,
            identity_resolution=identity_resolution,
        )
        return ResearchPayload(mode="live_web_tavily", fixture=fixture)

    def _run_search(self, query: str) -> dict[str, object]:
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
        if not isinstance(payload, dict):
            raise ResearchClientError("Tavily returned an invalid payload.")
        return payload

    def _build_candidate_query(self, tool_input: CandidatePublicProfileLookupInput) -> str:
        return " ".join(
            [
                f'"{tool_input.full_name}"',
                f'"{tool_input.current_employer}"',
                f'"{tool_input.current_title}"',
                "LinkedIn Australia asset management wealth distribution executive biography leadership profile",
            ]
        )

    def _build_firm_query(self, tool_input: CandidatePublicProfileLookupInput) -> str:
        return " ".join(
            [
                f'"{tool_input.current_employer}"',
                '"assets under management"',
                '"total AUM"',
                "Australia asset manager funds management",
            ]
        )

    def _filter_candidate_results(
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

    def _filter_firm_results(self, employer: str, results: list[object]) -> list[dict[str, object]]:
        scored_results: list[tuple[int, dict[str, object]]] = []
        employer_tokens = {token for token in re.findall(r"[a-zA-Z]{3,}", employer.lower())}
        for item in results:
            if not isinstance(item, dict):
                continue
            score = self._score_firm_result(employer, employer_tokens, item)
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

    def _score_firm_result(self, employer: str, employer_tokens: set[str], result: dict[str, object]) -> int:
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
        matched_tokens = [token for token in employer_tokens if token in combined]
        if not matched_tokens:
            return -5

        score = len(matched_tokens) * 2
        if any(term in combined for term in ("aum", "assets under management", "funds under management")):
            score += 5
        if self._contains_numeric_aum(combined_text):
            score += 5
        if "australia" in combined:
            score += 1
        if any(pattern in host for pattern in self.PREFERRED_HOST_PATTERNS):
            score += 3
        return score

    def _results_to_fixture(
        self,
        tool_input: CandidatePublicProfileLookupInput,
        results: Sequence[object],
        *,
        firm_results: Sequence[object],
        identity_resolution: dict[str, object],
    ) -> dict[str, object]:
        snippets: list[str] = []
        firm_snippets: list[str] = []
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

        for item in firm_results:
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
                firm_snippets.append(snippet)
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
                        "category": "firm_context",
                        "signal": "public_web_search_firm_context",
                        "snippet": snippet,
                        "source_labels": [title or url],
                    }
                )

        status = str(identity_resolution.get("status", "not_verified"))
        rationale = str(identity_resolution.get("rationale", "")).strip()
        matched_source_labels = [
            str(label).strip()
            for label in identity_resolution.get("matched_source_labels", [])
            if str(label).strip()
        ]
        possible_snippets = [
            str(snippet).strip()
            for snippet in identity_resolution.get("possible_public_snippets", [])
            if str(snippet).strip()
        ]
        if not snippets and possible_snippets:
            snippets = possible_snippets[:2]
        if not snippets:
            snippets = [
                f"Public search did not verify an exact-match profile for {tool_input.full_name} at {tool_input.current_employer}."
            ]

        inferred_tenure_years = self._infer_tenure_years(snippets)
        firm_aum_context = self._extract_firm_aum_context(tool_input.current_employer, firm_snippets or snippets)
        likely_channel_evidence = self._collect_channel_evidence(tool_input.current_title, snippets)
        likely_experience_evidence = self._collect_experience_evidence(tool_input.current_employer, snippets + firm_snippets)
        uncertain_fields = self._derive_uncertain_fields(snippets + firm_snippets, inferred_tenure_years, firm_aum_context)
        combined_context = self._assemble_context(snippets, firm_snippets)
        confidence_notes = [
            rationale or f"Identity resolution for {tool_input.full_name} remains incomplete from public-web search.",
            f"Live public-web research was attempted through Tavily for {tool_input.full_name}.",
            f"Candidate profile query: {self._build_candidate_query(tool_input)}",
            f"Firm context query: {self._build_firm_query(tool_input)}",
            "Only publicly available snippets were captured, so chronology and coverage scope may still require manual verification.",
            self._build_trajectory_hint(tool_input, inferred_tenure_years),
        ]
        if not firm_snippets:
            confidence_notes.append("Firm context search did not return a usable AUM snippet, so company-scale evidence remains limited.")

        return {
            "identity_resolution_status": status,
            "identity_resolution_rationale": rationale,
            "identity_resolution_source_labels": matched_source_labels,
            "verified_public_snippets": snippets[:3] if status == "verified_match" else [],
            "possible_public_snippets": possible_snippets[:3] if status != "verified_match" else [],
            "tenure_years": inferred_tenure_years,
            "tenure_rationale": self._build_tenure_rationale(inferred_tenure_years),
            "trajectory_hint": self._build_trajectory_hint(tool_input, inferred_tenure_years),
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
            "combined_context": combined_context,
        }

    def _assemble_context(self, candidate_snippets: list[str], firm_snippets: list[str]) -> str:
        candidate_block = "\n".join(candidate_snippets[:2]).strip() or "No candidate profile context found."
        firm_block = "\n".join(firm_snippets[:2]).strip() or "No specific firm context found."
        return (
            "=== CANDIDATE PROFILE ===\n"
            f"{candidate_block}\n\n"
            "=== FIRM CONTEXT & AUM ===\n"
            f"{firm_block}"
        )

    def _compose_snippet(self, *, title: str, content: str) -> str:
        text = " ".join(part for part in [title, content] if part).strip()
        text = re.sub(r"\s+", " ", text)
        return text[:320].strip()

    def _assess_identity_resolution(
        self,
        tool_input: CandidatePublicProfileLookupInput,
        results: Sequence[object],
    ) -> dict[str, object]:
        analyses: list[dict[str, object]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            analysis = self._analyze_candidate_result(tool_input, item)
            if analysis["exact_name"]:
                analyses.append(analysis)

        analyses.sort(key=lambda item: int(item["score"]), reverse=True)

        verified = [
            item
            for item in analyses
            if item["employer_match"] and (item["title_match"] or item["linkedin_match"] or item["high_confidence_source"])
        ]
        if verified:
            top = verified[0]
            return {
                "status": "verified_match",
                "rationale": (
                    f"Public search found a verified match for {tool_input.full_name} at {tool_input.current_employer}, "
                    f"supported by {top['source_label']}."
                ),
                "matched_source_labels": [str(top["source_label"])],
                "possible_public_snippets": [str(top["snippet"])],
            }

        possible = [item for item in analyses if item["employer_match"] or item["title_match"]]
        if possible:
            top = possible[0]
            return {
                "status": "possible_match",
                "rationale": (
                    f"Public search surfaced a possible match for {tool_input.full_name}, but the exact identity at "
                    f"{tool_input.current_employer} remains unconfirmed."
                ),
                "matched_source_labels": [str(top["source_label"])],
                "possible_public_snippets": [str(item["snippet"]) for item in possible[:2] if str(item["snippet"]).strip()],
            }

        return {
            "status": "not_verified",
            "rationale": (
                f"Public search did not verify an exact-match profile for {tool_input.full_name} at {tool_input.current_employer}; "
                "the task input should therefore be treated as unverified."
            ),
            "matched_source_labels": [],
            "possible_public_snippets": [],
        }

    def _analyze_candidate_result(
        self,
        tool_input: CandidatePublicProfileLookupInput,
        result: dict[str, object],
    ) -> dict[str, object]:
        title = str(result.get("title", "")).strip()
        content = str(result.get("content", "")).strip()
        url = str(result.get("url", "")).strip()
        combined_text = f"{title} {content}".strip()
        combined = combined_text.lower()
        source_type = self._source_type(url)
        exact_name = tool_input.full_name.lower() in combined
        employer_match = tool_input.current_employer.lower() in combined
        title_match = any(token in combined for token in self._significant_title_tokens(tool_input.current_title))
        linkedin_match = self._linkedin_identity_match(tool_input.linkedin_url, url)
        score = self._score_result(tool_input, result)
        return {
            "exact_name": exact_name,
            "employer_match": employer_match,
            "title_match": title_match,
            "linkedin_match": linkedin_match,
            "high_confidence_source": source_type in {"linkedin_public", "company_site", "regulatory_public", "industry_biography"},
            "score": score,
            "source_label": title or url,
            "snippet": self._compose_snippet(title=title, content=content),
        }

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
        if any(token in lowered for token in ("aum", "assets under management", "funds under management")) or self._contains_numeric_aum(snippet):
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
        normalized_employer = self._normalize_company_name(employer)
        preferred_sentence: str | None = None
        preferred_amount: str | None = None

        for snippet in snippets:
            amount, context_window = self._extract_attributed_aum(snippet, normalized_employer=normalized_employer)
            if amount is not None:
                preferred_sentence = self._best_firm_aum_sentence(context_window or snippet) or (context_window or snippet)
                preferred_amount = amount
                break

        if preferred_amount is not None and preferred_sentence is not None:
            return (
                f"{employer} appears to be an active asset manager with an estimated AUM of {preferred_amount}, "
                f"{self._firm_context_qualifier(preferred_sentence)}"
            )

        for snippet in snippets:
            lowered = snippet.lower()
            if "aum" in lowered or "assets under management" in lowered or "funds under management" in lowered:
                return f"{employer} appears in public-web snippets with firm-scale references, but the exact AUM figure should still be verified manually."
        return f"{employer} appears in public-web results as an established investment or wealth platform, but exact AUM still requires direct verification."

    def _contains_numeric_aum(self, text: str) -> bool:
        return bool(AUM_SCALED_VALUE_PATTERN.search(text) or AUM_LARGE_AMOUNT_PATTERN.search(text))

    def _first_aum_value(self, text: str) -> str | None:
        match = AUM_SCALED_VALUE_PATTERN.search(text) or AUM_LARGE_AMOUNT_PATTERN.search(text)
        return match.group(0).strip() if match else None

    def _normalize_company_name(self, employer: str) -> str:
        cleaned = COMPANY_SUFFIX_PATTERN.sub(" ", employer)
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.lower()

    def _extract_attributed_aum(self, text: str, *, normalized_employer: str) -> tuple[str | None, str | None]:
        matches = list(AUM_SCALED_VALUE_PATTERN.finditer(text)) + list(AUM_LARGE_AMOUNT_PATTERN.finditer(text))
        matches.sort(key=lambda item: item.start())
        if not matches:
            return None, None

        normalized_text = self._normalize_company_name(text)
        lowered_text = text.lower()
        for match in matches:
            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 150)
            context_window = text[start:end].strip()
            lowered_window = context_window.lower()
            sentence = self._sentence_containing_span(text, match.start(), match.end())
            sentence_normalized = self._normalize_company_name(sentence)

            if self._is_invalid_year_match(text, match):
                continue
            if self._fails_aum_sanity_check(match.group(0), normalized_employer=normalized_employer):
                continue
            if any(word in lowered_window for word in NEGATIVE_CONTEXT_WORDS):
                continue
            if normalized_employer and normalized_employer not in sentence_normalized:
                if not self._employer_is_near_amount(context_window, match, offset=start, normalized_employer=normalized_employer):
                    snippet_has_employer = normalized_employer in normalized_text
                    snippet_has_aum_signal = any(
                        term in lowered_text for term in ("aum", "assets under management", "funds under management")
                    )
                    if not (snippet_has_employer and snippet_has_aum_signal):
                        continue
            return match.group(0).strip(), context_window

        return None, text[:300].strip() if text else None

    def _best_firm_aum_sentence(self, snippet: str) -> str | None:
        segments = [
            segment.strip(" -:;,.")
            for segment in re.split(r"(?<=[.!?])\s+|(?<=;)\s+", snippet)
            if segment and segment.strip()
        ]
        candidates = [segment for segment in segments if self._looks_like_natural_aum_sentence(segment)]
        if candidates:
            candidates.sort(key=self._firm_sentence_score, reverse=True)
            return candidates[0]
        if self._looks_like_natural_aum_sentence(snippet):
            return snippet.strip()
        return None

    def _looks_like_natural_aum_sentence(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        lowered = normalized.lower()
        if len(normalized) < 20:
            return False
        if not self._contains_numeric_aum(normalized):
            return False
        if not any(term in lowered for term in ("aum", "assets under management", "funds under management")):
            return False
        if "($b)" in lowered or "($m)" in lowered:
            return False
        if lowered.count("|") >= 2 or lowered.count(" --- ") >= 1:
            return False
        return True

    def _firm_sentence_score(self, text: str) -> tuple[int, int]:
        lowered = text.lower()
        return (
            int("assets under management" in lowered or "funds under management" in lowered or "aum" in lowered),
            len(text),
        )

    def _firm_context_qualifier(self, sentence: str) -> str:
        lowered = sentence.lower()
        if any(term in lowered for term in ("acquisition", "acquired", "merger", "group fum", "transaction")):
            return "based on transaction-related public references and still subject to verification."
        if any(term in lowered for term in ("13f", "sec", "crd #", "cik #", "employees |", "aum ($b)")):
            return "based on regulatory and public-web references and still subject to verification."
        return "based on recent public-web references and still subject to verification."

    def _sentence_containing_span(self, text: str, start: int, end: int) -> str:
        left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
        right_candidates = [idx for idx in (text.find(".", end), text.find("!", end), text.find("?", end)) if idx != -1]
        right = min(right_candidates) if right_candidates else len(text)
        sentence = text[left + 1 : right].strip()
        return sentence or text.strip()

    def _employer_is_near_amount(self, context_window: str, match: re.Match[str], *, offset: int, normalized_employer: str) -> bool:
        if not normalized_employer:
            return True
        employer_pattern = re.compile(r"\b" + r"\W+".join(re.escape(part) for part in normalized_employer.split()) + r"\b", re.IGNORECASE)
        local_match_start = match.start() - offset
        for employer_match in employer_pattern.finditer(context_window):
            distance = min(abs(local_match_start - employer_match.end()), abs(match.end() - offset - employer_match.start()))
            if distance <= 40:
                return True
        return False

    def _is_invalid_year_match(self, text: str, match: re.Match[str]) -> bool:
        amount = match.group(0)
        numeric_value, _ = self._parse_aum_amount(amount)
        if numeric_value is None or not (1990 <= numeric_value <= 2026):
            return False
        local_start = max(0, match.start() - 10)
        local_end = min(len(text), match.end() + 10)
        nearby = text[local_start:local_end].lower()
        return any(re.search(rf"\b{re.escape(word)}\b", nearby) for word in YEAR_CONTEXT_WORDS)

    def _fails_aum_sanity_check(self, amount: str, *, normalized_employer: str) -> bool:
        numeric_value, unit = self._parse_aum_amount(amount)
        if numeric_value is None or unit is None:
            return False
        if unit == "billion" and numeric_value > 1000 and normalized_employer not in GLOBAL_AUM_ALLOWLIST:
            return True
        return False

    def _parse_aum_amount(self, amount: str) -> tuple[float | None, str | None]:
        cleaned = amount.strip().lower().replace(",", "")
        match = re.search(
            r"(?i)(?:a\$|aud|usd|\$|£|€)?\s*(\d+(?:\.\d+)?)\s*(b|m|bn|mn|billion|million|trillion)?\b",
            cleaned,
        )
        if match is None:
            return None, None
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        if unit in {"b", "bn", "billion"}:
            return value, "billion"
        if unit in {"m", "mn", "million"}:
            return value, "million"
        if unit == "trillion":
            return value, "trillion"
        if "," in amount:
            return value, "raw_currency"
        return value, None

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

    def _build_trajectory_hint(self, tool_input: CandidatePublicProfileLookupInput, inferred_tenure_years: float | None) -> str:
        title = tool_input.current_title.lower()
        if inferred_tenure_years is not None:
            return f"Public chronology suggests an established current-role tenure of about {inferred_tenure_years:.1f} years for the visible remit."
        if "head" in title or "director" in title:
            return "Public chronology is incomplete, but the visible role shape suggests an already-scaled senior remit rather than a brand-new step-up."
        if "bdm" in title or "sales" in title:
            return "Public chronology is incomplete, though the visible role shape suggests hands-on channel ownership rather than purely strategic oversight."
        return "Public chronology is incomplete and trajectory should be treated cautiously until a fuller public timeline is verified."

    def _hostname(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _significant_title_tokens(self, current_title: str) -> list[str]:
        ignored_tokens = {"head", "senior", "director"}
        return [token for token in re.findall(r"[a-zA-Z]{4,}", current_title.lower()) if token not in ignored_tokens]
