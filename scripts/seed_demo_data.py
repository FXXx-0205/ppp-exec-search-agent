from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    (raw / "sample_candidates").mkdir(parents=True, exist_ok=True)
    (raw / "sample_roles").mkdir(parents=True, exist_ok=True)
    (raw / "sample_firm_profiles").mkdir(parents=True, exist_ok=True)

    candidates = [
        {
            "candidate_id": "cand_001",
            "full_name": "Alex Chen",
            "current_title": "Senior Portfolio Manager, Infrastructure",
            "current_company": "Example Asset Management",
            "location": "Australia",
            "years_experience": 14,
            "sectors": ["Funds Management", "Infrastructure"],
            "functions": ["Portfolio Management"],
            "summary": "Leads institutional infrastructure portfolio construction and manager selection for Australian superannuation clients.",
            "evidence": [
                "Built infrastructure allocation for institutional investors",
                "Managed multi-asset portfolio construction responsibilities",
            ],
            "source_urls": ["https://example.com/profile/alex-chen"],
            "confidence_score": 0.78,
        },
        {
            "candidate_id": "cand_002",
            "full_name": "Jamie Patel",
            "current_title": "Investment Director",
            "current_company": "Infrastructure Partners",
            "location": "Sydney, Australia",
            "years_experience": 12,
            "sectors": ["Infrastructure", "Private Markets"],
            "functions": ["Direct Investments"],
            "summary": "Direct infrastructure investment experience across core/core-plus; limited explicit institutional PM mandate wording.",
            "evidence": ["Sourced and executed infrastructure deals", "Worked with Australian institutional LPs"],
            "source_urls": ["https://example.com/profile/jamie-patel"],
            "confidence_score": 0.7,
        },
        {
            "candidate_id": "cand_003",
            "full_name": "Morgan Lee",
            "current_title": "Portfolio Manager",
            "current_company": "Global Funds Co",
            "location": "Singapore",
            "years_experience": 10,
            "sectors": ["Funds Management"],
            "functions": ["Portfolio Management"],
            "summary": "Institutional portfolio management across alternatives; Australia relocation/mobility unclear.",
            "evidence": ["Institutional portfolio management", "Alternatives allocation committee exposure"],
            "source_urls": ["https://example.com/profile/morgan-lee"],
            "confidence_score": 0.62,
        },
    ]

    (raw / "sample_candidates" / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    role_samples = [
        {
            "title": "Senior Infrastructure Portfolio Manager",
            "raw_input": "Find senior infrastructure portfolio managers in Australia with institutional funds management experience.",
        }
    ]
    (raw / "sample_roles" / "roles.json").write_text(
        json.dumps(role_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    firm_profiles = [
        {
            "doc_id": "firm_001",
            "title": "Example Asset Management - Investment Team",
            "text": "Example Asset Management manages institutional portfolios with a dedicated infrastructure investment team in Australia.",
            "metadata": {"doc_type": "firm_profile", "sector": "Funds Management", "region": "Australia", "source": "demo"},
        }
    ]
    (raw / "sample_firm_profiles" / "firm_profiles.json").write_text(
        json.dumps(firm_profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Seeded demo data under data/raw/.")


if __name__ == "__main__":
    main()

