from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ppp import PPPTaskError, run_ppp_pipeline
from app.ppp.qa import validate_output_bundle, write_qa_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PPP candidate briefing task.")
    parser.add_argument(
        "positional_input",
        nargs="?",
        help="Optional positional CSV path. Equivalent to --input.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Path to the candidate CSV file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="data/ppp/output.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--role-spec",
        dest="role_spec_path",
        default="data/ppp/role_spec.json",
        help="Path to the PPP role specification JSON.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        help="Anthropic model name to use for generation.",
    )
    parser.add_argument(
        "--intermediate-dir",
        default="data/ppp/intermediate",
        help="Directory for enrichment artifacts and debug outputs.",
    )
    parser.add_argument(
        "--research-fixtures",
        default="data/ppp/research_fixtures.json",
        help="Fixture-backed public research source for the PPP enrichment tool.",
    )
    parser.add_argument(
        "--research-mode",
        default=None,
        choices=["fixture", "live", "auto"],
        help="PPP enrichment research mode: fixture-only, live public-web lookup, or auto with fixture fallback.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip generation and only validate an existing output bundle.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input_path or args.positional_input or "data/ppp/candidates.csv"

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    try:
        if args.validate_only:
            report = validate_output_bundle(
                output_path=args.output_path,
                input_path=input_path,
                intermediate_dir=args.intermediate_dir,
            )
            report_path = write_qa_report(report, path=f"{args.intermediate_dir}/qa_report.json")
            if not report.passed:
                logging.error("PPP output validation failed. Review %s", report_path)
                return 2
            logging.info("PPP output validation passed. Report: %s", report_path)
            return 0

        result = run_ppp_pipeline(
            input_path=input_path,
            output_path=args.output_path,
            role_spec_path=args.role_spec_path,
            model=args.model,
            intermediate_dir=args.intermediate_dir,
            research_fixture_path=args.research_fixtures,
            research_mode=args.research_mode,
        )
    except PPPTaskError as exc:
        logging.error("%s", exc)
        return 2
    except Exception:
        logging.exception("PPP task run failed unexpectedly.")
        return 1

    if result.delivery_status == "success":
        logging.info("PPP task completed successfully with %s/5 candidates.", result.successful_candidate_count)
    else:
        logging.warning(
            "PPP task completed with partial success: %s succeeded, %s failed. Review %s",
            result.successful_candidate_count,
            result.failed_candidate_count,
            result.run_report_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
