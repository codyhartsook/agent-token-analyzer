"""CLI entry point for token & context analysis.

Thin wrapper over the library API — all analysis logic lives in
``analyzer.py`` and friends, making it easy to reuse from an ADK agent.
"""

from __future__ import annotations

import argparse
import sys

from .analyzer import analyze_trace_tokens, analyze_window, discover_agents
from .client import get_client, relative_to_nano, to_nano, from_nano
from .report import format_report, write_reports


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="token-analysis",
        description="Token & Context Window Analysis for Agent Traces",
    )

    # ── Time-window selection (mutually exclusive) ──────────────────────
    window = p.add_argument_group("time window (choose one)")
    mx = window.add_mutually_exclusive_group(required=True)
    mx.add_argument(
        "--trace-id",
        help="Deep analysis of a single trace",
    )
    mx.add_argument(
        "--last",
        metavar="SPEC",
        help="Relative window, e.g. '30m', '2h', '1d'",
    )
    mx.add_argument(
        "--start",
        metavar="ISO",
        help="Absolute start (ISO-8601 UTC). Requires --end.",
    )
    mx.add_argument(
        "--discover",
        metavar="SPEC",
        help=(
            "Discover all agents in a time window. "
            "Accepts relative spec (e.g. '1h', '24h', '7d') or 'all'. "
            "Shows agent names usable with --agent."
        ),
    )

    window.add_argument(
        "--end",
        metavar="ISO",
        help="Absolute end (ISO-8601 UTC). Requires --start.",
    )

    # ── Filters ─────────────────────────────────────────────────────────
    filters = p.add_argument_group("filters")
    filters.add_argument(
        "--agent",
        metavar="NAME",
        help="Filter to a specific agent name",
    )

    # ── Analysis options ────────────────────────────────────────────────
    analysis = p.add_argument_group("analysis options")
    analysis.add_argument(
        "--cost",
        action="store_true",
        help="Include cost estimation",
    )
    analysis.add_argument(
        "--accumulation-threshold",
        type=float,
        default=2.0,
        metavar="FACTOR",
        help="Context growth factor to trigger alerts (default: 2.0)",
    )

    # ── Output ──────────────────────────────────────────────────────────
    output = p.add_argument_group("output")
    output.add_argument(
        "--format",
        dest="fmt",
        choices=["terminal", "json", "csv", "all"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    output.add_argument(
        "--output-dir",
        metavar="PATH",
        default="./analysis_output",
        help="Directory for json/csv files (default: ./analysis_output)",
    )
    output.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max traces to analyse (default: 100)",
    )
    output.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-call details and full context windows",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate --start/--end pairing
    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.end and not args.start:
        parser.error("--end requires --start")

    # Connect
    try:
        client = get_client()
    except Exception as e:
        print(f"Error connecting to ClickHouse: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine mode + run analysis
    try:
        if args.discover:
            # Discovery mode
            if args.discover.lower() == "all":
                # Use a very wide window (365 days)
                start_ns, end_ns = relative_to_nano("365d")
            else:
                start_ns, end_ns = relative_to_nano(args.discover)
            analysis = discover_agents(client, start_ns, end_ns)
        elif args.trace_id:
            analysis = analyze_trace_tokens(
                client,
                args.trace_id,
                include_cost=args.cost,
                growth_factor_warn=args.accumulation_threshold,
            )
        else:
            # Time window
            if args.last:
                start_ns, end_ns = relative_to_nano(args.last)
            else:
                start_ns = to_nano(args.start)
                end_ns = to_nano(args.end)

            analysis = analyze_window(
                client,
                start_ns,
                end_ns,
                agent_filter=args.agent,
                include_cost=args.cost,
                limit=args.limit,
                growth_factor_warn=args.accumulation_threshold,
            )
    except Exception as e:
        print(f"Analysis error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.fmt == "all":
        # Terminal to stdout + files to disk
        print(format_report(analysis, "terminal", verbose=args.verbose))
        files = write_reports(analysis, args.output_dir, verbose=args.verbose)
        print(f"\nFiles written to {args.output_dir}/:")
        for f in files:
            print(f"  {f.name}")
    elif args.fmt in ("json", "csv"):
        print(format_report(analysis, args.fmt, verbose=args.verbose))
    else:
        print(format_report(analysis, "terminal", verbose=args.verbose))
