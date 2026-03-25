"""Tests for cli.py — argument parsing and e2e CLI tests."""

from __future__ import annotations

import pytest

from token_analysis.cli import _build_parser


# ── Argument parsing tests ────────────────────────────────────────────────


class TestArgParsing:
    def _parse(self, args: list[str]):
        parser = _build_parser()
        return parser.parse_args(args)

    def test_trace_id(self):
        args = self._parse(["--trace-id", "abc123"])
        assert args.trace_id == "abc123"
        assert args.last is None

    def test_last(self):
        args = self._parse(["--last", "30m"])
        assert args.last == "30m"
        assert args.trace_id is None

    def test_start_end(self):
        args = self._parse([
            "--start", "2024-01-01T00:00:00Z",
            "--end", "2024-01-01T01:00:00Z",
        ])
        assert args.start == "2024-01-01T00:00:00Z"
        assert args.end == "2024-01-01T01:00:00Z"

    def test_discover(self):
        args = self._parse(["--discover", "24h"])
        assert args.discover == "24h"

    def test_mutual_exclusivity(self):
        with pytest.raises(SystemExit):
            self._parse(["--trace-id", "abc", "--last", "1h"])

    def test_missing_required(self):
        with pytest.raises(SystemExit):
            self._parse([])

    def test_format_default(self):
        args = self._parse(["--trace-id", "abc"])
        assert args.fmt == "terminal"

    def test_format_json(self):
        args = self._parse(["--trace-id", "abc", "--format", "json"])
        assert args.fmt == "json"

    def test_limit_default(self):
        args = self._parse(["--trace-id", "abc"])
        assert args.limit == 100

    def test_verbose_default(self):
        args = self._parse(["--trace-id", "abc"])
        assert args.verbose is False

    def test_cost_flag(self):
        args = self._parse(["--trace-id", "abc", "--cost"])
        assert args.cost is True

    def test_accumulation_threshold(self):
        args = self._parse(["--trace-id", "abc", "--accumulation-threshold", "3.0"])
        assert args.accumulation_threshold == 3.0


# ── E2E CLI tests ─────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestCLIE2E:
    def test_trace_id_completes(self):
        from conftest import RECRUITER_TRACE_ID
        from token_analysis.cli import main

        # Should not raise
        main(["--trace-id", RECRUITER_TRACE_ID])

    def test_discover_completes(self):
        from token_analysis.cli import main

        # Should not raise
        main(["--discover", "24h"])
