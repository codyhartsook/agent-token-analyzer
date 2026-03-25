"""Tests for client.py — timestamp helpers, get_client port autofix."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from token_analysis.client import from_nano, relative_to_nano, to_nano

import pytest


# ── to_nano tests ──────────────────────────────────────────────────────────


class TestToNano:
    def test_basic(self):
        # 2024-01-01T00:00:00Z
        ns = to_nano("2024-01-01T00:00:00Z")
        assert ns > 0
        assert isinstance(ns, int)

    def test_epoch(self):
        ns = to_nano("1970-01-01T00:00:00Z")
        assert ns == 0

    def test_specific_value(self):
        # 1 second after epoch
        ns = to_nano("1970-01-01T00:00:01Z")
        assert ns == 1_000_000_000

    def test_with_offset(self):
        ns = to_nano("2024-01-01T00:00:00+00:00")
        assert ns > 0


# ── from_nano tests ────────────────────────────────────────────────────────


class TestFromNano:
    def test_basic(self):
        result = from_nano(1_700_000_000_000_000_000)
        assert isinstance(result, str)
        assert "Z" in result

    def test_epoch(self):
        result = from_nano(0)
        assert result == "1970-01-01T00:00:00.000Z"

    def test_roundtrip(self):
        original = "2024-06-15T10:30:00.000Z"
        ns = to_nano(original)
        back = from_nano(ns)
        assert back == original


# ── relative_to_nano tests ────────────────────────────────────────────────


class TestRelativeToNano:
    def test_minutes(self):
        start_ns, end_ns = relative_to_nano("30m")
        assert end_ns > start_ns
        delta_s = (end_ns - start_ns) / 1_000_000_000
        assert abs(delta_s - 30 * 60) < 1  # Allow 1s tolerance

    def test_hours(self):
        start_ns, end_ns = relative_to_nano("2h")
        delta_s = (end_ns - start_ns) / 1_000_000_000
        assert abs(delta_s - 2 * 3600) < 1

    def test_days(self):
        start_ns, end_ns = relative_to_nano("1d")
        delta_s = (end_ns - start_ns) / 1_000_000_000
        assert abs(delta_s - 86400) < 1

    def test_with_whitespace(self):
        start_ns, end_ns = relative_to_nano("  30 m  ")
        delta_s = (end_ns - start_ns) / 1_000_000_000
        assert abs(delta_s - 30 * 60) < 1

    def test_invalid_suffix(self):
        with pytest.raises(ValueError, match="Invalid relative time spec"):
            relative_to_nano("30x")

    def test_empty(self):
        with pytest.raises(ValueError, match="Invalid relative time spec"):
            relative_to_nano("")

    def test_no_number(self):
        with pytest.raises(ValueError, match="Invalid relative time spec"):
            relative_to_nano("h")


# ── get_client tests ──────────────────────────────────────────────────────


class TestGetClient:
    @patch.dict(os.environ, {"DB_PORT": "9000"}, clear=False)
    @patch("token_analysis.client.clickhouse_connect")
    def test_port_autofix_9000_to_8123(self, mock_cc):
        from token_analysis.client import get_client

        mock_cc.get_client.return_value = MagicMock()
        get_client()
        call_kwargs = mock_cc.get_client.call_args[1]
        assert call_kwargs["port"] == 8123

    @patch.dict(os.environ, {"DB_PORT": "8123"}, clear=False)
    @patch("token_analysis.client.clickhouse_connect")
    def test_port_8123_unchanged(self, mock_cc):
        from token_analysis.client import get_client

        mock_cc.get_client.return_value = MagicMock()
        get_client()
        call_kwargs = mock_cc.get_client.call_args[1]
        assert call_kwargs["port"] == 8123

    @patch.dict(os.environ, {"DB_PORT": "9123"}, clear=False)
    @patch("token_analysis.client.clickhouse_connect")
    def test_custom_port_preserved(self, mock_cc):
        from token_analysis.client import get_client

        mock_cc.get_client.return_value = MagicMock()
        get_client()
        call_kwargs = mock_cc.get_client.call_args[1]
        assert call_kwargs["port"] == 9123

    @patch.dict(os.environ, {
        "DB_HOST": "myhost",
        "DB_USERNAME": "myuser",
        "DB_PASSWORD": "mypass",
        "DB_DATABASE": "mydb",
        "DB_PORT": "8123",
    }, clear=False)
    @patch("token_analysis.client.clickhouse_connect")
    def test_env_vars_passed(self, mock_cc):
        from token_analysis.client import get_client

        mock_cc.get_client.return_value = MagicMock()
        get_client()
        call_kwargs = mock_cc.get_client.call_args[1]
        assert call_kwargs["host"] == "myhost"
        assert call_kwargs["username"] == "myuser"
        assert call_kwargs["password"] == "mypass"
        assert call_kwargs["database"] == "mydb"
