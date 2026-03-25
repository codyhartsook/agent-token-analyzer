"""
ClickHouse connection factory for token & context analysis.

Uses ``clickhouse-connect`` (HTTP port 8123, sync API).
Connection parameters come from environment variables or a local ``.env``
file.  The ``.env`` template ships with ``DB_PORT=9000`` (native protocol),
but ``clickhouse-connect`` speaks HTTP — so we default to **8123** when
``DB_PORT`` is ``9000``.

Environment variables
---------------------
DB_HOST       ClickHouse hostname          (default: localhost)
DB_PORT       ClickHouse HTTP port         (default: 8123)
DB_USERNAME   ClickHouse user              (default: admin)
DB_PASSWORD   ClickHouse password          (default: admin)
DB_DATABASE   ClickHouse database          (default: default)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect


# ── Lightweight .env parser (no python-dotenv dependency) ────────────────────

def _load_dotenv() -> None:
    """Read a ``.env`` file from the project root if it exists."""
    candidate = Path(__file__).resolve().parent.parent.parent / ".env"
    if not candidate.is_file():
        return
    with open(candidate) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


_load_dotenv()


_HTTP_PORT_DEFAULT = 8123
_NATIVE_PORT = 9000


# ── Connection factory ───────────────────────────────────────────────────────

def get_client() -> clickhouse_connect.driver.Client:
    """Return a ``clickhouse-connect`` client configured from env vars.

    Handles the common mismatch where ``.env`` has ``DB_PORT=9000``
    (ClickHouse native protocol) but ``clickhouse-connect`` requires the
    HTTP port (default 8123).
    """
    raw_port = int(os.environ.get("DB_PORT", str(_HTTP_PORT_DEFAULT)))
    # Auto-fix: if user specified the native protocol port, swap to HTTP
    port = _HTTP_PORT_DEFAULT if raw_port == _NATIVE_PORT else raw_port

    return clickhouse_connect.get_client(
        host=os.environ.get("DB_HOST", "localhost"),
        port=port,
        username=os.environ.get("DB_USERNAME", "admin"),
        password=os.environ.get("DB_PASSWORD", "admin"),
        database=os.environ.get("DB_DATABASE", "default"),
    )


# ── Timestamp helpers ────────────────────────────────────────────────────────

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def to_nano(iso_ts: str) -> int:
    """Convert an ISO-8601 UTC timestamp string to nanosecond UInt64.

    The ``otel_traces`` table stores timestamps as nanosecond integers.
    """
    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return int((dt - _EPOCH).total_seconds() * 1_000_000_000)


def from_nano(ns: int) -> str:
    """Convert nanosecond UInt64 back to ISO-8601 UTC string."""
    from datetime import timedelta

    dt = _EPOCH + timedelta(microseconds=ns / 1_000)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def relative_to_nano(spec: str) -> tuple[int, int]:
    """Parse ``--last 30m`` / ``--last 2h`` into ``(start_ns, end_ns)``.

    Supported suffixes: ``m`` (minutes), ``h`` (hours), ``d`` (days).
    """
    import re

    m = re.fullmatch(r"(\d+)\s*([mhd])", spec.strip())
    if not m:
        raise ValueError(
            f"Invalid relative time spec '{spec}'. Use e.g. '30m', '2h', '1d'."
        )
    value, unit = int(m.group(1)), m.group(2)
    multipliers = {"m": 60, "h": 3600, "d": 86400}
    delta_s = value * multipliers[unit]

    now = datetime.now(timezone.utc)
    end_ns = int((now - _EPOCH).total_seconds() * 1_000_000_000)
    start_ns = end_ns - delta_s * 1_000_000_000
    return start_ns, end_ns
