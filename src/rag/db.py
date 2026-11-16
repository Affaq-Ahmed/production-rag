"""The one way to open a database connection: bounded timeouts and retried connects.

About one connection in three to the Supabase session pooler stalled after the TCP handshake
and never reached Postgres (see NOTES.md, Step 4.9 follow-up). Without timeouts a stalled
connection waits forever; these make it raise, and the retry turns that into a short delay.
Ingest and embed runs are resumable, so a run that dies mid-way is simply rerun.
"""

import random
import time
from typing import Any

import psycopg

from rag.config import get_settings

ATTEMPTS = 3
BASE_DELAY_S = 1.0

_CONNECT_KWARGS: dict[str, Any] = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
    # Linux only; libpq ignores it where TCP_USER_TIMEOUT doesn't exist (e.g. macOS).
    "tcp_user_timeout": 60_000,
}

# Applied with SET after connecting: the Supabase pooler silently drops the `options`
# startup parameter (statement_timeout stayed at the role default, 2min). The session pooler
# keeps one server session per client connection, so session-level SET sticks.
_SESSION_SETTINGS = (
    "set statement_timeout = '60s'",
    "set idle_in_transaction_session_timeout = '120s'",
)


def connect(
    url: str | None = None,
    *,
    autocommit: bool = False,
    attempts: int = ATTEMPTS,
    connect_timeout: int = 10,
) -> psycopg.Connection[Any]:
    """Connect with timeouts, retrying failed connects with full-jitter backoff."""
    target = url if url is not None else str(get_settings().database_url)
    for attempt in range(attempts):
        try:
            conn = psycopg.connect(
                target, autocommit=autocommit, connect_timeout=connect_timeout, **_CONNECT_KWARGS
            )
        except psycopg.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(random.uniform(0, BASE_DELAY_S * 2**attempt))
            continue
        try:
            for statement in _SESSION_SETTINGS:
                conn.execute(statement)
            if not conn.autocommit:
                conn.commit()
        except BaseException:
            conn.close()
            raise
        return conn
    raise AssertionError("unreachable: attempts must be positive")
