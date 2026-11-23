"""Connection helper: bounded connects, retries with jitter, server-side timeouts."""

import socket
import time
from collections.abc import Iterator
from typing import Any, cast

import psycopg
import pytest

from rag import db


@pytest.fixture
def silent_server() -> Iterator[str]:
    """A TCP listener that never answers: the handshake completes, the protocol never starts."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        port = server.getsockname()[1]
        yield f"postgresql://user:secret@127.0.0.1:{port}/db?sslmode=disable"


def test_stalled_connect_raises_instead_of_hanging(silent_server: str) -> None:
    started = time.monotonic()
    with pytest.raises(psycopg.OperationalError):
        db.connect(silent_server, attempts=1, connect_timeout=2)
    # libpq's minimum connect_timeout is 2 s; anything far beyond it means it hung.
    assert time.monotonic() - started < 10


def test_connect_error_does_not_leak_the_password(silent_server: str) -> None:
    with pytest.raises(psycopg.OperationalError) as error:
        db.connect(silent_server, attempts=1, connect_timeout=2)
    assert "secret" not in str(error.value)


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.committed = False
        self.autocommit = False

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        pass


def test_connect_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeConnection()
    outcomes: list[Any] = [psycopg.OperationalError("stall"), psycopg.OperationalError("stall")]
    sleeps: list[float] = []

    def flaky_connect(url: str, **kwargs: Any) -> FakeConnection:
        if outcomes:
            raise outcomes.pop(0)
        return fake

    monkeypatch.setattr("rag.db.psycopg.connect", flaky_connect)
    monkeypatch.setattr("rag.db.time.sleep", sleeps.append)
    assert db.connect("postgresql://x", attempts=3) is cast(object, fake)
    assert len(sleeps) == 2
    # Full jitter: each delay is somewhere in [0, base * 2**attempt].
    assert 0 <= sleeps[0] <= db.BASE_DELAY_S and 0 <= sleeps[1] <= db.BASE_DELAY_S * 2


def test_connect_gives_up_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def always_fails(url: str, **kwargs: Any) -> None:
        calls.append(url)
        raise psycopg.OperationalError("stall")

    monkeypatch.setattr("rag.db.psycopg.connect", always_fails)
    monkeypatch.setattr("rag.db.time.sleep", lambda _: None)
    with pytest.raises(psycopg.OperationalError):
        db.connect("postgresql://x", attempts=3)
    assert len(calls) == 3


def test_connect_sets_timeouts_on_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeConnection()
    seen: dict[str, Any] = {}

    def capture(url: str, **kwargs: Any) -> FakeConnection:
        seen.update(kwargs)
        return fake

    monkeypatch.setattr("rag.db.psycopg.connect", capture)
    db.connect("postgresql://x")
    assert seen["connect_timeout"] == 10
    assert seen["keepalives"] == 1
    assert seen["tcp_user_timeout"] == 60_000
    # The Supabase pooler ignores the `options` startup parameter, so SET them explicitly.
    assert "set statement_timeout = '60s'" in fake.executed
    assert "set idle_in_transaction_session_timeout = '120s'" in fake.executed
    assert fake.committed


def test_connect_sets_random_page_cost_so_stock_postgres_uses_hnsw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeConnection()
    monkeypatch.setattr("rag.db.psycopg.connect", lambda url, **kwargs: fake)
    db.connect("postgresql://x")
    # At the stock 4.0 the planner chose an exact sort over the partial HNSW index.
    assert "set random_page_cost = 1.1" in fake.executed
    assert fake.committed
