"""store(): scripted-fake tests run by default; database tests need TEST_DATABASE_URL."""

import hashlib
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

import psycopg
import pytest

from rag.ingest.models import Chunk, RawDoc
from rag.ingest.store import store


@pytest.fixture
def conn(integration_database_url: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(integration_database_url) as connection:
        try:
            yield connection
        finally:
            connection.rollback()


@pytest.fixture
def doc() -> RawDoc:
    return RawDoc(f"test/{uuid4()}.md", "https://example.com/", "Test", None, "raw")


def batch(text: str = "cleaned") -> list[Chunk]:
    return [Chunk("fixed", 0, "Test", text, 2)]


def test_insert_and_unchanged(conn: psycopg.Connection[Any], doc: RawDoc) -> None:
    # Outer transaction keeps test data disposable; store uses atomic savepoints.
    conn.execute("select 1")
    assert store(conn, doc, "cleaned", batch(), "commit") == "inserted"
    before = conn.execute("select * from documents where path = %s", (doc.path,)).fetchone()
    assert before is not None
    assert before[5] == hashlib.sha256(b"cleaned").hexdigest()
    rows = conn.execute("select * from chunks where document_id = %s", (before[0],)).fetchall()
    assert len(rows) == 1
    assert rows[0][7:9] == (None, None)
    assert store(conn, doc, "cleaned", batch("ignored"), "other") == "unchanged"
    assert conn.execute("select * from documents where path = %s", (doc.path,)).fetchone() == before
    assert (
        conn.execute("select * from chunks where document_id = %s", (before[0],)).fetchall() == rows
    )


def test_update_and_add_strategy(conn: psycopg.Connection[Any], doc: RawDoc) -> None:
    conn.execute("select 1")
    store(conn, doc, "cleaned", batch(), "commit")
    structural = [Chunk("structural", 0, "Test", "cleaned", 2)]
    assert store(conn, doc, "cleaned", structural, "commit") == "updated"
    assert store(conn, doc, "changed", batch("changed"), "next") == "updated"
    rows = conn.execute(
        "select strategy, text from chunks where document_id = "
        "(select id from documents where path = %s) order by strategy",
        (doc.path,),
    ).fetchall()
    assert rows == [("fixed", "changed"), ("structural", "cleaned")]


def test_failure_rolls_back_replacement(conn: psycopg.Connection[Any], doc: RawDoc) -> None:
    conn.execute("select 1")
    store(conn, doc, "cleaned", batch(), "commit")
    with pytest.raises(psycopg.errors.CheckViolation):
        store(conn, doc, "changed", [*batch("changed"), Chunk("fixed", 1, "", "", 0)], "next")
    assert store(conn, doc, "cleaned", batch(), "commit") == "unchanged"
    rows = conn.execute(
        "select text from chunks where document_id = (select id from documents where path = %s)",
        (doc.path,),
    ).fetchall()
    assert rows == [("cleaned",)]


def test_invalid_batch(conn: psycopg.Connection[Any], doc: RawDoc) -> None:
    for chunks in [[], [*batch(), Chunk("structural", 1, "Test", "text", 1)]]:
        with pytest.raises(ValueError, match="exactly one strategy"):
            store(conn, doc, "cleaned", chunks, "commit")


# --- Default-suite tests: a scripted fake connection, no database needed. ---------------------


@dataclass
class ScriptedConnection:
    """Answers store()'s reads from a scripted document/chunk state and records every write."""

    document: tuple[int, str] | None
    chunk_hashes: list[str | None]
    statements: list[tuple[str, Any]] = field(default_factory=list)
    _row: tuple[Any, ...] | None = None

    def execute(self, query: str, params: Any = None) -> "ScriptedConnection":
        self.statements.append((query, params))
        self._row = None
        if query.startswith("select id, content_hash from documents"):
            self._row = self.document
        elif query.startswith("select count(*)"):
            current = params[0]
            self._row = (len(self.chunk_hashes), self.chunk_hashes.count(current))
        elif query.startswith("select 1 from chunks"):
            self._row = (1,) if self.chunk_hashes else None
        elif query.startswith("insert into documents"):
            self._row = (99,)
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row

    def executemany(self, query: str, params_seq: Any) -> None:
        self.statements.append((query, list(params_seq)))

    def transaction(self) -> AbstractContextManager[None]:
        return nullcontext()

    def cursor(self) -> AbstractContextManager["ScriptedConnection"]:
        return nullcontext(self)

    def writes(self, verb: str) -> list[tuple[str, Any]]:
        return [(q, p) for q, p in self.statements if q.startswith(verb)]

    def as_conn(self) -> psycopg.Connection[Any]:
        # Duck-typed stand-in: store() only uses execute/executemany/transaction/cursor.
        return cast("psycopg.Connection[Any]", self)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


DOC = RawDoc("concepts/x.md", "https://example.com/", "X", None, "raw")
STRUCTURAL = [Chunk("structural", 0, "X", "new text", 2)]


def test_other_strategy_is_rechunked_after_one_strategy_updated_the_document() -> None:
    # Strategy A (fixed) already re-ran: the document hash is the new one. Strategy B
    # (structural) still has chunks from the old cleaned text, so B must be replaced.
    conn = ScriptedConnection(document=(7, sha("new text")), chunk_hashes=[sha("old")] * 3)
    assert store(conn.as_conn(), DOC, "new text", STRUCTURAL, "commit") == "updated"
    assert len(conn.writes("delete from chunks")) == 1
    [(_, rows)] = conn.writes("insert into chunks")
    assert [row[-1] for row in rows] == [sha("new text")]


def test_strategy_is_unchanged_only_when_all_its_chunks_carry_the_current_hash() -> None:
    conn = ScriptedConnection(document=(7, sha("new text")), chunk_hashes=[sha("new text")] * 3)
    assert store(conn.as_conn(), DOC, "new text", STRUCTURAL, "commit") == "unchanged"
    assert conn.writes("delete") == []
    assert conn.writes("insert") == []


def test_partially_current_strategy_is_replaced() -> None:
    hashes = [sha("new text"), sha("new text"), None]
    conn = ScriptedConnection(document=(7, sha("new text")), chunk_hashes=hashes)
    assert store(conn.as_conn(), DOC, "new text", STRUCTURAL, "commit") == "updated"
    assert len(conn.writes("delete from chunks")) == 1


def test_new_document_chunks_carry_the_source_hash() -> None:
    conn = ScriptedConnection(document=None, chunk_hashes=[])
    assert store(conn.as_conn(), DOC, "new text", STRUCTURAL, "commit") == "inserted"
    [(_, rows)] = conn.writes("insert into chunks")
    assert [row[-1] for row in rows] == [sha("new text")]


def test_rechunks_other_strategy_after_cleaning_change(
    conn: psycopg.Connection[Any], doc: RawDoc
) -> None:
    """Integration version of the first scripted test (needs TEST_DATABASE_URL)."""
    conn.execute("select 1")
    store(conn, doc, "cleaned", batch(), "commit")
    store(conn, doc, "cleaned", [Chunk("structural", 0, "Test", "cleaned", 2)], "commit")
    assert store(conn, doc, "changed", batch("changed"), "commit") == "updated"
    structural = [Chunk("structural", 0, "Test", "changed", 2)]
    assert store(conn, doc, "changed", structural, "commit") == "updated"
    assert store(conn, doc, "changed", structural, "commit") == "unchanged"
