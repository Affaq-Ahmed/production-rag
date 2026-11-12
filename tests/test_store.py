"""Database integration tests; set TEST_DATABASE_URL to a migrated database."""

import hashlib
import os
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import psycopg
import pytest

from rag.ingest.models import Chunk, RawDoc
from rag.ingest.store import store


@pytest.fixture
def conn() -> Iterator[psycopg.Connection[Any]]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run database integration tests")
    with psycopg.connect(url) as connection:
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
