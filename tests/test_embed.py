"""embed_batch / embed_pending with a fake client and a recording fake connection.

The data-protecting behaviours run in the default suite; the database test at the end is an
extra integration check that needs TEST_DATABASE_URL.
"""

import json
import os
import random
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import psycopg
import pytest
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage
from pgvector import Vector
from pgvector.psycopg import register_vector

from rag.embed import (
    PRICE_CHECKED,
    PRICE_PER_MILLION_TOKENS_USD,
    EmbedReport,
    embed_batch,
    embed_pending,
    embedding_input,
    merge_run,
    record_run,
)

DIMS = 3


def vector_for(text: str, dims: int = DIMS) -> list[float]:
    return [float(len(text))] + [0.0] * (dims - 1)


@dataclass
class FakeEmbeddings:
    """Shuffles items with a fixed seed to prove the caller sorts by index."""

    tokens_per_input: int = 7
    wrong_dims: int | None = None
    drop_last: bool = False
    calls: list[list[str]] = field(default_factory=list)

    def create(self, *, input: list[str], model: str, dimensions: int) -> CreateEmbeddingResponse:
        self.calls.append(input)
        data = [
            Embedding(
                embedding=vector_for(text, self.wrong_dims or dimensions),
                index=i,
                object="embedding",
            )
            for i, text in enumerate(input)
        ]
        if self.drop_last:
            data = data[:-1]
        random.Random(0).shuffle(data)
        tokens = self.tokens_per_input * len(input)
        return CreateEmbeddingResponse(
            data=data,
            model=model,
            object="list",
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
        )


@dataclass
class FakeClient:
    embeddings: FakeEmbeddings = field(default_factory=FakeEmbeddings)


@dataclass
class _Result:
    rows: list[tuple[Any, ...]]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


@dataclass
class RecordingConnection:
    """Answers the pending-chunks select with canned rows and records every statement."""

    rows: list[tuple[int, str, str]]
    statements: list[tuple[str, Any]] = field(default_factory=list)

    def execute(self, sql: str, params: Any = None) -> _Result:
        self.statements.append((sql, params))
        return _Result(list(self.rows) if sql.lstrip().startswith("select") else [])

    def executemany(self, sql: str, params_seq: Sequence[Any]) -> None:
        self.statements.append((sql, list(params_seq)))

    def transaction(self) -> AbstractContextManager[None]:
        return nullcontext()

    def cursor(self) -> AbstractContextManager["RecordingConnection"]:
        return nullcontext(self)

    def updates(self) -> list[tuple[str, Any]]:
        return [(sql, p) for sql, p in self.statements if sql.lstrip().startswith("update")]

    def as_conn(self) -> psycopg.Connection[Any]:
        # Duck-typed stand-in: embed_pending only uses execute/transaction/cursor/executemany.
        return cast("psycopg.Connection[Any]", self)


def chunk_rows(n: int) -> list[tuple[int, str, str]]:
    return [(100 + i, f"Page {i}", "x" * (i % 7 + 1)) for i in range(n)]


def test_embed_batch_pairs_vectors_by_index_not_response_order() -> None:
    batch = [(10, "a"), (11, "bbb"), (12, "cc"), (13, "dddd")]
    pairs, tokens = embed_batch(FakeClient(), "m", DIMS, batch)
    assert pairs == [(i, vector_for(text)) for i, text in batch]
    assert tokens == 28


def test_embed_batch_rejects_missing_items() -> None:
    client = FakeClient(FakeEmbeddings(drop_last=True))
    with pytest.raises(ValueError, match="indexes"):
        embed_batch(client, "m", DIMS, [(1, "a"), (2, "b")])


def test_embed_pending_writes_each_vector_to_its_own_chunk() -> None:
    rows = chunk_rows(5)
    conn = RecordingConnection(rows)
    embed_pending(conn.as_conn(), FakeClient(), "m", DIMS)
    [(_, params)] = conn.updates()
    expected = [(Vector(vector_for(embedding_input(h, t))), "m", i) for i, h, t in rows]
    assert params == expected


def test_embed_pending_wrong_dimensions_writes_nothing() -> None:
    conn = RecordingConnection(chunk_rows(3))
    client = FakeClient(FakeEmbeddings(wrong_dims=2))
    with pytest.raises(ValueError, match="dimensions"):
        embed_pending(conn.as_conn(), client, "m", DIMS)
    assert conn.updates() == []


def test_embed_pending_sums_tokens_across_batches() -> None:
    conn = RecordingConnection(chunk_rows(250))
    client = FakeClient()
    report = embed_pending(conn.as_conn(), client, "m", DIMS)
    assert [len(call) for call in client.embeddings.calls] == [100, 100, 50]
    assert len(conn.updates()) == 3
    assert (report.model, report.chunks, report.batches, report.tokens) == ("m", 250, 3, 1750)
    assert report.elapsed_s >= 0


def test_embed_pending_with_nothing_pending_costs_nothing() -> None:
    conn = RecordingConnection([])
    client = FakeClient()
    report = embed_pending(conn.as_conn(), client, "m", DIMS)
    assert (report.chunks, report.batches, report.tokens, report.cost_usd) == (0, 0, 0, 0.0)
    assert client.embeddings.calls == []
    assert conn.updates() == []


def test_report_cost_uses_billed_tokens() -> None:
    report = EmbedReport(model="m", chunks=1, batches=1, tokens=1_500_000, elapsed_s=0.0)
    assert report.cost_usd == pytest.approx(0.03)


def report(chunks: int, tokens: int, model: str = "m") -> EmbedReport:
    return EmbedReport(model=model, chunks=chunks, batches=1, tokens=tokens, elapsed_s=1.25)


def test_record_run_writes_guide_fields(tmp_path: Path) -> None:
    path = tmp_path / "results" / "embedding_run.json"
    assert record_run(path, report(20, 2_000), "fixed", 20)
    record = json.loads(path.read_text())
    assert record["model"] == "m"
    assert record["price_per_mtok"] == PRICE_PER_MILLION_TOKENS_USD
    assert record["price_checked"] == PRICE_CHECKED
    assert (record["chunks"], record["total_tokens"]) == (20, 2_000)
    assert record["cost_usd"] == pytest.approx(0.00004)
    assert record["elapsed_s"] == pytest.approx(1.2)
    assert [run["strategy"] for run in record["runs"]] == ["fixed"]


def test_record_run_totals_accumulate_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "embedding_run.json"
    record_run(path, report(2_272, 1_046_373), "fixed", None)
    record_run(path, report(4_773, 985_257), "structural", None)
    record = json.loads(path.read_text())
    assert (record["chunks"], record["total_tokens"]) == (7_045, 2_031_630)
    assert record["cost_usd"] == pytest.approx(0.040633)
    assert len(record["runs"]) == 2


def test_rerun_that_embeds_nothing_keeps_the_record(tmp_path: Path) -> None:
    path = tmp_path / "embedding_run.json"
    record_run(path, report(10, 500), "fixed", None)
    before = path.read_text()
    assert not record_run(path, report(0, 0), None, None)
    assert path.read_text() == before


def test_rerun_with_nothing_and_no_record_creates_no_file(tmp_path: Path) -> None:
    path = tmp_path / "embedding_run.json"
    assert not record_run(path, report(0, 0), None, None)
    assert not path.exists()


def test_merge_run_rejects_a_different_model() -> None:
    first = merge_run(None, report(1, 1, model="a"), None, None, "2026-11-12")
    with pytest.raises(ValueError, match="used b"):
        merge_run(first, report(1, 1, model="b"), None, None, "2026-11-12")


@pytest.fixture
def conn() -> Iterator[psycopg.Connection[Any]]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run database integration tests")
    with psycopg.connect(url) as connection:
        register_vector(connection)
        try:
            yield connection
        finally:
            connection.rollback()


def test_embed_pending_writes_and_resumes(conn: psycopg.Connection[Any]) -> None:
    row = conn.execute(
        "insert into documents (path, source_url, title, content_hash, corpus_commit) "
        "values (%s, 'u', 'T', 'h', 'c') returning id",
        (f"test/{uuid4()}.md",),
    ).fetchone()
    assert row is not None
    conn.execute(
        "insert into chunks (document_id, strategy, ord, heading_path, text, token_count, "
        "source_hash) "
        "select %s, 'fixed', g, 'T', 'text ' || g, 2, 'h' from generate_series(0, 2) g",
        (row[0],),
    )
    client = FakeClient()
    report = embed_pending(conn, client, "m", 1536, "fixed")
    again = embed_pending(conn, client, "m", 1536, "fixed")
    stored = conn.execute(
        "select embedding_model, embedding is not null from chunks where document_id = %s",
        (row[0],),
    ).fetchall()
    assert stored == [("m", True)] * 3
    assert client.embeddings.calls[0][0].startswith("T\n\n")
    assert report.chunks >= 3
    assert again.chunks == 0
