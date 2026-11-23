"""Vector search: query cache, model check, SQL, ef_search and timings, with fakes."""

import json
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql

from rag.ingest.models import Strategy
from rag.retrieve.models import Hit
from rag.retrieve.vector import (
    EF_SEARCH_DEFAULT,
    MAX_K,
    EmbeddingModelMismatchError,
    QueryEmbeddingCache,
    build_vector_sql,
    check_embedding_model,
    ef_search_for,
    embed_query,
    nearest_chunks,
    query_cache_key,
    search_vector,
)
from tests.test_embed import DIMS, FakeClient, vector_for

MODEL = "text-embedding-3-small"
COLUMN_DIMS = 1536  # chunks.embedding is vector(1536), whatever DIMS the fakes use


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "query_embeddings.json"


def test_miss_embeds_the_bare_query_and_saves_it(cache_path: Path) -> None:
    client = FakeClient()
    vector = embed_query(client, QueryEmbeddingCache(cache_path), MODEL, DIMS, "  Why? Pods.  ")

    assert client.embeddings.calls == [["Why? Pods."]]
    assert vector == vector_for("Why? Pods.")
    reloaded = QueryEmbeddingCache(cache_path)
    assert reloaded.get(query_cache_key(MODEL, "Why? Pods.")) == vector


def test_hit_from_disk_makes_no_api_call_and_returns_the_same_vector(cache_path: Path) -> None:
    first = embed_query(FakeClient(), QueryEmbeddingCache(cache_path), MODEL, DIMS, "q one")
    client = FakeClient()
    again = embed_query(client, QueryEmbeddingCache(cache_path), MODEL, DIMS, " q one\n")

    assert client.embeddings.calls == []
    assert again == first


def test_vector_from_another_model_is_never_reused(cache_path: Path) -> None:
    client = FakeClient()
    cache = QueryEmbeddingCache(cache_path)
    embed_query(client, cache, "model-a", DIMS, "same question")
    embed_query(client, cache, "model-b", DIMS, "same question")

    assert client.embeddings.calls == [["same question"], ["same question"]]
    assert query_cache_key("model-a", "q") != query_cache_key("model-b", "q")


def test_cached_vector_with_wrong_dimensions_raises_without_calling_api(
    cache_path: Path,
) -> None:
    QueryEmbeddingCache(cache_path).put(query_cache_key(MODEL, "q"), [0.1, 0.2])
    client = FakeClient()

    with pytest.raises(ValueError, match="2 dims, not 3"):
        embed_query(client, QueryEmbeddingCache(cache_path), MODEL, DIMS, "q")
    assert client.embeddings.calls == []


def test_empty_query_raises_without_calling_api(cache_path: Path) -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="empty"):
        embed_query(client, QueryEmbeddingCache(cache_path), MODEL, DIMS, " \n ")
    assert client.embeddings.calls == []


def test_cache_file_is_sorted_one_entry_per_line_and_floats_round_trip(
    cache_path: Path,
) -> None:
    cache = QueryEmbeddingCache(cache_path)
    awkward = [0.1 + 0.2, 1 / 3, -2.5e-08]
    cache.put(query_cache_key(MODEL, "b question"), awkward)
    cache.put(query_cache_key(MODEL, "a question"), [1.0, 2.0, 3.0])

    lines = cache_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "{" and lines[-1] == "}"
    assert [line.split('"')[1] for line in lines[1:-1]] == [
        f"{MODEL}\\u0000a question",
        f"{MODEL}\\u0000b question",
    ]
    assert json.loads(cache_path.read_text())[query_cache_key(MODEL, "b question")] == awkward
    assert not cache_path.with_suffix(".tmp").exists()


def test_cache_file_that_is_not_an_object_is_refused(cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        QueryEmbeddingCache(cache_path)


@dataclass
class _Rows:
    rows: list[tuple[Any, ...]]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


@dataclass
class ModelsConnection:
    rows: list[tuple[Any, ...]]
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def execute(self, query: str, params: tuple[Any, ...]) -> _Rows:
        self.calls.append((query, params))
        return _Rows(self.rows)


def _check(rows: list[tuple[Any, ...]]) -> ModelsConnection:
    conn = ModelsConnection(rows)
    check_embedding_model(cast(psycopg.Connection[Any], conn), "fixed", MODEL)
    return conn


def test_model_check_passes_on_exactly_the_configured_model() -> None:
    conn = _check([(MODEL,)])
    assert conn.calls == [
        ("select distinct embedding_model from chunks where strategy = %s", ("fixed",))
    ]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([("text-embedding-3-large",)], "\\['text-embedding-3-large'\\]"),
        ([(MODEL,), ("text-embedding-ada-002",)], "ada-002"),
        ([(MODEL,), (None,)], "some chunks are not embedded"),
        ([], "\\[\\]"),
    ],
)
def test_model_check_refuses_anything_but_one_matching_model(
    rows: list[tuple[Any, ...]], message: str
) -> None:
    with pytest.raises(EmbeddingModelMismatchError, match=message):
        _check(rows)


EXPECTED_SQL = """
select c.id, d.path, c.heading_path, c.text,
       c.embedding <=> %(q)s as distance
from chunks c
join documents d on d.id = c.document_id
where c.strategy = '{strategy}'
order by c.embedding <=> %(q)s
limit %(k)s
"""


@pytest.mark.parametrize("strategy", ["fixed", "structural"])
def test_vector_sql_matches_the_index_rules(strategy: Strategy) -> None:
    rendered = build_vector_sql(strategy).as_string(None)
    assert rendered == EXPECTED_SQL.strip().format(strategy=strategy)


def test_vector_sql_strategy_is_a_literal_and_only_q_and_k_are_parameters() -> None:
    rendered = build_vector_sql("fixed").as_string(None)
    assert "'fixed'" in rendered
    assert re.findall(r"%\(?\w*\)?s", rendered) == ["%(q)s", "%(q)s", "%(k)s"]
    assert "<->" not in rendered and "<#>" not in rendered
    assert " desc" not in rendered


def test_vector_sql_refuses_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown strategy 'structual'"):
        build_vector_sql(cast(Strategy, "structual"))


@dataclass
class _Info:
    transaction_status: psycopg.pq.TransactionStatus = psycopg.pq.TransactionStatus.IDLE


@dataclass
class SearchConnection:
    """Records transaction boundaries and statements; returns `rows` for the vector query."""

    rows: list[tuple[Any, ...]] = field(default_factory=list)
    info: _Info = field(default_factory=_Info)
    calls: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.calls.append(("begin", None))
        yield
        self.calls.append(("commit", None))

    def execute(self, query: sql.Composable, params: dict[str, Any] | None = None) -> _Rows:
        self.calls.append((query.as_string(None), params))
        return _Rows(self.rows if params else [])


def _search(conn: SearchConnection, k: int) -> list[Hit]:
    return nearest_chunks(cast(psycopg.Connection[Any], conn), [0.5, 0.25, -1.0], "fixed", k)


def test_ef_search_is_set_locally_in_the_same_transaction_just_before_the_query() -> None:
    conn = SearchConnection([(i, f"p/{i}.md", "H", "t", i / 100) for i in range(50)])
    hits = _search(conn, 50)

    assert [statement for statement, _ in conn.calls] == [
        "begin",
        "set local hnsw.ef_search = 100",
        build_vector_sql("fixed").as_string(None),
        "commit",
    ]
    params = conn.calls[2][1]
    assert params is not None and params["k"] == 50
    assert isinstance(params["q"], Vector) and params["q"].to_list() == [0.5, 0.25, -1.0]
    assert len(hits) == 50


def test_hits_are_ranked_from_one_with_distance_as_score() -> None:
    conn = SearchConnection([(7, "a.md", "A", "x", 0.12), (3, "b.md", "B", "y", 0.34)])
    assert _search(conn, 2) == [
        Hit(
            chunk_id=7, path="a.md", heading_path="A", text="x", score=0.12, rank=1, source="vector"
        ),
        Hit(
            chunk_id=3, path="b.md", heading_path="B", text="y", score=0.34, rank=2, source="vector"
        ),
    ]


@pytest.mark.parametrize(("k", "ef_search"), [(1, 40), (20, 40), (21, 42), (50, 100), (500, 1000)])
def test_ef_search_is_at_least_the_default_and_twice_k(k: int, ef_search: int) -> None:
    assert ef_search_for(k) == ef_search


@pytest.mark.parametrize("k", [0, -1, MAX_K + 1])
def test_k_outside_what_hnsw_can_return_is_refused_before_any_sql(k: int) -> None:
    conn = SearchConnection()
    with pytest.raises(ValueError, match=f"between 1 and {MAX_K}"):
        _search(conn, k)
    assert conn.calls == []


def test_open_transaction_is_refused_so_set_local_cannot_leak() -> None:
    conn = SearchConnection(info=_Info(psycopg.pq.TransactionStatus.INTRANS))
    with pytest.raises(RuntimeError, match="idle connection"):
        _search(conn, 10)
    assert conn.calls == []


def test_search_vector_times_the_embedding_and_the_database_separately(
    cache_path: Path,
) -> None:
    conn = SearchConnection([(7, "a.md", "A", "x", 0.12)])
    client = FakeClient()

    result = search_vector(
        cast(psycopg.Connection[Any], conn),
        client,
        "  Which probe?  ",
        "fixed",
        cache=QueryEmbeddingCache(cache_path),
        model=MODEL,
        dims=DIMS,
        clock=iter([10.0, 10.5, 12.0]).__next__,
    )

    assert client.embeddings.calls == [["Which probe?"]]
    params = conn.calls[2][1]
    assert params is not None and params["k"] == 10
    assert params["q"].to_list() == vector_for("Which probe?")
    assert result.hits == [
        Hit(
            chunk_id=7, path="a.md", heading_path="A", text="x", score=0.12, rank=1, source="vector"
        )
    ]
    assert (result.embed_s, result.db_s) == (0.5, 1.5)


def test_search_vector_with_a_warm_cache_makes_no_api_call(cache_path: Path) -> None:
    cache = QueryEmbeddingCache(cache_path)
    cache.put(query_cache_key(MODEL, "Which probe?"), [1.0, 2.0, 3.0])
    conn, client = SearchConnection(), FakeClient()

    search_vector(
        cast(psycopg.Connection[Any], conn),
        client,
        "Which probe?",
        "structural",
        5,
        cache=cache,
        model=MODEL,
        dims=DIMS,
    )

    assert client.embeddings.calls == []
    params = conn.calls[2][1]
    assert params is not None and params["q"].to_list() == [1.0, 2.0, 3.0] and params["k"] == 5


@pytest.mark.parametrize(
    ("strategy", "k", "status", "message"),
    [
        ("structual", 10, psycopg.pq.TransactionStatus.IDLE, "unknown strategy"),
        ("fixed", 0, psycopg.pq.TransactionStatus.IDLE, "between 1 and"),
        ("fixed", MAX_K + 1, psycopg.pq.TransactionStatus.IDLE, "between 1 and"),
        ("fixed", 10, psycopg.pq.TransactionStatus.INTRANS, "idle connection"),
    ],
)
def test_bad_search_is_refused_before_any_api_call_or_sql(
    cache_path: Path,
    strategy: str,
    k: int,
    status: psycopg.pq.TransactionStatus,
    message: str,
) -> None:
    conn, client = SearchConnection(info=_Info(status)), FakeClient()
    with pytest.raises((ValueError, RuntimeError), match=message):
        search_vector(
            cast(psycopg.Connection[Any], conn),
            client,
            "a fresh question",
            cast(Strategy, strategy),
            k,
            cache=QueryEmbeddingCache(cache_path),
            model=MODEL,
            dims=DIMS,
        )
    assert client.embeddings.calls == [] and conn.calls == []


@pytest.fixture
def live_conn(integration_database_url: str) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(integration_database_url, autocommit=True) as connection:
        register_vector(connection)
        yield connection


@pytest.fixture
def seeded_vectors(live_conn: psycopg.Connection[Any]) -> Iterator[list[list[float]]]:
    """100 committed fixed chunks with random vectors, deleted afterwards.

    Committed rather than rolled back, because nearest_chunks refuses a connection with an open
    transaction. 100 rows is enough to tell ef_search's 40 from k = 50.
    """
    rng = random.Random(0)
    # Random in 8 dimensions, zero in the rest: 1,536 random dimensions make every pair nearly
    # orthogonal, and HNSW's graph walk then stops short of ef_search rows (39 of 40 here).
    vectors = [[rng.gauss(0, 1) for _ in range(8)] + [0.0] * (COLUMN_DIMS - 8) for _ in range(100)]
    row = live_conn.execute(
        "insert into documents (path, source_url, title, content_hash, corpus_commit) "
        "values (%s, 'u', 'T', 'h', 'c') returning id",
        (f"test/{uuid4()}.md",),
    ).fetchone()
    assert row is not None
    try:
        with live_conn.transaction(), live_conn.cursor() as cur:
            cur.executemany(
                "insert into chunks (document_id, strategy, ord, heading_path, text,"
                " token_count, source_hash, embedding, embedding_model)"
                " values (%s, 'fixed', %s, 'T', 'seed', 1, 'h', %s, 'm')",
                [(row[0], ord_, Vector(vector)) for ord_, vector in enumerate(vectors)],
            )
        yield vectors
    finally:
        # Scoped to the seeded document; its chunks go with it (on delete cascade).
        live_conn.execute("delete from documents where id = %s", (row[0],))


def test_k_50_returns_50_rows_where_default_ef_search_returns_40(
    live_conn: psycopg.Connection[Any], seeded_vectors: list[list[float]]
) -> None:
    vector = seeded_vectors[0]
    # Leave the planner only the HNSW index. On 100 rows it would rather read and sort them
    # all, which returns every row asked for and hides the trap this test is about.
    live_conn.execute("set enable_seqscan = off")
    live_conn.execute("set enable_sort = off")

    default = live_conn.execute(
        build_vector_sql("fixed"), {"q": Vector(vector), "k": 50}
    ).fetchall()
    hits = nearest_chunks(live_conn, vector, "fixed", 50)
    after = live_conn.execute("select current_setting('hnsw.ef_search')").fetchone()

    # The trap: HNSW returns at most ef_search rows whatever the limit. An exact sort would
    # return all 50, so this also proves the index was used.
    assert len(default) <= EF_SEARCH_DEFAULT
    assert len(hits) == 50
    assert [hit.rank for hit in hits] == list(range(1, 51))
    assert after == (str(EF_SEARCH_DEFAULT),)
