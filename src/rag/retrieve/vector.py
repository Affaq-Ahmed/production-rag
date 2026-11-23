"""Vector search: cached query embeddings, the embedding-model check, the SQL and the search."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import psycopg
from pgvector import Vector
from psycopg import sql

from rag.embed import EmbeddingClient, embed_batch, query_input
from rag.ingest.models import Strategy
from rag.retrieve.models import Hit

QUERY_CACHE_PATH = Path(__file__).resolve().parents[3] / "eval" / "cache" / "query_embeddings.json"
# HNSW returns at most hnsw.ef_search rows whatever the limit. 40 is pgvector's default, and
# 1000 its maximum, so max(40, 2k) works up to k = 500.
EF_SEARCH_DEFAULT = 40
EF_SEARCH_MAX = 1000
MAX_K = EF_SEARCH_MAX // 2


class EmbeddingModelMismatchError(RuntimeError):
    """Query and chunk vectors would come from different models, or some chunks have none."""


@dataclass(frozen=True)
class VectorSearch:
    """The hits, and the two halves of the time taken, in seconds.

    Timed apart because they behave differently: embed_s is near zero on a cache hit and an API
    round trip on a miss, while db_s is what the reranker will later be compared against.
    """

    hits: list[Hit]
    embed_s: float
    db_s: float


def query_cache_key(model: str, query: str) -> str:
    # The model is in the key for the same reason chunks store embedding_model: a vector from
    # another model must never be reused. query_input here means callers can't skip it.
    return f"{model}\x00{query_input(query)}"


class QueryEmbeddingCache:
    """Query vectors in a committed JSON file: free, identical and offline after the first run."""

    def __init__(self, path: Path = QUERY_CACHE_PATH) -> None:
        self.path = path
        # json.loads returns Any; the shape is checked per vector in embed_query.
        loaded: Any = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must hold a JSON object of key -> vector")
        self._vectors: dict[str, list[float]] = loaded

    def get(self, key: str) -> list[float] | None:
        return self._vectors.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        self._vectors[key] = vector
        self._save()

    def _save(self) -> None:
        # One sorted entry per line, so a new question is a one-line diff. json.dumps writes
        # floats with repr, which reads back bit-identical.
        entries = [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in sorted(self._vectors.items())]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".tmp")
        partial.write_text("{\n" + ",\n".join(entries) + "\n}\n", encoding="utf-8")
        # Atomic swap: an interrupted run never leaves a half-written cache.
        partial.replace(self.path)


def embed_query(
    client: EmbeddingClient, cache: QueryEmbeddingCache, model: str, dims: int, query: str
) -> list[float]:
    """The bare query's vector, from the cache, or embedded once and cached."""
    text = query_input(query)
    if not text:
        raise ValueError("query is empty")
    key = query_cache_key(model, query)
    cached = cache.get(key)
    if cached is not None:
        if len(cached) != dims:
            raise ValueError(f"cached vector for {text!r} has {len(cached)} dims, not {dims}")
        return cached
    pairs, _tokens = embed_batch(client, model, dims, [(0, text)])
    vector = pairs[0][1]
    cache.put(key, vector)
    return vector


def check_embedding_model(conn: psycopg.Connection[Any], strategy: Strategy, model: str) -> None:
    """Raise unless every chunk of this strategy is embedded with `model`. Once per session.

    Comparing a query vector from one model with chunk vectors from another returns confident
    nonsense, never an error. Chunks with no embedding show up as None, and are refused too:
    vector search can never return them.
    """
    rows = conn.execute(
        "select distinct embedding_model from chunks where strategy = %s", (strategy,)
    ).fetchall()
    found = sorted((row[0] for row in rows), key=str)
    if found != [model]:
        raise EmbeddingModelMismatchError(
            f"{strategy} chunks have embedding_model {found}, queries use {model!r}"
            + (" (None: some chunks are not embedded)" if None in found else "")
        )


def build_vector_sql(strategy: Strategy) -> sql.Composed:
    """Nearest chunks of one strategy by cosine distance. Parameters: %(q)s Vector, %(k)s int.

    Each rule keeps the strategy's partial HNSW index in use; breaking one makes Postgres scan
    and sort every row instead, with no error:
    - the strategy is a literal, so the planner can match the index's `where strategy = ...`
      (safe as a literal because it is checked against the two-value Strategy type);
    - `<=>` only: the index is vector_cosine_ops, which serves neither `<->` nor `<#>`;
    - order by the distance expression itself, ascending;
    - always a limit: HNSW answers "nearest k", nothing else.
    """
    _check_strategy(strategy)
    return sql.SQL(
        "select c.id, d.path, c.heading_path, c.text,\n"
        "       c.embedding <=> %(q)s as distance\n"
        "from chunks c\n"
        "join documents d on d.id = c.document_id\n"
        "where c.strategy = {strategy}\n"
        "order by c.embedding <=> %(q)s\n"
        "limit %(k)s"
    ).format(strategy=sql.Literal(strategy))


def ef_search_for(k: int) -> int:
    """Candidate list size for a top-k search: never below the default, 2k for headroom."""
    return max(EF_SEARCH_DEFAULT, 2 * k)


def nearest_chunks(
    conn: psycopg.Connection[Any], query_vector: list[float], strategy: Strategy, k: int
) -> list[Hit]:
    """The database half: the k nearest chunks of one strategy, ranked from 1.

    Needs register_vector(conn) once after connecting, and an idle connection (autocommit, or
    no open transaction).
    """
    _check_search(conn, strategy, k)
    with conn.transaction():
        conn.execute(sql.SQL("set local hnsw.ef_search = {}").format(sql.Literal(ef_search_for(k))))
        rows = conn.execute(
            build_vector_sql(strategy), {"q": Vector(query_vector), "k": k}
        ).fetchall()
    return [
        Hit(
            chunk_id=chunk_id,
            path=path,
            heading_path=heading_path,
            text=text,
            score=distance,
            rank=rank,
            source="vector",
        )
        for rank, (chunk_id, path, heading_path, text, distance) in enumerate(rows, start=1)
    ]


def search_vector(
    conn: psycopg.Connection[Any],
    client: EmbeddingClient,
    query: str,
    strategy: Strategy,
    k: int = 10,
    *,
    cache: QueryEmbeddingCache,
    model: str,
    dims: int,
    clock: Callable[[], float] = time.perf_counter,
) -> VectorSearch:
    """Embed the query (cache first, API on a miss), then fetch its k nearest chunks.

    Call check_embedding_model(conn, strategy, model) once per session before searching; it is
    not repeated here, so it is never inside the timings. Pass one cache for the whole run:
    loading the file is setup, not embedding time.
    """
    # Checked before embedding, so a bad argument never costs an API call.
    _check_search(conn, strategy, k)
    started = clock()
    vector = embed_query(client, cache, model, dims, query)
    embedded = clock()
    hits = nearest_chunks(conn, vector, strategy, k)
    finished = clock()
    return VectorSearch(hits, embed_s=embedded - started, db_s=finished - embedded)


def _check_strategy(strategy: Strategy) -> None:
    # mypy enforces Strategy for typed callers; this catches a CLI string or a typo at runtime,
    # which would otherwise match no index and silently return zero rows.
    if strategy not in get_args(Strategy):
        raise ValueError(f"unknown strategy {strategy!r}")


def _check_search(conn: psycopg.Connection[Any], strategy: Strategy, k: int) -> None:
    _check_strategy(strategy)
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}, got {k}")
    # Inside an open transaction, conn.transaction() is only a savepoint, and `set local`
    # outlives a released savepoint (NOTES.md, Step 4.8): ef_search would leak into later work.
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        raise RuntimeError("vector search needs an idle connection with no open transaction")
