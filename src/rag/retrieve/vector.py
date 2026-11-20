"""Vector search: cached query embeddings and the embedding-model check."""

import json
from pathlib import Path
from typing import Any

import psycopg

from rag.embed import EmbeddingClient, embed_batch, query_input
from rag.ingest.models import Strategy

QUERY_CACHE_PATH = Path(__file__).resolve().parents[3] / "eval" / "cache" / "query_embeddings.json"


class EmbeddingModelMismatchError(RuntimeError):
    """Query and chunk vectors would come from different models, or some chunks have none."""


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
