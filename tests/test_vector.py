"""Query embedding cache and the model check, with a fake client and a fake connection."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest

from rag.retrieve.vector import (
    EmbeddingModelMismatchError,
    QueryEmbeddingCache,
    check_embedding_model,
    embed_query,
    query_cache_key,
)
from tests.test_embed import DIMS, FakeClient, vector_for

MODEL = "text-embedding-3-small"


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
