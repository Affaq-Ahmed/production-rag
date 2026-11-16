"""Batched, resumable chunk embedding: uv run python -m rag.embed [--strategy S] [--limit N]."""

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import psycopg
import tiktoken
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from pgvector import Vector
from pgvector.psycopg import register_vector

from rag.config import get_settings
from rag.db import connect

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Checked on OpenAI's pricing page; see NOTES.md "Step 4.3". Update both together.
PRICE_PER_MILLION_TOKENS_USD = 0.02
PRICE_CHECKED = "2026-11-12"
REPORT_PATH = Path(__file__).resolve().parents[2] / "eval" / "results" / "embedding_run.json"

Strategy = Literal["fixed", "structural"]


class _Embeddings(Protocol):
    def create(
        self, *, input: list[str], model: str, dimensions: int
    ) -> CreateEmbeddingResponse: ...


class EmbeddingClient(Protocol):
    """The slice of `openai.OpenAI` this module uses, so tests can pass a fake."""

    @property
    def embeddings(self) -> _Embeddings: ...


@dataclass(frozen=True)
class EmbedReport:
    model: str
    chunks: int
    batches: int
    tokens: int
    elapsed_s: float

    @property
    def cost_usd(self) -> float:
        return self.tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD


def embedding_input(heading_path: str, text: str) -> str:
    """Prefix chunk text with its heading path, separated by a blank line."""
    return f"{heading_path}\n\n{text}"


def query_input(query: str) -> str:
    """Queries are embedded bare: a heading path is chunk context, and a query has none.

    Deliberately not embedding_input("", query), whose "\\n\\n" prefix would be an accident of
    that function rather than a decision. See NOTES.md "Query embedding decision".
    """
    return query.strip()


def batches(
    items: list[tuple[int, str]], max_inputs: int = 100, max_tokens: int = 50_000
) -> Iterator[list[tuple[int, str]]]:
    """Batch in input order; an oversized item is yielded alone unchanged."""
    if max_inputs <= 0 or max_tokens <= 0:
        raise ValueError("max_inputs and max_tokens must be positive")
    batch: list[tuple[int, str]] = []
    token_count = 0
    for item in items:
        size = len(_ENCODING.encode(item[1], disallowed_special=()))
        if batch and (len(batch) >= max_inputs or token_count + size > max_tokens):
            yield batch
            batch = []
            token_count = 0
        if size > max_tokens:
            yield [item]
        else:
            batch.append(item)
            token_count += size
    if batch:
        yield batch


def embed_batch(
    client: EmbeddingClient, model: str, dims: int, batch: list[tuple[int, str]]
) -> tuple[list[tuple[int, list[float]]], int]:
    """Embed one batch; return (chunk_id, vector) pairs and the billed token count."""
    response = client.embeddings.create(
        input=[text for _, text in batch], model=model, dimensions=dims
    )
    # The API does not promise response order; each item's index points back into the input.
    data = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in data] != list(range(len(batch))):
        raise ValueError(f"expected indexes 0..{len(batch) - 1}, got {len(data)} items")
    for item in data:
        if len(item.embedding) != dims:
            raise ValueError(f"expected {dims} dimensions, got {len(item.embedding)}")
    pairs = [(chunk_id, item.embedding) for (chunk_id, _), item in zip(batch, data, strict=True)]
    return pairs, response.usage.total_tokens


def pending_chunks(
    conn: psycopg.Connection[Any], strategy: Strategy | None, limit: int | None
) -> list[tuple[int, str]]:
    """Chunks without an embedding, as (id, embedding_input), in id order."""
    rows = conn.execute(
        "select id, heading_path, text from chunks where embedding is null "
        "and (%(strategy)s::text is null or strategy = %(strategy)s) "
        "order by id limit %(limit)s",
        {"strategy": strategy, "limit": limit},
    ).fetchall()
    return [(chunk_id, embedding_input(path, text)) for chunk_id, path, text in rows]


def write_embeddings(
    conn: psycopg.Connection[Any], model: str, pairs: list[tuple[int, list[float]]]
) -> None:
    """Store one batch atomically, so an interrupted run resumes at the next batch."""
    with conn.transaction(), conn.cursor() as cursor:
        cursor.executemany(
            "update chunks set embedding = %s, embedding_model = %s "
            "where id = %s and embedding is null",
            [(Vector(vector), model, chunk_id) for chunk_id, vector in pairs],
        )


def embed_pending(
    conn: psycopg.Connection[Any],
    client: EmbeddingClient,
    model: str,
    dims: int,
    strategy: Strategy | None = None,
    *,
    limit: int | None = None,
) -> EmbedReport:
    started = time.monotonic()
    items = pending_chunks(conn, strategy, limit)
    groups = list(batches(items))
    chunks = tokens = 0
    for number, batch in enumerate(groups, start=1):
        pairs, used = embed_batch(client, model, dims, batch)
        write_embeddings(conn, model, pairs)
        chunks += len(pairs)
        tokens += used
        print(f"batch {number}/{len(groups)}: {len(pairs)} chunks, {used:,} tokens", flush=True)
    return EmbedReport(
        model=model,
        chunks=chunks,
        batches=len(groups),
        tokens=tokens,
        elapsed_s=time.monotonic() - started,
    )


def merge_run(
    existing: dict[str, Any] | None,
    report: EmbedReport,
    strategy: Strategy | None,
    limit: int | None,
    date: str,
) -> dict[str, Any]:
    """Add one run to the cumulative cost record; top-level fields are totals over all runs."""
    runs: list[dict[str, Any]] = list(existing["runs"]) if existing else []
    if existing and existing["model"] != report.model:
        raise ValueError(f"report is for {existing['model']}, this run used {report.model}")
    runs.append(
        {
            "date": date,
            "strategy": strategy,
            "limit": limit,
            "chunks": report.chunks,
            "batches": report.batches,
            "total_tokens": report.tokens,
            "cost_usd": round(report.cost_usd, 6),
            "elapsed_s": round(report.elapsed_s, 1),
        }
    )
    total_tokens = sum(run["total_tokens"] for run in runs)
    return {
        "model": report.model,
        "price_per_mtok": PRICE_PER_MILLION_TOKENS_USD,
        "price_checked": PRICE_CHECKED,
        "chunks": sum(run["chunks"] for run in runs),
        "total_tokens": total_tokens,
        "cost_usd": round(total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD, 6),
        "elapsed_s": round(sum(run["elapsed_s"] for run in runs), 1),
        "runs": runs,
    }


def record_run(
    path: Path, report: EmbedReport, strategy: Strategy | None, limit: int | None
) -> bool:
    """Append a run that embedded something; a no-op rerun must not erase the real cost."""
    if report.chunks == 0:
        return False
    existing = json.loads(path.read_text()) if path.exists() else None
    today = datetime.now(UTC).date().isoformat()
    record = merge_run(existing, report, strategy, limit, today)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return True


def positive_limit(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed", "structural"])
    parser.add_argument("--limit", type=positive_limit)
    args = parser.parse_args()
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    with connect(autocommit=True) as conn:
        register_vector(conn)
        report = embed_pending(
            conn,
            client,
            settings.embedding_model,
            settings.embedding_dims,
            args.strategy,
            limit=args.limit,
        )
    print(f"model:    {report.model} ({settings.embedding_dims} dims)")
    print(f"chunks:   {report.chunks:,} in {report.batches} batches")
    print(f"tokens:   {report.tokens:,} (billed, from response.usage)")
    print(f"cost:     ${report.cost_usd:.6f} at ${PRICE_PER_MILLION_TOKENS_USD}/1M tokens")
    print(f"elapsed:  {report.elapsed_s:.1f}s")
    if record_run(REPORT_PATH, report, args.strategy, args.limit):
        print(f"report:   appended to {REPORT_PATH.relative_to(REPORT_PATH.parents[2])}")
    else:
        print("report:   nothing embedded; embedding_run.json left unchanged")


if __name__ == "__main__":
    main()
