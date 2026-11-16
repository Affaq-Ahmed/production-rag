"""Atomic, strategy-specific document and chunk writes."""

import hashlib
from typing import Any, Literal

import psycopg

from rag.ingest.models import Chunk, RawDoc


def store(
    conn: psycopg.Connection[Any],
    doc: RawDoc,
    cleaned: str,
    chunks: list[Chunk],
    corpus_commit: str,
) -> Literal["inserted", "updated", "unchanged"]:
    """Store one strategy atomically; empty or mixed-strategy batches are invalid.

    An existing caller transaction owns the final commit when this creates a savepoint.
    """
    strategies = {chunk.strategy for chunk in chunks}
    if len(strategies) != 1:
        raise ValueError("chunks must be nonempty and contain exactly one strategy")
    strategy = chunks[0].strategy
    content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    with conn.transaction(), conn.cursor() as cursor:
        # Serialize writers for this path, including when the document does not exist yet.
        cursor.execute("select pg_advisory_xact_lock(hashtextextended(%s, 0))", (doc.path,))
        cursor.execute(
            "select id, content_hash from documents where path = %s for update", (doc.path,)
        )
        existing = cursor.fetchone()
        result: Literal["inserted", "updated", "unchanged"]
        if existing is not None:
            document_id = existing[0]
            # The document hash is shared by both strategies, so it can't say whether *this*
            # strategy's chunks are current. Each chunk records the hash it was built from.
            cursor.execute(
                "select count(*), count(*) filter (where source_hash = %s) "
                "from chunks where document_id = %s and strategy = %s",
                (content_hash, document_id, strategy),
            )
            counts = cursor.fetchone()
            assert counts is not None
            total, current = counts
            if total > 0 and current == total:
                return "unchanged"
            cursor.execute(
                "update documents set source_url = %s, title = %s, content_type = %s, "
                "content_hash = %s, corpus_commit = %s, ingested_at = now() where id = %s",
                (
                    doc.source_url,
                    doc.title,
                    doc.content_type,
                    content_hash,
                    corpus_commit,
                    document_id,
                ),
            )
            cursor.execute(
                "delete from chunks where document_id = %s and strategy = %s",
                (document_id, strategy),
            )
            result = "updated"
        else:
            cursor.execute(
                "insert into documents "
                "(path, source_url, title, content_type, content_hash, corpus_commit) "
                "values (%s, %s, %s, %s, %s, %s) returning id",
                (
                    doc.path,
                    doc.source_url,
                    doc.title,
                    doc.content_type,
                    content_hash,
                    corpus_commit,
                ),
            )
            row = cursor.fetchone()
            assert row is not None
            document_id = row[0]
            result = "inserted"
        cursor.executemany(
            "insert into chunks "
            "(document_id, strategy, ord, heading_path, text, token_count, source_hash) "
            "values (%s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    document_id,
                    c.strategy,
                    c.ord,
                    c.heading_path,
                    c.text,
                    c.token_count,
                    content_hash,
                )
                for c in chunks
            ],
        )
        return result
