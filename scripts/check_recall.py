"""Measure HNSW recall@k against exact search, per strategy.

uv run python scripts/check_recall.py [--samples 20] [--k 10] [--seed 0] [--ef-search N]

Each sampled chunk's own embedding is the query. The indexed search is the normal query with the
strategy as a SQL literal (so the partial index matches); the exact search is the same query with
index scans disabled for that transaction. Recall = |indexed ∩ exact| / k.
"""

import argparse
import random
import statistics
import sys
from typing import Any, Literal

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql

from rag.db import connect

Strategy = Literal["fixed", "structural"]
STRATEGIES: tuple[Strategy, ...] = ("fixed", "structural")


def neighbours_query(strategy: Strategy, k: int) -> sql.Composed:
    return sql.SQL(
        "select id from chunks where strategy = {strategy} order by embedding <=> %s limit {k}"
    ).format(strategy=sql.Literal(strategy), k=sql.Literal(k))


def search(
    conn: psycopg.Connection[Any],
    query: sql.Composed,
    vector: Vector,
    *,
    exact: bool,
    ef_search: int | None,
    explain: bool = False,
) -> list[Any]:
    """Run one search in its own transaction so `set local` settings end with it."""
    # Inside an open transaction, conn.transaction() is only a savepoint, and `set local`
    # outlives a released savepoint: enable_indexscan=off would leak into later searches.
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        raise RuntimeError("search needs an autocommit connection with no open transaction")
    with conn.transaction():
        if exact:
            conn.execute("set local enable_indexscan = off")
        if ef_search is not None:
            conn.execute(sql.SQL("set local hnsw.ef_search = {}").format(sql.Literal(ef_search)))
        statement = sql.SQL("explain ") + query if explain else query
        return [row[0] for row in conn.execute(statement, (vector,)).fetchall()]


def check_plans(conn: psycopg.Connection[Any], strategy: Strategy, k: int, vector: Vector) -> None:
    """Fail loudly if the indexed search isn't using the HNSW index, or the exact one is."""
    index = f"chunks_embedding_{strategy}_hnsw"
    query = neighbours_query(strategy, k)
    indexed = "\n".join(search(conn, query, vector, exact=False, ef_search=None, explain=True))
    exact = "\n".join(search(conn, query, vector, exact=True, ef_search=None, explain=True))
    if f"Index Scan using {index}" not in indexed:
        sys.exit(f"{strategy}: indexed search does not use {index}:\n{indexed}")
    if "hnsw" in exact:
        sys.exit(f"{strategy}: exact search still uses an HNSW index:\n{exact}")
    print(f"{strategy}: plan ok, indexed -> Index Scan using {index}; exact -> no index scan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ef-search", type=int, help="hnsw.ef_search (pgvector default 40)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    results: dict[Strategy, list[float]] = {}
    with connect(autocommit=True) as conn:
        register_vector(conn)
        print(f"k={args.k}  samples={args.samples}  seed={args.seed}")
        for strategy in STRATEGIES:
            ids = [
                row[0]
                for row in conn.execute(
                    "select id from chunks where strategy = %s and embedding is not null "
                    "order by id",
                    (strategy,),
                )
            ]
            sample = rng.sample(ids, args.samples)
            vectors: dict[int, Vector] = dict(
                conn.execute(
                    "select id, embedding from chunks where id = any(%s)", (sample,)
                ).fetchall()
            )
            check_plans(conn, strategy, args.k, vectors[sample[0]])
            # hnsw.* settings exist only once a vector query has loaded pgvector in this session.
            row = conn.execute("select current_setting('hnsw.ef_search', true)").fetchone()
            default_ef = row[0] if row else None
            ef = args.ef_search if args.ef_search is not None else f"{default_ef} (default)"
            print(f"{strategy}: hnsw.ef_search = {ef}")
            query = neighbours_query(strategy, args.k)
            recalls: list[float] = []
            for chunk_id in sample:
                vector = vectors[chunk_id]
                indexed = search(conn, query, vector, exact=False, ef_search=args.ef_search)
                exact = search(conn, query, vector, exact=True, ef_search=None)
                recall = len(set(indexed) & set(exact)) / args.k
                recalls.append(recall)
                if recall < 1.0:
                    missed = sorted(set(exact) - set(indexed))
                    print(f"  chunk {chunk_id}: recall {recall:.2f}, missed {missed}")
            results[strategy] = recalls

    print()
    print(f"{'strategy':<11} {'mean recall@' + str(args.k):>15} {'min':>6} {'perfect':>8}")
    for strategy, recalls in results.items():
        perfect = sum(r == 1.0 for r in recalls)
        print(
            f"{strategy:<11} {statistics.mean(recalls):>15.3f} {min(recalls):>6.2f}"
            f" {perfect:>4}/{len(recalls)}"
        )


if __name__ == "__main__":
    main()
