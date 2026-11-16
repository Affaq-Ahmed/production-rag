"""Eyeball check: embed one query and list the nearest chunks' heading paths (Step 4.9).

uv run python scripts/smoke_search.py ["my pod keeps restarting"] [--strategy structural] [--k 5]

Not an evaluation: this only shows whether results look related. It embeds the query with
`query_input` (the decided convention) and, for comparison, as `embedding_input("", query)`.
"""

import argparse
from typing import Any, Literal

import psycopg
from openai import OpenAI
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql

from rag.config import get_settings
from rag.db import connect
from rag.embed import embed_batch, embedding_input, query_input

Strategy = Literal["fixed", "structural"]


def nearest(
    conn: psycopg.Connection[Any], strategy: Strategy, vector: Vector, k: int
) -> list[tuple[Any, ...]]:
    query = sql.SQL(
        "select c.id, round((c.embedding <=> %s)::numeric, 4), d.path, c.heading_path "
        "from chunks c join documents d on d.id = c.document_id "
        "where c.strategy = {strategy} order by c.embedding <=> %s limit {k}"
    ).format(strategy=sql.Literal(strategy), k=sql.Literal(k))
    return conn.execute(query, (vector, vector)).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="my pod keeps restarting")
    parser.add_argument("--strategy", choices=["fixed", "structural"], default="structural")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    forms = {
        'embedding_input("", query)': embedding_input("", args.query),
        "query_input(query) [decided]": query_input(args.query),
    }
    pairs, tokens = embed_batch(
        client, settings.embedding_model, settings.embedding_dims, list(enumerate(forms.values()))
    )
    vectors = [Vector(vector) for _, vector in pairs]
    with connect(autocommit=True) as conn:
        register_vector(conn)
        for label, vector in zip(forms, vectors, strict=True):
            print(f"\n== {args.query!r} as {label}  ({args.strategy}, top {args.k})")
            for chunk_id, distance, path, heading in nearest(conn, args.strategy, vector, args.k):
                print(f"  {distance}  #{chunk_id:<5} {heading}")
                print(f"  {'':6}  {'':6} {path}")
        same = conn.execute("select 1 - (%s <=> %s)", (vectors[0], vectors[1])).fetchone()
    print(f"\ncosine similarity between the two query forms: {same[0] if same else '?':.4f}")
    print(f"tokens billed: {tokens}")


if __name__ == "__main__":
    main()
