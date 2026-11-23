"""Apply migrations/*.sql in order, each once: for scripts/migrate.py and the test database."""

from pathlib import Path
from typing import Any

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def apply_migrations(
    conn: psycopg.Connection[Any], directory: Path = MIGRATIONS_DIR
) -> list[tuple[str, bool]]:
    """Apply each file not yet in schema_migrations. Returns (file name, applied) in order.

    On a connection with autocommit off, which connect() gives by default, every file runs in
    a savepoint of one transaction. The caller's commit makes the run permanent, and a failing
    file rolls back the whole run.
    """
    conn.execute(
        "create table if not exists schema_migrations ("
        " name text primary key, applied_at timestamptz not null default now())"
    )
    conn.commit()
    applied = {row[0] for row in conn.execute("select name from schema_migrations")}

    results: list[tuple[str, bool]] = []
    for path in sorted(directory.glob("*.sql")):
        if path.name in applied:
            results.append((path.name, False))
            continue
        with conn.transaction():
            conn.execute(path.read_text())
            conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
        results.append((path.name, True))
    return results
