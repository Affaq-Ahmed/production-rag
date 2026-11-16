"""Apply migrations/*.sql in order, each once, each in its own transaction."""

from pathlib import Path

from rag.db import connect

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def main() -> None:
    with connect() as conn:
        conn.execute(
            "create table if not exists schema_migrations ("
            " name text primary key, applied_at timestamptz not null default now())"
        )
        conn.commit()
        applied = {row[0] for row in conn.execute("select name from schema_migrations")}

        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name in applied:
                print(f"skip   {path.name}")
                continue
            with conn.transaction():
                conn.execute(path.read_text())
                conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
            print(f"apply  {path.name}")


if __name__ == "__main__":
    main()