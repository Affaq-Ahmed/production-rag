"""Apply pending migrations to DATABASE_URL (the logic is rag.migrate.apply_migrations)."""

from rag.db import connect
from rag.migrate import apply_migrations


def main() -> None:
    with connect() as conn:
        for name, applied in apply_migrations(conn):
            print(f"{'apply' if applied else 'skip':<5}  {name}")


if __name__ == "__main__":
    main()
