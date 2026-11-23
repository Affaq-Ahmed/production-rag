"""Guard for the integration tests' database: never the one DATABASE_URL points at."""

from psycopg.conninfo import conninfo_to_dict

# Spellings of this machine, so a URL can't slip past the guard by writing localhost differently.
# An empty host (libpq's Unix socket) counts too: refusing too much is the safe side.
_LOCAL_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1"})


def database_identity(url: str) -> tuple[str, str, str]:
    """(host, port, database name) with libpq's defaults filled in."""
    params = conninfo_to_dict(url)
    host = str(params.get("host") or "").lower()
    port = str(params.get("port") or "5432")
    # libpq connects to the database named after the user when the URL names none.
    dbname = str(params.get("dbname") or params.get("user") or "")
    return ("localhost" if host in _LOCAL_HOSTS else host, port, dbname)


def check_test_database(test_url: str, dev_url: str | None) -> None:
    """Raise if the test database is the dev database: the integration tests write to it.

    The message never includes either URL, which would carry the password.
    """
    if dev_url is not None and database_identity(test_url) == database_identity(dev_url):
        raise ValueError(
            "TEST_DATABASE_URL names the same database as DATABASE_URL. The integration tests "
            "write to it, so they need a database of their own (prag_test); refusing to run."
        )
