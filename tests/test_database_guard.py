"""The integration tests' database guard: the dev database is refused however it is spelled."""

import pytest

from tests.database import check_test_database, database_identity

DEV = "postgresql://postgres:postgres@localhost:5433/postgres"
TEST = "postgresql://postgres:postgres@localhost:5433/prag_test"


def test_a_separate_test_database_is_accepted() -> None:
    check_test_database(TEST, DEV)


def test_no_dev_database_url_is_accepted() -> None:
    check_test_database(TEST, None)


@pytest.mark.parametrize(
    "test_url",
    [
        DEV,
        "postgresql://postgres:postgres@127.0.0.1:5433/postgres",
        "postgresql://other:pw@LOCALHOST:5433/postgres?sslmode=disable",
    ],
    ids=["same-url", "127.0.0.1-for-localhost", "other-user-case-and-options"],
)
def test_the_dev_database_is_refused(test_url: str) -> None:
    with pytest.raises(ValueError, match="same database as DATABASE_URL"):
        check_test_database(test_url, DEV)


def test_another_port_is_another_database() -> None:
    check_test_database("postgresql://postgres:postgres@localhost:5432/postgres", DEV)


def test_libpq_defaults_fill_in_port_and_database() -> None:
    # No port means 5432; no database name means the user's name.
    assert database_identity("postgresql://u@db.example.com") == database_identity(
        "postgresql://u@db.example.com:5432/u"
    )


def test_refusal_does_not_leak_the_password() -> None:
    url = "postgresql://postgres:hunter2@localhost:5433/postgres"
    with pytest.raises(ValueError) as error:
        check_test_database(url, url)
    assert "hunter2" not in str(error.value)
