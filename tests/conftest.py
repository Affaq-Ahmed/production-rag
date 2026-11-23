"""Shared fixtures: the database integration tests' own database, migrated once per session."""

import pytest

from rag.config import IntegrationTestSettings
from rag.db import connect
from rag.migrate import apply_migrations
from tests.database import check_test_database


@pytest.fixture(scope="session")
def integration_database_url() -> str:
    """TEST_DATABASE_URL (from .env or the environment), migrated to the latest schema.

    Skips when it is unset. Fails, before connecting, when it names the DATABASE_URL database:
    integration tests write rows, and the dev database holds the evaluated index.
    """
    settings = IntegrationTestSettings()
    if settings.test_database_url is None:
        pytest.skip("Set TEST_DATABASE_URL to run database integration tests")
    url = str(settings.test_database_url)
    dev_url = None if settings.database_url is None else str(settings.database_url)
    try:
        check_test_database(url, dev_url)
    except ValueError as error:
        pytest.fail(str(error), pytrace=False)
    with connect(url) as conn:
        apply_migrations(conn)
    return url
