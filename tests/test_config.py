import pytest

from rag.config import Settings


def test_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost:5432/db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    settings = Settings(_env_file=None)  # ignore any real .env during tests

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dims == 1536
    assert settings.openai_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings)


def test_settings_reject_bad_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "not-a-url")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with pytest.raises(ValueError):
        Settings(_env_file=None)