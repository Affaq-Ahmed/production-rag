from functools import lru_cache

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: PostgresDsn
    openai_api_key: SecretStr
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = Field(default=1536, gt=0)
    corpus_dir: str = "data/k8s-docs/content/en/docs"


class IntegrationTestSettings(BaseSettings):
    """The URLs the database integration tests need, read apart from Settings so they run
    without an OpenAI key. tests/conftest.py refuses a TEST_DATABASE_URL that names the
    DATABASE_URL database.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    test_database_url: PostgresDsn | None = None
    database_url: PostgresDsn | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
