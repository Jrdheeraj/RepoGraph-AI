from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(alias="DATABASE_URL", min_length=1)
    secret_key: SecretStr = Field(alias="SECRET_KEY", min_length=1)
    environment: str = Field(alias="ENVIRONMENT", min_length=1)
    github_token: Optional[str] = Field(default=None, alias="GITHUB_TOKEN")

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
