from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 3000
    cors_origins: List[str] = ["*"]

    neo4j_uri: str = "neo4j+ssc://localhost"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    google_api_key_list: str = ""
    google_ai_model: str = "gemini-2.5-flash"
    google_ai_embeddings: str = "gemini/gemini-embedding-001"

    jina_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore
