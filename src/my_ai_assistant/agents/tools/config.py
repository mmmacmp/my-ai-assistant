from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Required API keys ---
    OPENAI_API_KEY: str
    COMET_API_KEY: str

    # --- Model config ---
    OPENAI_MODEL_ID: str = "gpt-4o-mini"

    # --- MongoDB ---
    MONGODB_URI: str
    MONGODB_DATABASE_NAME: str = "second_brain"
    MONGODB_COLLECTION_NAME: str = "rag_chunks"

    # --- Retriever defaults ---
    RETRIEVER_K: int = 5
    RETRIEVER_DEVICE: str = "cpu"

    # --- Agent ---
    AGENT_MAX_STEPS: int = 3


settings = Settings()  # type: ignore[call-arg]  # values come from .env