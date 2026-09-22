"""Typed configuration for the offline application.

Only executable boundaries, such as ZenML steps or command-line entrypoints,
should load these settings. Application and infrastructure modules receive the
specific values they need through function or constructor arguments.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ConfigurationGroup(BaseModel):
    """Shared behavior for immutable nested configuration groups."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class OpenAISettings(_ConfigurationGroup):
    """OpenAI credentials and model defaults."""

    api_key: SecretStr | None = None
    model_id: str = "gpt-4o-mini"

    def require_api_key(self) -> str:
        """Return the API key or fail at the boundary that needs OpenAI."""

        if self.api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return self.api_key.get_secret_value()


class MongoDBSettings(_ConfigurationGroup):
    """MongoDB connection and collection names."""

    uri: SecretStr | None = None
    database_name: str = "my_ai_assistant"
    raw_collection_name: str = "raw_documents"
    rag_collection_name: str = Field(
        default="rag_chunks",
        validation_alias=AliasChoices(
            "RAG_COLLECTION_NAME",
            "COLLECTION_NAME",
        ),
    )

    def require_uri(self) -> str:
        """Return the MongoDB URI or fail at the boundary that needs MongoDB."""

        if self.uri is None:
            raise RuntimeError("MONGODB_URI is not configured")
        return self.uri.get_secret_value()


class RagSettings(_ConfigurationGroup):
    """Defaults shared by RAG ingestion and retrieval."""

    retriever_k: int = Field(
        default=5,
        gt=0,
        validation_alias=AliasChoices("RETRIEVER_K", "K"),
    )
    device: str = "cpu"


class AgentSettings(_ConfigurationGroup):
    """General agent execution defaults."""

    max_steps: int = Field(default=3, gt=0)


class CometSettings(_ConfigurationGroup):
    """Optional observability integration."""

    api_key: SecretStr | None = None


class HuggingFaceSettings(_ConfigurationGroup):
    """Optional Hugging Face dedicated-endpoint configuration."""

    dedicated_endpoint: str | None = None
    access_token: SecretStr | None = None


class OfflineSettings(BaseSettings):
    """Configuration required by the offline application.

    A single underscore is intentional. With ``env_nested_max_split=1``, the
    existing flat names map naturally to nested groups, for example
    ``OPENAI_API_KEY`` -> ``openai.api_key`` and
    ``MONGODB_DATABASE_NAME`` -> ``mongodb.database_name``.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        env_nested_max_split=1,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    mongodb: MongoDBSettings = Field(default_factory=MongoDBSettings)
    rag: RagSettings = Field(
        default_factory=RagSettings,
        validation_alias=AliasChoices("RAG", "RETRIEVER"),
    )
    agent: AgentSettings = Field(default_factory=AgentSettings)
    comet: CometSettings = Field(default_factory=CometSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)


@lru_cache(maxsize=1)
def get_settings() -> OfflineSettings:
    """Load and validate one immutable settings object per process."""

    return OfflineSettings()


# Backward-compatible class name for callers that imported ``Settings``.
Settings = OfflineSettings


__all__ = [
    "AgentSettings",
    "CometSettings",
    "HuggingFaceSettings",
    "MongoDBSettings",
    "OfflineSettings",
    "OpenAISettings",
    "RagSettings",
    "Settings",
    "get_settings",
]
