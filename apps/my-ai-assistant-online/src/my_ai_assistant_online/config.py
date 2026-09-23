"""Typed configuration for the online application."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ConfigurationGroup(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class OpenAISettings(_ConfigurationGroup):
    api_key: SecretStr | None = None
    model_id: str = "gpt-4o-mini"

    def require_api_key(self) -> str:
        if self.api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return self.api_key.get_secret_value()


class MongoDBSettings(_ConfigurationGroup):
    uri: SecretStr | None = None
    database_name: str = "my_ai_assistant"
    rag_collection_name: str = Field(
        default="rag_chunks",
        validation_alias=AliasChoices(
            "RAG_COLLECTION_NAME",
            "COLLECTION_NAME",
        ),
    )

    def require_uri(self) -> str:
        if self.uri is None:
            raise RuntimeError("MONGODB_URI is not configured")
        return self.uri.get_secret_value()


class RagSettings(_ConfigurationGroup):
    retriever_k: int = Field(
        default=5,
        gt=0,
        validation_alias=AliasChoices("RETRIEVER_K", "K"),
    )
    device: str = "cpu"


class AgentSettings(_ConfigurationGroup):
    max_steps: int = Field(default=3, gt=0)


class EvaluationSettings(_ConfigurationGroup):
    dataset_name: str = "my-ai-assistant-agentic-rag"
    experiment_name_prefix: str = "agentic-rag"


class CometSettings(_ConfigurationGroup):
    api_key: SecretStr | None = None


class OpikSettings(_ConfigurationGroup):
    enabled: bool = False
    api_key: SecretStr | None = None
    workspace: str | None = None
    project_name: str = "my-ai-assistant-online"

    def require_api_key(self) -> str:
        if self.api_key is None:
            raise RuntimeError("OPIK_API_KEY is not configured")
        return self.api_key.get_secret_value()

    def require_workspace(self) -> str:
        if not self.workspace:
            raise RuntimeError("OPIK_WORKSPACE is not configured")
        return self.workspace


class OnlineSettings(BaseSettings):
    """Online configuration loaded only at the composition boundary."""

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
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    comet: CometSettings = Field(default_factory=CometSettings)
    opik: OpikSettings = Field(default_factory=OpikSettings)


@lru_cache(maxsize=1)
def get_settings() -> OnlineSettings:
    return OnlineSettings()


Settings = OnlineSettings


__all__ = [
    "AgentSettings",
    "CometSettings",
    "EvaluationSettings",
    "MongoDBSettings",
    "OpikSettings",
    "OnlineSettings",
    "OpenAISettings",
    "RagSettings",
    "Settings",
    "get_settings",
]
