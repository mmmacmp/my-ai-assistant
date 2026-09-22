from collections.abc import Generator
from typing import Any

import pytest

from my_ai_assistant_offline import config as config_module
from my_ai_assistant_offline.config import OfflineSettings, get_settings


CONFIG_ENVIRONMENT_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL_ID",
    "MONGODB_URI",
    "MONGODB_DATABASE_NAME",
    "MONGODB_RAW_COLLECTION_NAME",
    "MONGODB_RAG_COLLECTION_NAME",
    "RETRIEVER_K",
    "RETRIEVER_DEVICE",
    "RAG_RETRIEVER_K",
    "RAG_DEVICE",
    "AGENT_MAX_STEPS",
    "COMET_API_KEY",
    "HUGGINGFACE_DEDICATED_ENDPOINT",
    "HUGGINGFACE_ACCESS_TOKEN",
)


@pytest.fixture(autouse=True)
def isolate_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Keep the cache and the developer's real environment out of unit tests."""

    get_settings.cache_clear()
    for name in CONFIG_ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield
    get_settings.cache_clear()


def test_settings_parse_legacy_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "OPENAI_API_KEY": "openai-test-secret",
        "OPENAI_MODEL_ID": "gpt-test",
        "MONGODB_URI": "mongodb://mongo-test-secret@localhost:27017",
        "MONGODB_DATABASE_NAME": "assistant_test",
        "MONGODB_RAW_COLLECTION_NAME": "raw_test",
        "MONGODB_RAG_COLLECTION_NAME": "rag_test",
        "RETRIEVER_K": "7",
        "RETRIEVER_DEVICE": "cuda",
        "AGENT_MAX_STEPS": "4",
        "COMET_API_KEY": "comet-test-secret",
        "HUGGINGFACE_DEDICATED_ENDPOINT": "https://example.invalid/v1",
        "HUGGINGFACE_ACCESS_TOKEN": "huggingface-test-secret",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    settings = OfflineSettings(_env_file=None)

    assert settings.openai.api_key is not None
    assert settings.openai.api_key.get_secret_value() == "openai-test-secret"
    assert settings.openai.model_id == "gpt-test"
    assert settings.mongodb.uri is not None
    assert (
        settings.mongodb.uri.get_secret_value()
        == "mongodb://mongo-test-secret@localhost:27017"
    )
    assert settings.mongodb.database_name == "assistant_test"
    assert settings.mongodb.raw_collection_name == "raw_test"
    assert settings.mongodb.rag_collection_name == "rag_test"
    assert settings.rag.retriever_k == 7
    assert settings.rag.device == "cuda"
    assert settings.agent.max_steps == 4
    assert settings.comet.api_key is not None
    assert settings.comet.api_key.get_secret_value() == "comet-test-secret"
    assert settings.huggingface.dedicated_endpoint == "https://example.invalid/v1"
    assert settings.huggingface.access_token is not None
    assert (
        settings.huggingface.access_token.get_secret_value()
        == "huggingface-test-secret"
    )


def test_optional_secrets_are_masked_and_required_on_demand() -> None:
    settings = OfflineSettings(
        _env_file=None,
        openai={"api_key": "openai-test-secret"},
        mongodb={"uri": "mongodb://mongo-test-secret@localhost:27017"},
    )

    rendered_settings = repr(settings)

    assert "openai-test-secret" not in rendered_settings
    assert "mongo-test-secret" not in rendered_settings
    assert settings.openai.require_api_key() == "openai-test-secret"
    assert (
        settings.mongodb.require_uri() == "mongodb://mongo-test-secret@localhost:27017"
    )


def test_required_secret_helpers_fail_only_when_used() -> None:
    settings = OfflineSettings(_env_file=None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        settings.openai.require_api_key()

    with pytest.raises(RuntimeError, match="MONGODB_URI"):
        settings.mongodb.require_uri()


def test_settings_parse_canonical_grouped_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_RETRIEVER_K", "9")
    monkeypatch.setenv("RAG_DEVICE", "cuda:0")

    settings = OfflineSettings(_env_file=None)

    assert settings.rag.retriever_k == 9
    assert settings.rag.device == "cuda:0"


def test_get_settings_is_lazy_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = object()

    def build_settings() -> object:
        calls.append({})
        return sentinel

    monkeypatch.setattr(config_module, "OfflineSettings", build_settings)

    assert calls == []
    assert get_settings() is sentinel
    assert get_settings() is sentinel
    assert len(calls) == 1
