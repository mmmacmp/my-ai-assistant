from collections.abc import Generator
from typing import Any

import pytest

from my_ai_assistant_online import config as config_module
from my_ai_assistant_online.config import OnlineSettings, get_settings


CONFIG_ENVIRONMENT_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL_ID",
    "MONGODB_URI",
    "MONGODB_DATABASE_NAME",
    "MONGODB_COLLECTION_NAME",
    "MONGODB_RAG_COLLECTION_NAME",
    "RETRIEVER_K",
    "RETRIEVER_DEVICE",
    "RAG_RETRIEVER_K",
    "RAG_DEVICE",
    "AGENT_MAX_STEPS",
    "EVALUATION_DATASET_NAME",
    "EVALUATION_EXPERIMENT_NAME_PREFIX",
    "COMET_API_KEY",
    "OPIK_ENABLED",
    "OPIK_API_KEY",
    "OPIK_WORKSPACE",
    "OPIK_PROJECT_NAME",
)


@pytest.fixture(autouse=True)
def isolate_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
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
        "MONGODB_COLLECTION_NAME": "rag_test",
        "RETRIEVER_K": "7",
        "RETRIEVER_DEVICE": "cuda",
        "AGENT_MAX_STEPS": "4",
        "COMET_API_KEY": "comet-test-secret",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    settings = OnlineSettings(_env_file=None)

    assert settings.openai.api_key is not None
    assert settings.openai.api_key.get_secret_value() == "openai-test-secret"
    assert settings.openai.model_id == "gpt-test"
    assert settings.mongodb.uri is not None
    assert (
        settings.mongodb.uri.get_secret_value()
        == "mongodb://mongo-test-secret@localhost:27017"
    )
    assert settings.mongodb.database_name == "assistant_test"
    assert settings.mongodb.rag_collection_name == "rag_test"
    assert settings.rag.retriever_k == 7
    assert settings.rag.device == "cuda"
    assert settings.agent.max_steps == 4
    assert settings.comet.api_key is not None
    assert settings.comet.api_key.get_secret_value() == "comet-test-secret"


def test_secrets_are_masked_and_required_only_when_used() -> None:
    configured = OnlineSettings(
        _env_file=None,
        openai={"api_key": "openai-test-secret"},
        mongodb={"uri": "mongodb://mongo-test-secret@localhost:27017"},
    )
    missing = OnlineSettings(_env_file=None)

    assert "openai-test-secret" not in repr(configured)
    assert "mongo-test-secret" not in repr(configured)
    assert configured.openai.require_api_key() == "openai-test-secret"
    assert (
        configured.mongodb.require_uri()
        == "mongodb://mongo-test-secret@localhost:27017"
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        missing.openai.require_api_key()
    with pytest.raises(RuntimeError, match="MONGODB_URI"):
        missing.mongodb.require_uri()


def test_get_settings_is_lazy_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = object()

    def build_settings() -> object:
        calls.append({})
        return sentinel

    monkeypatch.setattr(config_module, "OnlineSettings", build_settings)

    assert calls == []
    assert get_settings() is sentinel
    assert get_settings() is sentinel
    assert len(calls) == 1


def test_settings_parse_canonical_grouped_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_RETRIEVER_K", "9")
    monkeypatch.setenv("RAG_DEVICE", "cuda:0")
    monkeypatch.setenv("MONGODB_RAG_COLLECTION_NAME", "canonical_rag")
    monkeypatch.setenv("OPIK_ENABLED", "true")
    monkeypatch.setenv("OPIK_API_KEY", "opik-test-secret")
    monkeypatch.setenv("OPIK_WORKSPACE", "test-workspace")
    monkeypatch.setenv("OPIK_PROJECT_NAME", "test-project")
    monkeypatch.setenv("EVALUATION_DATASET_NAME", "test-dataset")
    monkeypatch.setenv("EVALUATION_EXPERIMENT_NAME_PREFIX", "test-experiment")

    settings = OnlineSettings(_env_file=None)

    assert settings.rag.retriever_k == 9
    assert settings.rag.device == "cuda:0"
    assert settings.mongodb.rag_collection_name == "canonical_rag"
    assert settings.opik.enabled is True
    assert settings.opik.api_key is not None
    assert settings.opik.api_key.get_secret_value() == "opik-test-secret"
    assert settings.opik.workspace == "test-workspace"
    assert settings.opik.project_name == "test-project"
    assert settings.evaluation.dataset_name == "test-dataset"
    assert settings.evaluation.experiment_name_prefix == "test-experiment"
