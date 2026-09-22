from unittest.mock import MagicMock

import pytest

from my_ai_assistant_offline.application.rag import embeddings as embeddings_module


def test_openai_embeddings_require_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_embeddings = MagicMock()
    monkeypatch.setattr(
        embeddings_module,
        "OpenAIEmbeddings",
        openai_embeddings,
    )

    with pytest.raises(RuntimeError, match="OpenAI API key"):
        embeddings_module.get_embedding_model(
            model_id="text-embedding-test",
            model_type="openai",
            api_key=None,
        )

    openai_embeddings.assert_not_called()


def test_huggingface_embeddings_do_not_require_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_embedding_model = object()
    huggingface_embeddings = MagicMock(return_value=expected_embedding_model)
    monkeypatch.setattr(
        embeddings_module,
        "HuggingFaceEmbeddings",
        huggingface_embeddings,
    )

    result = embeddings_module.get_embedding_model(
        model_id="sentence-transformer-test",
        model_type="huggingface",
        device="cpu",
        api_key=None,
    )

    assert result is expected_embedding_model
    huggingface_embeddings.assert_called_once_with(
        model_name="sentence-transformer-test",
        model_kwargs={"device": "cpu"},
    )


def test_openai_embeddings_receive_the_explicit_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_embedding_model = object()
    openai_embeddings = MagicMock(return_value=expected_embedding_model)
    monkeypatch.setattr(
        embeddings_module,
        "OpenAIEmbeddings",
        openai_embeddings,
    )

    result = embeddings_module.get_embedding_model(
        model_id="text-embedding-test",
        model_type="openai",
        api_key="openai-test-secret",
    )

    assert result is expected_embedding_model
    openai_embeddings.assert_called_once_with(
        model="text-embedding-test",
        api_key="openai-test-secret",
    )
