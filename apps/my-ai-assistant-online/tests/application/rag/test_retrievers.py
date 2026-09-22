from unittest.mock import MagicMock

import pytest

from my_ai_assistant_online.application.rag import retrievers as retrievers_module


@pytest.mark.parametrize("retriever_type", ["contextual", "parent"])
def test_get_retriever_forwards_explicit_configuration(
    monkeypatch: pytest.MonkeyPatch,
    retriever_type: str,
) -> None:
    embedding_model = object()
    expected_retriever = object()
    embedding_factory = MagicMock(return_value=embedding_model)
    contextual_factory = MagicMock(return_value=expected_retriever)
    parent_factory = MagicMock(return_value=expected_retriever)
    monkeypatch.setattr(retrievers_module, "get_embedding_model", embedding_factory)
    monkeypatch.setattr(
        retrievers_module,
        "get_hybrid_search_retriever",
        contextual_factory,
    )
    monkeypatch.setattr(
        retrievers_module,
        "get_parent_retriever",
        parent_factory,
    )

    result = retrievers_module.get_retriever(
        embedding_model_id="embedding-test",
        embedding_model_type="openai",
        collection_name="rag_test",
        retriever_type=retriever_type,  # type: ignore[arg-type]
        k=7,
        device="cuda",
        mongodb_uri="mongodb://explicit.invalid",
        database_name="assistant_test",
        openai_api_key="openai-test-secret",
    )

    assert result is expected_retriever
    embedding_factory.assert_called_once_with(
        model_id="embedding-test",
        model_type="openai",
        device="cuda",
        api_key="openai-test-secret",
    )

    expected_arguments = {
        "mongodb_uri": "mongodb://explicit.invalid",
        "database_name": "assistant_test",
    }
    if retriever_type == "contextual":
        contextual_factory.assert_called_once_with(
            embedding_model,
            "rag_test",
            7,
            **expected_arguments,
        )
        parent_factory.assert_not_called()
    else:
        parent_factory.assert_called_once_with(
            embedding_model=embedding_model,
            collection_name="rag_test",
            k=7,
            **expected_arguments,
        )
        contextual_factory.assert_not_called()
