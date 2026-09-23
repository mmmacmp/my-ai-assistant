from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from my_ai_assistant_online.application.agents.tools import (
    mongodb_retriever as retriever_module,
)
from my_ai_assistant_online.application.agents.tools.mongodb_retriever import (
    MongoDBRetrieverTool,
)


def test_retriever_records_safe_trace_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(
            page_content="sensitive document text",
            metadata={"title": "Private note", "url": "https://private.invalid"},
        )
    ]
    monkeypatch.setattr(
        retriever_module,
        "get_retriever",
        MagicMock(return_value=retriever),
    )
    update_current_span = MagicMock()
    monkeypatch.setattr(
        retriever_module.opik_context,
        "update_current_span",
        update_current_span,
    )
    tool = MongoDBRetrieverTool(
        collection_name="private_collection",
        embedding_model_id="embedding-test",
        embedding_model_type="huggingface",
        retriever_type="contextual",
        k=5,
        mongodb_uri="mongodb://user:secret@example.invalid",
        database_name="private_database",
        openai_api_key="secret-api-key",
    )

    result = tool.forward("private query")

    metadata = update_current_span.call_args.kwargs["metadata"]
    assert metadata == {
        "embedding_model_id": "embedding-test",
        "embedding_model_type": "huggingface",
        "retriever_type": "contextual",
        "top_k": 5,
        "device": "cpu",
        "query_character_count": 13,
        "retrieved_document_count": 1,
    }
    assert "secret" not in str(metadata)
    assert "private" not in str(metadata)
    assert "https://private.invalid" in result
