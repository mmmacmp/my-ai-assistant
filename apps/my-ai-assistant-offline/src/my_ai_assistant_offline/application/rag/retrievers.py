from typing import Any, Literal

from my_ai_assistant_offline.application.rag.contextual_retriever import (
    get_hybrid_search_retriever,
)
from my_ai_assistant_offline.application.rag.embeddings import get_embedding_model
from my_ai_assistant_offline.application.rag.parent_retriever import (
    get_parent_retriever,
)
from my_ai_assistant_offline.application.rag.types import EmbeddingModelType


RetrieverType = Literal["parent", "contextual"]
RetrieverModel = Any


def get_retriever(
    embedding_model_id: str,
    collection_name: str,
    retriever_type: RetrieverType = "contextual",
    embedding_model_type: EmbeddingModelType = "huggingface",
    k: int = 3,
    device: str = "cpu",
    *,
    mongodb_uri: str,
    database_name: str,
    openai_api_key: str | None = None,
) -> RetrieverModel:
    embedding_model = get_embedding_model(
        model_id=embedding_model_id,
        model_type=embedding_model_type,
        device=device,
        api_key=openai_api_key,
    )

    if retriever_type == "contextual":
        return get_hybrid_search_retriever(
            embedding_model,
            collection_name,
            k,
            mongodb_uri=mongodb_uri,
            database_name=database_name,
        )

    if retriever_type == "parent":
        return get_parent_retriever(
            embedding_model=embedding_model,
            collection_name=collection_name,
            k=k,
            mongodb_uri=mongodb_uri,
            database_name=database_name,
        )

    raise ValueError(f"Invalid retriever type: {retriever_type}")
