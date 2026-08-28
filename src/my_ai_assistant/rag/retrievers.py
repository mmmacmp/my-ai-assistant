

from typing import Any, Literal, Protocol

from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from my_ai_assistant.rag.contextual_retriever import get_hybrid_search_retriever
from my_ai_assistant.rag.embeddings import get_embedding_model
from my_ai_assistant.rag.parent_retriever import get_parent_retriever
from my_ai_assistant.rag.types import EmbeddingModelType


RetrieverType = Literal["parent", "contextual"]
RetrieverModel = Any


def get_retriever(
    embedding_model_id: str,
    collection_name: str,
    retriever_type: RetrieverType = "contextual",
    embedding_model_type: EmbeddingModelType = "huggingface",
    k: int = 3,
    device: str = "cpu",
) -> RetrieverModel:
    embedding_model = get_embedding_model(
        model_id=embedding_model_id,
        model_type=embedding_model_type,
        device=device,
    )

    if retriever_type == "contextual":
        return get_hybrid_search_retriever(
            embedding_model,
            collection_name,
            k,
        )

    if retriever_type == "parent":
        return get_parent_retriever(
            embedding_model = embedding_model,
            collection_name = collection_name,
            k = k,
        )

    raise ValueError(
        f"Invalid retriever type: {retriever_type}"
    )