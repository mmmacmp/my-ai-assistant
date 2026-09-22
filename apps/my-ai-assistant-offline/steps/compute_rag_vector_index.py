from langchain_core.documents import (
    Document as LangChainDocument,
)
from zenml import step

from my_ai_assistant_offline.config import get_settings
from my_ai_assistant_offline.domain.document import Document
from my_ai_assistant_offline.infrastructure.mongodb import MongoDBService
from my_ai_assistant_offline.application.rag.mongodb_indexes import (
    create_fulltext_search_index,
    create_vector_search_index,
)
from my_ai_assistant_offline.application.rag.processing import process_docs
from my_ai_assistant_offline.application.rag.retrievers import (
    RetrieverType,
    get_retriever,
)
from my_ai_assistant_offline.application.rag.splitters import (
    SummarizationType,
    get_splitter,
)
from my_ai_assistant_offline.application.rag.types import EmbeddingModelType


@step(enable_cache=False)
def chunk_embed_load(
    documents: list[Document],
    collection_name: str,
    retriever_type: RetrieverType,
    embedding_model_id: str,
    embedding_model_type: EmbeddingModelType,
    embedding_model_dim: int,
    chunk_size: int,
    contextual_summarization_type: SummarizationType | None,
    contextual_agent_model_id: str,
    contextual_agent_max_characters: int,
    processing_batch_size: int = 2,
    processing_max_workers: int = 2,
    retriever_k: int = 5,
    device: str = "cpu",
    mock: bool = False,
) -> None:
    settings = get_settings()
    mongodb_uri = settings.mongodb.require_uri()

    embedding_api_key = (
        settings.openai.require_api_key() if embedding_model_type == "openai" else None
    )
    summarization_api_key = (
        settings.openai.require_api_key()
        if contextual_summarization_type in ("contextual", "simple") and not mock
        else None
    )

    retriever = get_retriever(
        collection_name=collection_name,
        embedding_model_id=embedding_model_id,
        embedding_model_type=embedding_model_type,
        retriever_type=retriever_type,
        k=retriever_k,
        device=device,
        mongodb_uri=mongodb_uri,
        database_name=settings.mongodb.database_name,
        openai_api_key=embedding_api_key,
    )

    splitter = get_splitter(
        chunk_size=chunk_size,
        summarization_type=contextual_summarization_type,
        model_id=contextual_agent_model_id,
        max_characters=contextual_agent_max_characters,
        mock=mock,
        api_key=summarization_api_key,
    )

    langchain_documents = [
        LangChainDocument(
            page_content=document.content,
            metadata=document.metadata.model_dump(),
        )
        for document in documents
        if document
    ]

    with MongoDBService(
        model=Document,
        collection_name=collection_name,
        mongodb_uri=mongodb_uri,
        database_name=settings.mongodb.database_name,
    ) as mongodb_service:
        mongodb_service.clear_collection()

    process_docs(
        retriever=retriever,
        docs=langchain_documents,
        splitter=splitter,
        batch_size=processing_batch_size,
        max_workers=processing_max_workers,
    )

    create_vector_search_index(
        collection_name,
        embedding_model_dim,
        mongodb_uri=mongodb_uri,
        database_name=settings.mongodb.database_name,
    )

    if retriever_type == "contextual":
        create_fulltext_search_index(
            collection_name,
            mongodb_uri=mongodb_uri,
            database_name=settings.mongodb.database_name,
        )
