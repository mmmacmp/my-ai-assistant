from langchain_core.documents import (
    Document as LangChainDocument,
)
from zenml import step

from my_ai_assistant.domain.document import Document
from my_ai_assistant.rag.mongodb_indexes import create_fulltext_search_index, create_vector_search_index
from my_ai_assistant.rag.processing import process_docs
from my_ai_assistant.rag.retrievers import (
    RetrieverType,
    get_retriever,
)
from my_ai_assistant.rag.splitters import (
    SummarizationType,
    get_splitter,
)
from my_ai_assistant.rag.types import EmbeddingModelType


@step
def chunk_embed_load(
    documents: list[Document],
    collection_name: str,
    retriever_type: RetrieverType,
    embedding_model_id: str,
    embedding_model_type: EmbeddingModelType,
    embedding_model_dim: int,
    chunk_size: int,
    contextual_summarization_type: SummarizationType | None,
    contextual_agent_model_id: str | None = None,
    contextual_agent_max_characters: int | None = None,
    processing_batch_size: int = 2,
    processing_max_workers: int = 2,
    device: str = "cpu",
    mock: bool = False,
) -> None:
    retriever = get_retriever(
        collection_name=collection_name,
        embedding_model_id=embedding_model_id,
        embedding_model_type=embedding_model_type,
        retriever_type=retriever_type,
        device=device,
    )

    splitter = get_splitter(
        chunk_size=chunk_size,
        summarization_type=contextual_summarization_type,
        model_id=contextual_agent_model_id,
        max_characters=contextual_agent_max_characters,
        mock=mock,
    )

    langchain_documents = [
        LangChainDocument(
            page_content=document.content,
            metadata=document.metadata.model_dump(),
        )
        for document in documents
        if document
    ]

    process_docs(
        retriever=retriever,
        docs=langchain_documents,
        splitter=splitter,
        batch_size=processing_batch_size,
        max_workers=processing_max_workers,
    )

    create_vector_search_index(collection_name, embedding_model_dim)

    if retriever_type == "contextual":
        create_fulltext_search_index(collection_name)
    
