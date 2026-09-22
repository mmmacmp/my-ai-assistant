from langchain_core.embeddings import Embeddings
from langchain_mongodb.retrievers import MongoDBAtlasParentDocumentRetriever

from my_ai_assistant_offline.application.rag.splitters import get_splitter
from my_ai_assistant_offline.application.rag.types import (
    VECTOR_INDEX_NAME,
)


def get_parent_retriever(
    embedding_model: Embeddings,
    collection_name: str,
    k: int = 3,
    *,
    mongodb_uri: str,
    database_name: str,
) -> MongoDBAtlasParentDocumentRetriever:
    if not mongodb_uri:
        raise ValueError("mongodb_uri cannot be empty")
    if not database_name:
        raise ValueError("database_name cannot be empty")

    child_splitter = get_splitter(chunk_size=200)
    parent_splitter = get_splitter(chunk_size=800)

    return MongoDBAtlasParentDocumentRetriever.from_connection_string(
        child_splitter=child_splitter,
        connection_string=mongodb_uri,
        embedding_model=embedding_model,
        parent_splitter=parent_splitter,
        database_name=database_name,
        collection_name=collection_name,
        text_key="page_content",
        index_name=VECTOR_INDEX_NAME,
        search_kwargs={"k": k},
    )
