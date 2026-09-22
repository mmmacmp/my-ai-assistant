from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from my_ai_assistant_offline.application.rag.types import (
    FULLTEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
)


def get_hybrid_search_retriever(
    embedding_model: Embeddings,
    collection_name: str,
    k: int,
    *,
    mongodb_uri: str,
    database_name: str,
) -> MongoDBAtlasHybridSearchRetriever:
    if not mongodb_uri:
        raise ValueError("mongodb_uri cannot be empty")
    if not database_name:
        raise ValueError("database_name cannot be empty")

    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=mongodb_uri,
        embedding=embedding_model,
        namespace=f"{database_name}.{collection_name}",
        index_name=VECTOR_INDEX_NAME,
        text_key="chunk",
        embedding_key="embedding",
        relevance_score_fn="dotProduct",
    )
    return MongoDBAtlasHybridSearchRetriever(
        vectorstore=vectorstore,
        search_index_name=FULLTEXT_INDEX_NAME,
        top_k=k,
        vector_penalty=50,
        fulltext_penalty=50,
    )
