

import os

from langchain_core.embeddings import Embeddings
from langchain_mongodb.retrievers import MongoDBAtlasParentDocumentRetriever

from my_ai_assistant.rag.splitters import get_splitter


def get_parent_retriever(
        embedding_model: Embeddings,
        collection_name: str,
        k: int = 3,
) -> MongoDBAtlasParentDocumentRetriever:
    
    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv(
        "MONGODB_DATABASE_NAME",
        "my_ai_assistant",)
    
    if not mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured")
    child_splitter = get_splitter(chunk_size=200)
    parent_splitter = get_splitter(chunk_size=800)

    return MongoDBAtlasParentDocumentRetriever.from_connection_string(child_splitter=child_splitter,
                                                                      connection_string=mongodb_uri,
                                                                      embedding_model=embedding_model,
                                                                      parent_splitter=parent_splitter,
                                                                      database_name= database_name,
                                                                      collection_name=collection_name,
                                                                      text_key="page_content",
                                                                      search_kwargs={"k": k})
    