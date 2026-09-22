from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from my_ai_assistant_offline.application.rag.types import (
    FULLTEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
)


def create_vector_search_index(
    collection_name: str,
    embedding_dimension: int,
    *,
    mongodb_uri: str,
    database_name: str,
) -> None:
    if not mongodb_uri:
        raise ValueError("mongodb_uri cannot be empty")
    if not database_name:
        raise ValueError("database_name cannot be empty")

    client = MongoClient(mongodb_uri)
    try:
        collection = client[database_name][collection_name]
        vector_index = SearchIndexModel(
            name=VECTOR_INDEX_NAME,
            type="vectorSearch",
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": embedding_dimension,
                        "similarity": "dotProduct",
                    }
                ]
            },
        )
        collection.create_search_index(model=vector_index)
    finally:
        client.close()


def create_fulltext_search_index(
    collection_name: str,
    *,
    mongodb_uri: str,
    database_name: str,
) -> None:
    if not mongodb_uri:
        raise ValueError("mongodb_uri cannot be empty")
    if not database_name:
        raise ValueError("database_name cannot be empty")

    client = MongoClient(mongodb_uri)
    try:
        collection = client[database_name][collection_name]
        text_index = SearchIndexModel(
            name=FULLTEXT_INDEX_NAME,
            type="search",
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "chunk": {
                            "type": "string",
                        }
                    },
                }
            },
        )
        collection.create_search_index(model=text_index)
    finally:
        client.close()
