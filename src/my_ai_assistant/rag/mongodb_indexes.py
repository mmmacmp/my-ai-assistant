

import os

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from my_ai_assistant.domain.document import Document
from my_ai_assistant.infrastructure.mongodb import MongoDBService


def create_vector_search_index(
        collection_name: str,
        embedding_dimension: int,
) -> None:
    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb = os.getenv("MONGODB_DATABASE_NAME", "my_ai_assistant")

    if not mongodb_uri:
        raise ValueError("mongodb uri is not configured")

    client = MongoClient(mongodb_uri)
    try:
        collection = client[mongodb][collection_name]
        vector_index = SearchIndexModel(name="chunk_vector_index",
                                        type="vectorSearch",
                                        definition={
                                            "fields" : [{"type" : "vector",
                                                         "path" : "embedding",
                                                         "numDimensions": embedding_dimension,
                                                         "similarity": "dotProduct"}]
                                        })
        collection.create_search_index(model=vector_index)
    finally:
        client.close()

    



def create_fulltext_search_index(
        collection_name: str,
) -> None:
        mongodb_uri = os.getenv("MONGODB_URI")
        mongodb = os.getenv("MONGODB_DATABASE_NAME", "my_ai_assistant")
    
        if not mongodb_uri:
            raise ValueError("mongodb uri is not configured")
    
        client = MongoClient(mongodb_uri)
        try:
            collection = client[mongodb][collection_name]
            text_index = SearchIndexModel(name="chunk_text_index",
                                            type="search",
                                            definition={ "mappings": {
                                                 "dynamic": False,
                                                "fields" :{"chunk":{"type" : "string",}}
                                            }})
            collection.create_search_index(model=text_index)
        finally:
            client.close()