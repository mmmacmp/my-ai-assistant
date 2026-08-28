import os

from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever



def get_hybrid_search_retriever(
    embedding_model: Embeddings, collection_name:str, k: int
) -> MongoDBAtlasHybridSearchRetriever:

    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE_NAME", "my_ai_assistant")
    if not mongodb_uri:
        raise ValueError(f"configuration Error")
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
                                                                connection_string=mongodb_uri,
                                                                embedding=embedding_model,
                                                                namespace=(
                                                                    f"{database_name}."
                                                                    f"{collection_name}"
                                                                ),
                                                                text_key="chunk",
                                                                embedding_key="embedding",
                                                                relevance_score_fn="dotProduct",
                                                            )
    return MongoDBAtlasHybridSearchRetriever(
        vectorstore=vectorstore,
        search_index_name="chunk_text_search",
        top_k=k,
        vector_penalty=50,
        fulltext_penalty=50,
    )

