from langchain_core.documents import Document as LangChainDocument
from zenml import pipeline

from my_ai_assistant.rag.types import EmbeddingModelType, RetrieverType, SummarizationType
from my_ai_assistant.steps.compute_rag_vector_index import chunk_embed_load
from my_ai_assistant.steps.fetch_from_mongodb import fetch_from_mongodb
from my_ai_assistant.steps.filter_by_quality import filter_by_quality


@pipeline
def rag_feature_pipeline(
    extract_collection_name: str,
    fetch_limit: int,
    load_collection_name: str,
    content_quality_score_threshold: float,
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
    documents = fetch_from_mongodb(collection_name = extract_collection_name, limit = fetch_limit)

    filtered_documents = filter_by_quality(documents, content_quality_score_threshold)


    chunk_embed_load(
        documents=filtered_documents,
        collection_name=load_collection_name,
        retriever_type=retriever_type,
        embedding_model_id=embedding_model_id,
        embedding_model_type=embedding_model_type,
        embedding_model_dim=embedding_model_dim,
        chunk_size=chunk_size,
        contextual_summarization_type=contextual_summarization_type,
        contextual_agent_model_id=contextual_agent_model_id,
        contextual_agent_max_characters=contextual_agent_max_characters,
        processing_batch_size=processing_batch_size,
        processing_max_workers=processing_max_workers,
        device=device,
        mock=mock,
    )

if __name__ == "__main__":
    rag_feature_pipeline(

    extract_collection_name="raw_documents",

    fetch_limit=2,

    load_collection_name="rag_documents_test",

    content_quality_score_threshold=0,

    retriever_type="contextual",

    embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",

    embedding_model_type="huggingface",

    embedding_model_dim=384,

    chunk_size=256,

    contextual_summarization_type="contextual",

    contextual_agent_model_id="openai/gpt-4o-mini",

    contextual_agent_max_characters=128,

    processing_batch_size=2,

    processing_max_workers=2,

    device="cpu",

    mock=True,

)
 
