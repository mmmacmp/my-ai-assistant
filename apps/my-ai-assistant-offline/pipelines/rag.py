from zenml import pipeline

from my_ai_assistant_offline.config import get_settings
from my_ai_assistant_offline.application.rag.types import (
    EmbeddingModelType,
    RetrieverType,
    SummarizationType,
)
from steps.compute_rag_vector_index import chunk_embed_load
from steps.fetch_from_mongodb import fetch_from_mongodb
from steps.filter_by_quality import filter_by_quality


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
    contextual_agent_model_id: str,
    contextual_agent_max_characters: int,
    processing_batch_size: int = 2,
    processing_max_workers: int = 2,
    retriever_k: int = 5,
    device: str = "cpu",
    mock: bool = False,
) -> None:
    documents = fetch_from_mongodb(
        collection_name=extract_collection_name, limit=fetch_limit
    )

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
        retriever_k=retriever_k,
        device=device,
        mock=mock,
    )


if __name__ == "__main__":
    settings = get_settings()
    rag_feature_pipeline(
        extract_collection_name=settings.mongodb.raw_collection_name,
        fetch_limit=1000,
        load_collection_name=settings.mongodb.rag_collection_name,
        content_quality_score_threshold=0,
        retriever_type="contextual",
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_type="huggingface",
        embedding_model_dim=384,
        chunk_size=256,
        contextual_summarization_type="simple",
        contextual_agent_model_id=settings.openai.model_id,
        contextual_agent_max_characters=128,
        processing_batch_size=2,
        processing_max_workers=2,
        retriever_k=settings.rag.retriever_k,
        device=settings.rag.device,
        mock=False,
    )
