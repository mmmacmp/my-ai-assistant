from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from my_ai_assistant_offline.application.rag.types import EmbeddingModelType


def get_embedding_model(
    model_id: str,
    model_type: EmbeddingModelType = "huggingface",
    device: str = "cpu",
    api_key: str | None = None,
) -> Embeddings:
    if model_type == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={"device": device},
        )

    if model_type == "openai":
        if api_key is None:
            raise RuntimeError("An OpenAI API key is required for OpenAI embeddings")
        return OpenAIEmbeddings(model=model_id, api_key=api_key)

    raise ValueError(f"Unsupported embedding model type: {model_type}")
