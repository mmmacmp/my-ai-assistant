
from re import L
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from my_ai_assistant.rag.types import EmbeddingModelType


def get_embedding_model(model_id: str,
                        model_type: EmbeddingModelType = "huggingface",
                        device: str = "cpu"
                        ) -> Embeddings:
    if model_type == "huggingface":
        return HuggingFaceEmbeddings(model_name = model_id,
                                     model_kwargs = {"device":device})
    elif model_type == "openai":
        return OpenAIEmbeddings(model = model_id)
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")