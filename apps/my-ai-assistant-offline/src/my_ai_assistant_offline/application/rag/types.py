
from typing import Literal


VECTOR_INDEX_NAME = "chunk_vector_index"
FULLTEXT_INDEX_NAME = "chunk_text_search"

EmbeddingModelType = Literal["openai", "huggingface"]

RetrieverType = Literal["parent", "contextual"]

SummarizationType = Literal["none", "simple", "contextual"]