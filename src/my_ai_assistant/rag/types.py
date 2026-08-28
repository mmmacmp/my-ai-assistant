
from typing import Literal


EmbeddingModelType = Literal["openai", "huggingface"]

RetrieverType = Literal["parent", "contextual"]

SummarizationType = Literal["none", "simple", "contextual"]