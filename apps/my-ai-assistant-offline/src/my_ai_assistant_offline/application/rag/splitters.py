
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from my_ai_assistant_offline.application.agents.contextual_summarization import ContextualSummarizationAgent
from my_ai_assistant_offline.application.agents.simple_summarization import SimpleSummarizationAgent
from my_ai_assistant_offline.application.rag.handler_splitter import HandlerRecursiveCharacterTextSplitter
from my_ai_assistant_offline.application.rag.types import SummarizationType


logger = logging.getLogger(__name__)


def get_splitter(
    chunk_size: int,
    summarization_type: SummarizationType | None = None,
    **kwargs,
) -> RecursiveCharacterTextSplitter:
    chunk_overlap = int(chunk_size * 0.15)
    logger.info(
        f"Getting splitter with chunk size: {chunk_size} and overlap: {chunk_overlap}"
    )

    handler = None
    if summarization_type == "contextual":
        handler = ContextualSummarizationAgent(**kwargs)
    elif summarization_type == "simple":
        handler = SimpleSummarizationAgent(**kwargs)
    elif summarization_type not in (None, "none"):
        raise ValueError(f"Unsupported summarization type: {summarization_type}")

    return HandlerRecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        disallowed_special=(),
        handler=handler,
    )
