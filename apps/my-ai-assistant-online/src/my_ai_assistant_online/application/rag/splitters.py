


from langchain_text_splitters import RecursiveCharacterTextSplitter

from my_ai_assistant_online.application.rag.types import SummarizationType



def get_splitter(
    chunk_size: int,
    summarization_type: SummarizationType | None = None,
    **kwargs
) -> RecursiveCharacterTextSplitter:
    chunk_overlap = int(chunk_size * .15)
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size = chunk_size, chunk_overlap = chunk_overlap)
