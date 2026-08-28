


from langchain_text_splitters import RecursiveCharacterTextSplitter

from my_ai_assistant.rag.types import SummarizationType


# def get_splitter(
#     chunk_size: int, 
#     summarization_type: SummarizationType = "none", 
#     **kwargs
# ) -> RecursiveCharacterTextSplitter:

#     chunk_overlap = int(0.15 * chunk_size)

#     logger.info(
#         f"Getting splitter with chunk size: {chunk_size} and overlap: {chunk_overlap}"
#     )

#     if summarization_type == "none":
#         return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             encoding_name="cl100k_base",
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#         )

#     if summarization_type == "contextual":
#         handler = ContextualSummarizationAgent(**kwargs)
#     elif summarization_type == "simple":
#         handler = SimpleSummarizationAgent(**kwargs)

#     return HandlerRecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         encoding_name="cl100k_base",
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         handler=handler,
#     )



def get_splitter(
    chunk_size: int, 
    summarization_type: SummarizationType | None = None, 
    **kwargs
) -> RecursiveCharacterTextSplitter:
    chunk_overlap = int(chunk_size * .15)
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size = chunk_size, chunk_overlap = chunk_overlap)