from collections.abc import Generator
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any

from langchain_core.documents import (
    Document as LangChainDocument,
)
from langchain_mongodb.retrievers import MongoDBAtlasParentDocumentRetriever
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
import logging


logger = logging.getLogger(__name__)


# def get_batches(
#     docs: list[LangChainDocument],
#     batch_size: int,
# ) -> Generator[list[LangChainDocument], None, None]:
#     if batch_size < 0:
#         raise ValueError(f"batch size should be more than zero")

#     for i in range(len(docs)):
#         yield docs[i : i+batch_size]



# def process_batch(
#     retriever: Any,
#     batch: list[LangChainDocument],
#     splitter: RecursiveCharacterTextSplitter,
# ) -> None:
#     try:
#         if isinstance(retriever, MongoDBAtlasParentDocumentRetriever):
#             retriever.add_documents(batch)
#         else:
#             split_docs = splitter.split_documents(batch)
#             retriever.vectorstore.add_documents(split_docs)

#         logger.info(f"Successfully processed {len(batch)} documents.")
#     except Exception as e:
#         logger.warning(f"Error processing batch of {len(batch)} documents: {str(e)}")



# def process_docs(
#     retriever: Any,
#     docs: list[LangChainDocument],
#     splitter: RecursiveCharacterTextSplitter,
#     batch_size: int = 4,
#     max_workers: int = 2,
# ) -> list[None]:
    
#     batches = list(get_batches(docs, batch_size))

#     results = []

#     total_docs = len(docs)

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(process_batch, 
#                                    retriever, 
#                                    batch,
#                                    splitter)
#             for batch in batches]

#         with tqdm(total=total_docs, desc="Processing documents") as pbar:
#             for future in as_completed(futures):
#                 result = future.result()
#                 results.append(result)
#                 pbar.update(batch_size)

#     return results



def get_batches(
    docs: list[LangChainDocument],
    batch_size: int,
) -> Generator[list[LangChainDocument], None, None]:
    docs_len = len(docs)

    for i in range(0, docs_len, batch_size):
        yield docs[i : i+batch_size]



def process_batch(
    retriever: Any,
    batch: list[LangChainDocument],
    splitter: RecursiveCharacterTextSplitter,
) -> None:
    
    if isinstance(retriever, MongoDBAtlasParentDocumentRetriever):
        retriever.add_documents(batch)
    else:
        splitted_docs = splitter.split_documents(batch)
        retriever.vectorstore.add_documents(splitted_docs)


def process_docs(
    retriever: Any,
    docs: list[LangChainDocument],
    splitter: RecursiveCharacterTextSplitter,
    batch_size: int = 4,
    max_workers: int = 2,
) -> list[None]:
    batches = list(get_batches(docs, batch_size))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, retriever, batch, splitter)
                   for batch in batches]

    completed = as_completed(futures)
    results = []
    for complete in completed:
        results.append(complete.result())

    return results
