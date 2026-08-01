"""Ingest processed documents into MongoDB."""

from typing import Annotated

from zenml import get_step_context, step

from my_ai_assistant.domain.document import Document
from my_ai_assistant.infrastructure.mongodb import MongoDBService


@step
def ingest_to_mongodb(
    documents: list[Document],
    collection_name: str = "raw_documents",
    clear_collection: bool = True,
) -> Annotated[int, "inserted_documents_count"]:
    """Store processed documents in MongoDB."""

    service = MongoDBService(
        model=Document,
        collection_name=collection_name,
    )

    try:
        service.ping()

        inserted_count = service.ingest_documents(
            documents=documents,
            clear_collection=clear_collection,
        )
    finally:
        service.close()

    get_step_context().add_output_metadata(
        output_name="inserted_documents_count",
        metadata={
            "collection_name": collection_name,
            "inserted_documents": inserted_count,
            "cleared_before_insert": clear_collection,
        },
    )

    return inserted_count