"""Ingest processed documents into MongoDB."""

from typing import Annotated

from zenml import get_step_context, step

from my_ai_assistant_offline.config import get_settings
from my_ai_assistant_offline.domain.document import Document
from my_ai_assistant_offline.infrastructure.mongodb import MongoDBService


@step(enable_cache=False)
def ingest_to_mongodb(
    documents: list[Document],
    collection_name: str = "raw_documents",
    clear_collection: bool = True,
) -> Annotated[int, "inserted_documents_count"]:
    """Store processed documents in MongoDB."""

    settings = get_settings()
    service = MongoDBService(
        model=Document,
        collection_name=collection_name,
        mongodb_uri=settings.mongodb.require_uri(),
        database_name=settings.mongodb.database_name,
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
