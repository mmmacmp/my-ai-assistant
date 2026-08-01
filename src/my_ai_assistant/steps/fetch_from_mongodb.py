from typing import Annotated

from zenml import get_step_context, step

from my_ai_assistant.domain.document import Document
from my_ai_assistant.infrastructure.mongodb import MongoDBService


@step
def fetch_from_mongodb(
    collection_name: str = "raw_documents",
    limit: int = 1000,
) -> Annotated[list[Document], "documents"]:
    service = MongoDBService(
        model=Document,
        collection_name=collection_name,
    )

    try:
        service.ping()

        documents = service.fetch_documents(
            limit=limit,
            query={},
        )
    finally:
        service.close()

    get_step_context().add_output_metadata(
        output_name="documents",
        metadata={
            "collection_name": collection_name,
            "count": len(documents),
            "fetch_limit": limit,
        },
    )

    return documents