from typing import Annotated

from zenml import get_step_context, step

from my_ai_assistant_offline.config import get_settings
from my_ai_assistant_offline.domain.document import Document
from my_ai_assistant_offline.infrastructure.mongodb import MongoDBService


@step(enable_cache=False)
def fetch_from_mongodb(
    collection_name: str = "raw_documents",
    limit: int = 1000,
) -> Annotated[list[Document], "documents"]:
    settings = get_settings()

    with MongoDBService(
        collection_name=collection_name,
        model=Document,
        mongodb_uri=settings.mongodb.require_uri(),
        database_name=settings.mongodb.database_name,
    ) as service:
        docs = service.fetch_documents(limit=limit)

    get_step_context().add_output_metadata(
        output_name="documents",
        metadata={
            "collection_name": collection_name,
            "limit": limit,
            "count": len(docs),
        },
    )

    return docs
