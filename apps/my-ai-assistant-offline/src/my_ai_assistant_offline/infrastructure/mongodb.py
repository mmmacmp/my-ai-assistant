"""MongoDB connection service."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pymongo import MongoClient

T = TypeVar("T", bound=BaseModel)


class MongoDBService(Generic[T]):
    """Store and retrieve Pydantic models using MongoDB."""

    def __init__(
        self,
        model: type[T],
        collection_name: str,
        *,
        mongodb_uri: str,
        database_name: str,
    ) -> None:
        if not mongodb_uri:
            raise ValueError("mongodb_uri cannot be empty")
        if not database_name:
            raise ValueError("database_name cannot be empty")
        if not collection_name:
            raise ValueError("collection_name cannot be empty")

        self.model = model
        self.client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

    def __enter__(self) -> "MongoDBService[T]":
        self.ping()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def ping(self) -> None:
        """Verify that MongoDB is reachable."""

        self.client.admin.command("ping")

    def close(self) -> None:
        """Close the MongoDB connection."""

        self.client.close()

    def clear_collection(self) -> int:
        """Delete every document while preserving the collection and its indexes."""

        result = self.collection.delete_many({})
        return result.deleted_count

    def ingest_documents(
        self,
        documents: list[T],
        clear_collection: bool = False,
    ) -> int:
        if not documents:
            raise ValueError("No documents were provided")

        if not all(isinstance(document, self.model) for document in documents):
            raise TypeError(f"Every item must be a {self.model.__name__}")
        mongo_documents = [document.model_dump(mode="json") for document in documents]

        for mongo_document in mongo_documents:
            mongo_document.pop("_id", None)

        if clear_collection:
            self.clear_collection()

        result = self.collection.insert_many(mongo_documents)

        return len(result.inserted_ids)

    def fetch_documents(
        self, limit: int = 1000, query: dict[str, Any] | None = None
    ) -> list[T]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        mongo_query = query or {}

        records = self.collection.find(mongo_query).limit(limit)

        documents: list[T] = []

        for record in records:
            record.pop("_id", None)

            document = self.model.model_validate(record)
            documents.append(document)

        return documents
