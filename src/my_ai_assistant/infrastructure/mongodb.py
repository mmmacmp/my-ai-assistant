"""MongoDB connection service."""

import os
from typing import Any, Generic, TypeVar

from dotenv import load_dotenv
from my_ai_assistant.domain.document import Document
from pydantic import BaseModel
from pymongo import MongoClient


load_dotenv()

T = TypeVar("T", bound=BaseModel)


class MongoDBService(Generic[T]):
    """Store and retrieve Pydantic models using MongoDB."""

    def __init__(
        self,
        model: type[T],
        collection_name: str,
    ) -> None:
        mongodb_uri = os.getenv("MONGODB_URI")
        database_name = os.getenv(
            "MONGODB_DATABASE_NAME",
            "my_ai_assistant",
        )

        if not mongodb_uri:
            raise RuntimeError("MONGODB_URI is not configured")

        self.model = model
        self.client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

    def __enter__(self):
        print(f"Opening {self.ping()}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self:
            self.close()
            print(f"Closed {self.collection._name}")

        
    def ping(self) -> None:
        """Verify that MongoDB is reachable."""

        self.client.admin.command("ping")

    def close(self) -> None:
        """Close the MongoDB connection."""

        self.client.close()

    def ingest_documents(self,
                        documents: list[Document],
                        clear_collection: bool = False) -> int:
        if not documents:
            raise ValueError("No documents were provided")

        if not all(
            isinstance(document, self.model)
            for document in documents
        ):
            raise TypeError(
                f"Every item must be a {self.model.__name__}"
            )
        mongo_documents = [
        document.model_dump(mode="json")
        for document in documents
        ]

        for mongo_document in mongo_documents:
            mongo_document.pop("_id", None)

        if clear_collection:
            self.collection.delete_many({})

        result = self.collection.insert_many(mongo_documents)

        return len(result.inserted_ids)


    def fetch_documents(self, 
                        limit: int = 1000,
                        query: dict[str, Any] | None = None) -> list[T]:
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


    