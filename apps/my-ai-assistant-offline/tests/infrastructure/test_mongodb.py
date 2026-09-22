from unittest.mock import MagicMock

import pytest

from my_ai_assistant_offline.domain.document import Document
from my_ai_assistant_offline.infrastructure import mongodb as mongodb_module
from my_ai_assistant_offline.infrastructure.mongodb import MongoDBService


def test_service_uses_explicit_connection_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mongo_client = MagicMock()
    database = MagicMock()
    collection = MagicMock()
    mongo_client.__getitem__.return_value = database
    database.__getitem__.return_value = collection
    mongo_client_factory = MagicMock(return_value=mongo_client)
    monkeypatch.setattr(mongodb_module, "MongoClient", mongo_client_factory)

    # Ambient values must not override the dependencies supplied by the caller.
    monkeypatch.setenv("MONGODB_URI", "mongodb://ambient.invalid")
    monkeypatch.setenv("MONGODB_DATABASE_NAME", "ambient_database")

    service = MongoDBService(
        model=Document,
        collection_name="explicit_collection",
        mongodb_uri="mongodb://explicit.invalid",
        database_name="explicit_database",
    )

    mongo_client_factory.assert_called_once_with(
        "mongodb://explicit.invalid",
        serverSelectionTimeoutMS=5000,
    )
    mongo_client.__getitem__.assert_called_once_with("explicit_database")
    database.__getitem__.assert_called_once_with("explicit_collection")
    assert service.collection is collection


def test_service_requires_explicit_connection_settings() -> None:
    with pytest.raises(TypeError):
        MongoDBService(  # type: ignore[call-arg]
            model=Document,
            collection_name="documents",
        )


def test_context_manager_pings_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mongo_client = MagicMock()
    database = MagicMock()
    collection = MagicMock()
    collection._name = "documents"
    mongo_client.__getitem__.return_value = database
    database.__getitem__.return_value = collection
    monkeypatch.setattr(
        mongodb_module,
        "MongoClient",
        MagicMock(return_value=mongo_client),
    )

    service = MongoDBService(
        model=Document,
        collection_name="documents",
        mongodb_uri="mongodb://explicit.invalid",
        database_name="explicit_database",
    )

    with service as opened_service:
        assert opened_service is service

    mongo_client.admin.command.assert_called_once_with("ping")
    mongo_client.close.assert_called_once_with()


def test_clear_collection_deletes_every_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mongo_client = MagicMock()
    database = MagicMock()
    collection = MagicMock()
    collection.delete_many.return_value.deleted_count = 3
    mongo_client.__getitem__.return_value = database
    database.__getitem__.return_value = collection
    monkeypatch.setattr(
        mongodb_module,
        "MongoClient",
        MagicMock(return_value=mongo_client),
    )

    service = MongoDBService(
        model=Document,
        collection_name="rag_chunks",
        mongodb_uri="mongodb://explicit.invalid",
        database_name="explicit_database",
    )

    deleted_count = service.clear_collection()

    assert deleted_count == 3
    collection.delete_many.assert_called_once_with({})
