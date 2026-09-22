from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from my_ai_assistant_offline.application.rag import (
    mongodb_indexes as indexes_module,
)


@pytest.mark.parametrize(
    ("index_creator", "extra_arguments"),
    [
        (indexes_module.create_vector_search_index, {"embedding_dimension": 384}),
        (indexes_module.create_fulltext_search_index, {}),
    ],
)
def test_index_creator_uses_explicit_connection_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
    index_creator: Callable[..., None],
    extra_arguments: dict[str, int],
) -> None:
    client = MagicMock()
    database = MagicMock()
    collection = MagicMock()
    client.__getitem__.return_value = database
    database.__getitem__.return_value = collection
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(indexes_module, "MongoClient", client_factory)

    index_creator(
        collection_name="rag_test",
        mongodb_uri="mongodb://explicit.invalid",
        database_name="assistant_test",
        **extra_arguments,
    )

    client_factory.assert_called_once_with("mongodb://explicit.invalid")
    client.__getitem__.assert_called_once_with("assistant_test")
    database.__getitem__.assert_called_once_with("rag_test")
    collection.create_search_index.assert_called_once()
    client.close.assert_called_once_with()


def test_index_creator_closes_client_when_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    database = MagicMock()
    collection = MagicMock()
    client.__getitem__.return_value = database
    database.__getitem__.return_value = collection
    collection.create_search_index.side_effect = RuntimeError("creation failed")
    monkeypatch.setattr(
        indexes_module,
        "MongoClient",
        MagicMock(return_value=client),
    )

    with pytest.raises(RuntimeError, match="creation failed"):
        indexes_module.create_vector_search_index(
            collection_name="rag_test",
            embedding_dimension=384,
            mongodb_uri="mongodb://explicit.invalid",
            database_name="assistant_test",
        )

    client.close.assert_called_once_with()
