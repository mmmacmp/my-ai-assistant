from unittest.mock import MagicMock

import pytest

from my_ai_assistant_offline.application.agents import (
    simple_summarization as summarization_module,
)
from my_ai_assistant_offline.application.agents.simple_summarization import (
    SimpleSummarizationAgent,
)


def test_mock_mode_runs_without_api_key_or_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_client = MagicMock(
        side_effect=AssertionError("OpenAI client must not be built in mock mode")
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(summarization_module, "OpenAI", openai_client)

    agent = SimpleSummarizationAgent(mock=True)

    assert agent(
        text="Full document",
        chunks=["first chunk", "second chunk"],
    ) == [
        "This is a fake summary \nfirst chunk",
        "This is a fake summary \nsecond chunk",
    ]
    openai_client.assert_not_called()


def test_real_mode_rejects_missing_api_key_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_client = MagicMock()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(summarization_module, "OpenAI", openai_client)

    with pytest.raises(RuntimeError, match="API key or configured client"):
        SimpleSummarizationAgent(mock=False, api_key=None, client=None)

    openai_client.assert_not_called()


def test_real_mode_accepts_an_injected_client_without_api_key() -> None:
    client = MagicMock()

    agent = SimpleSummarizationAgent(
        mock=False,
        api_key=None,
        client=client,
    )

    assert agent.client is client
