from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from my_ai_assistant_online.application.agents.tools import (
    summarizer as summarizer_module,
)
from my_ai_assistant_online.application.agents.tools.summarizer import SummarizerTool


def test_mock_mode_needs_no_api_key() -> None:
    tool = SummarizerTool(mock=True, max_characters=4)

    assert tool.forward("abcdef") == "abcd"


def test_summarizer_records_safe_trace_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update_current_span = MagicMock()
    monkeypatch.setattr(
        summarizer_module.opik_context,
        "update_current_span",
        update_current_span,
    )
    tool = SummarizerTool(
        model_id="openai/gpt-test",
        mock=True,
        max_characters=4,
    )

    tool.forward("sensitive document text")

    metadata = update_current_span.call_args.kwargs["metadata"]
    assert metadata == {
        "model_id": "openai/gpt-test",
        "max_characters": 4,
        "input_character_count": 23,
        "mock": True,
    }
    assert "sensitive document text" not in str(metadata)


def test_real_mode_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="API key"):
        SummarizerTool(mock=False, api_key=None)


def test_real_mode_forwards_explicit_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="summary"),
                )
            ]
        )

    monkeypatch.setattr(summarizer_module, "completion", fake_completion)
    tool = SummarizerTool(model_id="openai/gpt-test", api_key="test-secret")

    assert tool.forward("document") == "summary"
    assert calls[0]["api_key"] == "test-secret"
    assert "test-secret" not in str(calls[0]["messages"])
