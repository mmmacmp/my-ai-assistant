import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from my_ai_assistant_offline.application.agents import quality as quality_module
from my_ai_assistant_offline.application.agents.quality import (
    HeuristicQualityAgent,
    QualityScoreAgent,
)
from my_ai_assistant_offline.domain.document import Document, DocumentMetadata


def make_document(
    document_id: str,
    *,
    content: str = "A useful document about retrieval augmented generation.",
    score: float | None = None,
) -> Document:
    return Document(
        id=document_id,
        metadata=DocumentMetadata(
            id=document_id,
            url=f"https://example.com/{document_id}",
            title=document_id,
            properties={},
        ),
        content=content,
        content_quality_score=score,
    )


def model_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_real_agent_calls_litellm_and_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return model_response('{"score": 0.8}')

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)
    source = make_document("doc-1")

    [result] = QualityScoreAgent(
        model_id="gpt-test",
        max_retries=0,
    )([source])

    assert result.content_quality_score == 0.8
    assert source.content_quality_score is None
    assert result.id == source.id
    assert calls[0]["model"] == "gpt-test"
    assert calls[0]["messages"][1]["content"] == source.content
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["stream"] is False


def test_api_key_is_forwarded_without_being_added_to_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return model_response('{"score": 0.8}')

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)

    QualityScoreAgent(api_key="test-secret", max_retries=0)([make_document("api-key")])

    assert calls[0]["api_key"] == "test-secret"
    assert "test-secret" not in str(calls[0]["messages"])


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_real_agent_accepts_boundary_scores(
    monkeypatch: pytest.MonkeyPatch,
    score: float,
) -> None:
    async def fake_acompletion(**_: Any) -> SimpleNamespace:
        return model_response(f'{{"score": {score}}}')

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)

    [result] = QualityScoreAgent(max_retries=0)([make_document("boundary")])

    assert result.content_quality_score == score


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        "{}",
        '{"score": "0.7"}',
        '{"score": -0.1}',
        '{"score": 1.1}',
        '{"score": 0.7, "reason": "extra field"}',
        None,
    ],
)
def test_invalid_model_output_leaves_document_unscored(
    monkeypatch: pytest.MonkeyPatch,
    content: str | None,
) -> None:
    async def fake_acompletion(**_: Any) -> SimpleNamespace:
        return model_response(content)

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)
    source = make_document("invalid")

    [result] = QualityScoreAgent(max_retries=0)([source])

    assert result is source
    assert result.content_quality_score is None


def test_missing_choices_leaves_document_unscored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(choices=[])

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)
    source = make_document("no-choice")

    [result] = QualityScoreAgent(max_retries=0)([source])

    assert result is source


def test_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    async def fake_acompletion(**_: Any) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError
        return model_response('{"score": 0.6}')

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)

    [result] = QualityScoreAgent(
        max_retries=1,
        retry_delay_seconds=0,
    )([make_document("retry")])

    assert attempts == 2
    assert result.content_quality_score == 0.6


def test_failure_is_isolated_and_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        content = kwargs["messages"][1]["content"]
        if content == "bad":
            raise RuntimeError
        return model_response('{"score": 0.9}')

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)
    documents = [
        make_document("first", content="good"),
        make_document("second", content="bad"),
        make_document("third", content="good"),
    ]

    results = QualityScoreAgent(max_retries=0)(documents)

    assert [document.id for document in results] == [
        "first",
        "second",
        "third",
    ]
    assert [document.content_quality_score for document in results] == [
        0.9,
        None,
        0.9,
    ]


def test_empty_and_already_scored_documents_skip_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_acompletion(**_: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return model_response('{"score": 0.1}')

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)
    scored = make_document("scored", score=0.4)
    agent = QualityScoreAgent(max_retries=0)

    assert agent([]) == []
    assert agent([scored]) == [scored]
    assert calls == 0


def test_concurrency_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    active_requests = 0
    peak_requests = 0

    async def fake_acompletion(**_: Any) -> SimpleNamespace:
        nonlocal active_requests, peak_requests
        active_requests += 1
        peak_requests = max(peak_requests, active_requests)
        await asyncio.sleep(0.01)
        active_requests -= 1
        return model_response('{"score": 0.75}')

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)
    documents = [make_document(f"doc-{index}") for index in range(6)]

    results = QualityScoreAgent(
        max_concurrent_requests=2,
        max_retries=0,
    )(documents)

    assert peak_requests == 2
    assert all(document.content_quality_score == 0.75 for document in results)


def test_document_content_is_clipped_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return model_response('{"score": 0.7}')

    monkeypatch.setattr(quality_module, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        quality_module,
        "encode",
        lambda **_: [10, 20, 30, 40],
    )
    monkeypatch.setattr(
        quality_module,
        "decode",
        lambda **kwargs: f"tokens:{kwargs['tokens']}",
    )

    QualityScoreAgent(max_document_tokens=2, max_retries=0)([make_document("long")])

    assert calls[0]["messages"][1]["content"] == "tokens:[10, 20]"


def test_mock_mode_is_deterministic_and_skips_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_call(**_: Any) -> SimpleNamespace:
        raise AssertionError("LiteLLM must not be called in mock mode")

    monkeypatch.setattr(quality_module, "acompletion", unexpected_call)
    already_scored = make_document("existing", score=0.2)

    results = QualityScoreAgent(mock=True)([make_document("new"), already_scored])

    assert [document.content_quality_score for document in results] == [
        0.5,
        0.2,
    ]
    assert results[1] is already_scored


def test_heuristics_score_empty_and_link_heavy_documents() -> None:
    documents = [
        make_document("empty", content="   "),
        make_document("mostly-links", content="x" * 100).model_copy(
            update={"child_urls": ["h" * 70]}
        ),
        make_document("some-links", content="x" * 100).model_copy(
            update={"child_urls": ["h" * 50]}
        ),
        make_document("useful", content="x" * 100).model_copy(
            update={"child_urls": ["h" * 49]}
        ),
    ]

    results = HeuristicQualityAgent()(documents)

    assert [document.content_quality_score for document in results] == [
        0.0,
        0.0,
        0.2,
        None,
    ]


def test_heuristics_preserve_existing_scores() -> None:
    source = make_document("existing-empty", content="", score=0.9)

    [result] = HeuristicQualityAgent()([source])

    assert result is source
    assert result.content_quality_score == 0.9


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("max_concurrent_requests", 0),
        ("max_retries", -1),
        ("retry_delay_seconds", -1),
        ("request_timeout_seconds", 0),
        ("max_document_tokens", 0),
    ],
)
def test_invalid_configuration_fails_fast(argument: str, value: int) -> None:
    with pytest.raises(ValueError):
        QualityScoreAgent(**{argument: value})
