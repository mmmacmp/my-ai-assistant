from types import SimpleNamespace

import pytest

from my_ai_assistant_online.application.evaluation import (
    build_evaluation_sample,
    extract_retrieval_context,
)


def _action_step(tool_name: str, observation: str | None) -> dict:
    return {
        "step_number": 1,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": tool_name, "arguments": {}},
            }
        ],
        "observations": observation,
    }


def test_extract_retrieval_context_returns_only_retriever_observations() -> None:
    steps = [
        {"task": "What is RAG?"},
        _action_step("mongodb_retriever", "<search_results>RAG</search_results>"),
        _action_step("summarizer", "RAG combines retrieval and generation."),
        _action_step("final_answer", "Final answer"),
    ]

    context = extract_retrieval_context(steps)

    assert context == ["<search_results>RAG</search_results>"]


def test_extract_retrieval_context_ignores_empty_or_malformed_steps() -> None:
    steps = [
        _action_step("mongodb_retriever", None),
        {"tool_calls": "not-a-list", "observations": "ignored"},
        {"tool_calls": [], "observations": "ignored"},
    ]

    assert extract_retrieval_context(steps) == []


def test_build_evaluation_sample_uses_opik_metric_field_names() -> None:
    retrieval_step = _action_step(
        "mongodb_retriever",
        "<search_results>Retrieved evidence</search_results>",
    )
    result = SimpleNamespace(
        output="The final answer",
        steps=[retrieval_step, _action_step("final_answer", "The final answer")],
    )

    sample = build_evaluation_sample("The question", result)

    assert sample == {
        "input": "The question",
        "output": "The final answer",
        "context": ["<search_results>Retrieved evidence</search_results>"],
    }


def test_build_evaluation_sample_allows_empty_retrieval_context() -> None:
    result = SimpleNamespace(output="Direct answer", steps=[])

    sample = build_evaluation_sample("Simple question", result)

    assert sample["context"] == []


def test_build_evaluation_sample_rejects_missing_agent_output() -> None:
    result = SimpleNamespace(output=None, steps=[])

    with pytest.raises(ValueError, match="without an output"):
        build_evaluation_sample("The question", result)
