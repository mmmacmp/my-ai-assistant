from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from my_ai_assistant_online.application.evaluation_task import AgentEvaluationTask


def test_evaluation_task_runs_agent_and_builds_metric_inputs() -> None:
    retrieval_step = {
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "mongodb_retriever", "arguments": {}},
            }
        ],
        "observations": "Retrieved evidence",
    }
    agent = MagicMock()
    agent.run.return_value = SimpleNamespace(
        output="Agent answer",
        steps=[retrieval_step],
    )
    task = AgentEvaluationTask(agent)

    output = task({"input": "  What is RAG?  ", "expected_output": "Reference"})

    agent.run.assert_called_once_with(
        "What is RAG?",
        reset=True,
        return_full_result=True,
    )
    assert output == {
        "input": "What is RAG?",
        "output": "Agent answer",
        "context": ["Retrieved evidence"],
    }


@pytest.mark.parametrize("dataset_item", [{}, {"input": ""}, {"input": None}])
def test_evaluation_task_requires_a_question(dataset_item: dict) -> None:
    agent = MagicMock()
    task = AgentEvaluationTask(agent)

    with pytest.raises(ValueError, match="non-empty 'input'"):
        task(dataset_item)

    agent.run.assert_not_called()
