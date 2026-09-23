from threading import Lock
from typing import Any, Protocol

from my_ai_assistant_online.application.evaluation import (
    AgentRunResult,
    EvaluationSample,
    build_evaluation_sample,
)


class AgentRunner(Protocol):
    def run(self, task: str, **kwargs: Any) -> AgentRunResult: ...


class AgentEvaluationTask:
    """Run one dataset question through the agent and prepare Opik inputs."""

    def __init__(self, agent: AgentRunner) -> None:
        self._agent = agent
        self._run_lock = Lock()

    def __call__(self, dataset_item: dict[str, Any]) -> EvaluationSample:
        question = dataset_item.get("input")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Each evaluation item must contain a non-empty 'input'.")

        question = question.strip()
        with self._run_lock:
            result = self._agent.run(
                question,
                reset=True,
                return_full_result=True,
            )

        return build_evaluation_sample(question, result)
