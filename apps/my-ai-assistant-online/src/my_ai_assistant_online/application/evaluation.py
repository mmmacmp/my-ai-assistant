from collections.abc import Iterable, Mapping
from typing import Any, Protocol, TypedDict

from my_ai_assistant_online.application.agents.tools.names import (
    MONGODB_RETRIEVER_TOOL_NAME,
)


class AgentRunResult(Protocol):
    output: Any
    steps: list[dict[str, Any]]


class EvaluationSample(TypedDict):
    input: str
    output: str
    context: list[str]


def build_evaluation_sample(
    question: str,
    result: AgentRunResult,
) -> EvaluationSample:
    """Convert one SmolAgents run into the fields expected by Opik metrics."""
    if result.output is None:
        raise ValueError("Cannot evaluate an agent run without an output.")

    return {
        "input": question,
        "output": str(result.output),
        "context": extract_retrieval_context(result.steps),
    }


def extract_retrieval_context(
    steps: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Extract MongoDB retrieval observations from a SmolAgents run result."""
    context: list[str] = []

    for step in steps:
        observation = step.get("observations")
        if not isinstance(observation, str) or not observation.strip():
            continue

        tool_names = _extract_tool_names(step.get("tool_calls"))
        if MONGODB_RETRIEVER_TOOL_NAME in tool_names:
            context.append(observation.strip())

    return context


def _extract_tool_names(tool_calls: Any) -> set[str]:
    if not isinstance(tool_calls, list):
        return set()

    names: set[str] = set()
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue

        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            continue

        name = function.get("name")
        if isinstance(name, str):
            names.add(name)

    return names
