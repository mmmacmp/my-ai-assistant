from collections.abc import Sequence
from typing import Any

from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.evaluation_result import EvaluationResult

from my_ai_assistant_online.application.evaluation_metrics import (
    build_evaluation_metrics,
)
from my_ai_assistant_online.application.evaluation_task import (
    AgentEvaluationTask,
    AgentRunner,
)
from my_ai_assistant_online.config import OnlineSettings, get_settings
from my_ai_assistant_online.opik_utils import configure_opik


DEFAULT_EVALUATION_ITEMS: tuple[dict[str, str], ...] = (
    {"input": "What is the difference between a vector index and a vector database?"},
    {"input": "What types of memory can an AI agent use?"},
    {"input": "How does contextual retrieval improve ordinary vector search?"},
)


def run_evaluation(
    agent: AgentRunner,
    *,
    settings: OnlineSettings | None = None,
    items: Sequence[dict[str, Any]] = DEFAULT_EVALUATION_ITEMS,
) -> EvaluationResult:
    """Create the Opik dataset and evaluate the agent against its items."""
    app_settings = settings or get_settings()
    configure_opik(app_settings.opik)

    client = Opik(
        project_name=app_settings.opik.project_name,
        workspace=app_settings.opik.require_workspace(),
        api_key=app_settings.opik.require_api_key(),
    )
    dataset = client.get_or_create_dataset(
        name=app_settings.evaluation.dataset_name,
        description="Small regression dataset for the Second Brain agentic RAG app.",
        project_name=app_settings.opik.project_name,
    )
    dataset.insert(list(items))

    return evaluate(
        dataset=dataset,
        task=AgentEvaluationTask(agent),
        scoring_metrics=build_evaluation_metrics(app_settings),
        experiment_name_prefix=app_settings.evaluation.experiment_name_prefix,
        experiment_config={
            "agent_model_id": app_settings.openai.model_id,
            "retriever_k": app_settings.rag.retriever_k,
        },
        experiment_tags=["agentic-rag", "second-brain"],
        task_threads=1,
    )
