from unittest.mock import MagicMock

from my_ai_assistant_online.application import evaluation_runner as runner_module
from my_ai_assistant_online.config import OnlineSettings
from my_ai_assistant_online.tools import evaluate as evaluate_cli


def test_run_evaluation_connects_dataset_task_and_metrics(monkeypatch) -> None:
    agent = object()
    dataset = MagicMock()
    client = MagicMock()
    client.get_or_create_dataset.return_value = dataset
    opik_class = MagicMock(return_value=client)
    task = object()
    task_class = MagicMock(return_value=task)
    metrics = [object(), object()]
    build_metrics = MagicMock(return_value=metrics)
    evaluation_result = object()
    evaluate = MagicMock(return_value=evaluation_result)
    configure_opik = MagicMock()
    monkeypatch.setattr(runner_module, "Opik", opik_class)
    monkeypatch.setattr(runner_module, "AgentEvaluationTask", task_class)
    monkeypatch.setattr(runner_module, "build_evaluation_metrics", build_metrics)
    monkeypatch.setattr(runner_module, "evaluate", evaluate)
    monkeypatch.setattr(runner_module, "configure_opik", configure_opik)
    settings = OnlineSettings(
        _env_file=None,
        openai={"api_key": "openai-test-secret", "model_id": "gpt-test"},
        opik={
            "enabled": True,
            "api_key": "opik-test-secret",
            "workspace": "test-workspace",
            "project_name": "test-project",
        },
        rag={"retriever_k": 7},
        evaluation={
            "dataset_name": "test-dataset",
            "experiment_name_prefix": "test-experiment",
        },
    )
    items = [{"input": "What is RAG?"}]

    result = runner_module.run_evaluation(agent, settings=settings, items=items)

    assert result is evaluation_result
    configure_opik.assert_called_once_with(settings.opik)
    opik_class.assert_called_once_with(
        project_name="test-project",
        workspace="test-workspace",
        api_key="opik-test-secret",
    )
    client.get_or_create_dataset.assert_called_once_with(
        name="test-dataset",
        description="Small regression dataset for the Second Brain agentic RAG app.",
        project_name="test-project",
    )
    dataset.insert.assert_called_once_with(items)
    task_class.assert_called_once_with(agent)
    build_metrics.assert_called_once_with(settings)
    evaluate.assert_called_once_with(
        dataset=dataset,
        task=task,
        scoring_metrics=metrics,
        experiment_name_prefix="test-experiment",
        experiment_config={"agent_model_id": "gpt-test", "retriever_k": 7},
        experiment_tags=["agentic-rag", "second-brain"],
        task_threads=1,
    )


def test_evaluation_cli_builds_agent_once(monkeypatch) -> None:
    agent = object()
    build_agent = MagicMock(return_value=agent)
    run_evaluation = MagicMock()
    monkeypatch.setattr(evaluate_cli, "run_evaluation", run_evaluation)

    evaluate_cli.run(build_agent)

    build_agent.assert_called_once_with()
    run_evaluation.assert_called_once_with(agent)
