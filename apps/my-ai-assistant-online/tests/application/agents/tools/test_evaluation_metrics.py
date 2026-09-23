from unittest.mock import MagicMock

from my_ai_assistant_online.application import evaluation_metrics as metrics_module
from my_ai_assistant_online.config import OnlineSettings


def test_build_evaluation_metrics_uses_one_configured_judge_model(
    monkeypatch,
) -> None:
    judge_model = object()
    relevance_metric = object()
    hallucination_metric = object()
    density_metric = object()
    model_class = MagicMock(return_value=judge_model)
    answer_relevance = MagicMock(return_value=relevance_metric)
    hallucination = MagicMock(return_value=hallucination_metric)
    summary_density = MagicMock(return_value=density_metric)
    configure_opik = MagicMock()
    monkeypatch.setattr(metrics_module, "LiteLLMChatModel", model_class)
    monkeypatch.setattr(metrics_module, "AnswerRelevance", answer_relevance)
    monkeypatch.setattr(metrics_module, "Hallucination", hallucination)
    monkeypatch.setattr(metrics_module, "SummaryDensityHeuristic", summary_density)
    monkeypatch.setattr(metrics_module, "configure_opik", configure_opik)
    settings = OnlineSettings(
        _env_file=None,
        openai={"api_key": "openai-test-secret", "model_id": "gpt-test"},
        opik={
            "enabled": True,
            "api_key": "opik-test-secret",
            "workspace": "test-workspace",
            "project_name": "evaluation-test",
        },
    )

    metrics = metrics_module.build_evaluation_metrics(settings)

    assert metrics == [relevance_metric, hallucination_metric, density_metric]
    configure_opik.assert_called_once_with(settings.opik)
    model_class.assert_called_once_with(
        model_name="openai/gpt-test",
        api_key="openai-test-secret",
        temperature=0,
        track=True,
    )
    expected_options = {
        "model": judge_model,
        "track": True,
        "project_name": "evaluation-test",
    }
    answer_relevance.assert_called_once_with(
        require_context=False,
        **expected_options,
    )
    hallucination.assert_called_once_with(**expected_options)
    summary_density.assert_called_once_with(
        track=True,
        project_name="evaluation-test",
    )


def test_build_evaluation_metrics_omits_project_when_tracking_is_disabled(
    monkeypatch,
) -> None:
    model_class = MagicMock(return_value=object())
    answer_relevance = MagicMock(return_value=object())
    hallucination = MagicMock(return_value=object())
    summary_density = MagicMock(return_value=object())
    monkeypatch.setattr(metrics_module, "LiteLLMChatModel", model_class)
    monkeypatch.setattr(metrics_module, "AnswerRelevance", answer_relevance)
    monkeypatch.setattr(metrics_module, "Hallucination", hallucination)
    monkeypatch.setattr(metrics_module, "SummaryDensityHeuristic", summary_density)
    monkeypatch.setattr(metrics_module, "configure_opik", MagicMock())
    settings = OnlineSettings(
        _env_file=None,
        openai={"api_key": "openai-test-secret"},
        opik={"enabled": False},
    )

    metrics_module.build_evaluation_metrics(settings)

    answer_relevance.assert_called_once_with(
        require_context=False,
        model=model_class.return_value,
        track=False,
    )
    hallucination.assert_called_once_with(
        model=model_class.return_value,
        track=False,
    )
    summary_density.assert_called_once_with(track=False)
