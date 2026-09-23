from opik.evaluation.metrics import AnswerRelevance, BaseMetric, Hallucination
from opik.evaluation.models import LiteLLMChatModel

from my_ai_assistant_online.application.evaluation_heuristics import (
    SummaryDensityHeuristic,
)
from my_ai_assistant_online.config import OnlineSettings, get_settings
from my_ai_assistant_online.opik_utils import configure_opik


def build_evaluation_metrics(
    settings: OnlineSettings | None = None,
) -> list[BaseMetric]:
    """Build the LLM-as-a-judge metrics used for agent evaluation."""
    app_settings = settings or get_settings()
    configure_opik(app_settings.opik)

    judge_model = LiteLLMChatModel(
        model_name=f"openai/{app_settings.openai.model_id}",
        api_key=app_settings.openai.require_api_key(),
        temperature=0,
        track=app_settings.opik.enabled,
    )
    tracking_options = {
        "track": app_settings.opik.enabled,
    }
    if app_settings.opik.enabled:
        tracking_options["project_name"] = app_settings.opik.project_name

    judge_metric_options = {
        "model": judge_model,
        **tracking_options,
    }

    return [
        AnswerRelevance(require_context=False, **judge_metric_options),
        Hallucination(**judge_metric_options),
        SummaryDensityHeuristic(**tracking_options),
    ]
