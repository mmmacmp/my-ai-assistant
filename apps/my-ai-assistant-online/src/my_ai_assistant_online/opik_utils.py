"""Privacy-aware Opik configuration for the online application."""

import opik
from opik import config as opik_config

from my_ai_assistant_online.config import OpikSettings


def configure_opik(settings: OpikSettings) -> None:
    """Configure tracing in memory before the first tracked function runs."""
    if not settings.enabled:
        opik.set_tracing_active(False)
        return

    opik_config.update_session_config("api_key", settings.require_api_key())
    opik_config.update_session_config("workspace", settings.require_workspace())
    opik_config.update_session_config("project_name", settings.project_name)
    opik.set_tracing_active(True)
