from unittest.mock import MagicMock

import pytest
from smolagents.tools import BaseTool

from my_ai_assistant_online.application.agents.tools import agents as agents_module
from my_ai_assistant_online.application.agents.tools.what_can_i_do import (
    what_can_i_do,
)
from my_ai_assistant_online.config import OnlineSettings
from my_ai_assistant_online.tools import app as app_module


def test_what_can_i_do_is_a_smolagents_tool() -> None:
    assert isinstance(what_can_i_do, BaseTool)


def test_agent_wrapper_exposes_name_for_gradio() -> None:
    smol_agent = MagicMock()
    smol_agent.name = "second-brain-agent"

    agent = agents_module.AgentWrapper(smol_agent)

    assert agent.name == "second-brain-agent"


def test_ui_launches_gradio_without_running_a_cli_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = MagicMock()
    gradio_ui = MagicMock()

    monkeypatch.setattr(app_module, "build_agent", MagicMock(return_value=agent))
    gradio_ui_class = MagicMock(return_value=gradio_ui)
    monkeypatch.setattr(app_module, "GradioUI", gradio_ui_class)

    app_module.run(ui=True, query="unused", share=False)

    gradio_ui_class.assert_called_once_with(agent, reset_agent_memory=False)
    gradio_ui.launch.assert_called_once_with(share=False)
    agent.run.assert_not_called()


def test_get_agent_resolves_and_injects_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_agent = object()
    build = MagicMock(return_value=expected_agent)
    configure_opik = MagicMock()
    monkeypatch.setattr(agents_module.AgentWrapper, "build", build)
    monkeypatch.setattr(agents_module, "configure_opik", configure_opik)
    settings = OnlineSettings(
        _env_file=None,
        openai={"api_key": "openai-test-secret", "model_id": "gpt-test"},
        mongodb={
            "uri": "mongodb://explicit.invalid",
            "database_name": "assistant_test",
            "rag_collection_name": "rag_test",
        },
        rag={"retriever_k": 7, "device": "cuda"},
        agent={"max_steps": 4},
    )

    result = agents_module.get_agent(
        collection_name=None,
        embedding_model_id="embedding-test",
        embedding_model_type="openai",
        retriever_type="contextual",
        settings=settings,
    )

    assert result is expected_agent
    configure_opik.assert_called_once_with(settings.opik)
    build.assert_called_once_with(
        collection_name="rag_test",
        embedding_model_id="embedding-test",
        embedding_model_type="openai",
        retriever_type="contextual",
        mongodb_uri="mongodb://explicit.invalid",
        database_name="assistant_test",
        openai_api_key="openai-test-secret",
        openai_model_id="gpt-test",
        retriever_k=7,
        retriever_device="cuda",
        max_steps=4,
    )
