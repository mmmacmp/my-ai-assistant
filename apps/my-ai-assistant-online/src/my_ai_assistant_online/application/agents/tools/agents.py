from typing import Any

import opik
from opik import opik_context
from smolagents import LiteLLMModel, MultiStepAgent, ToolCallingAgent

from my_ai_assistant_online.application.agents.tools.mongodb_retriever import (
    MongoDBRetrieverTool,
)
from my_ai_assistant_online.application.agents.tools.summarizer import SummarizerTool
from my_ai_assistant_online.application.agents.tools.what_can_i_do import what_can_i_do
from my_ai_assistant_online.application.rag.types import (
    EmbeddingModelType,
    RetrieverType,
)
from my_ai_assistant_online.config import OnlineSettings, get_settings
from my_ai_assistant_online.opik_utils import configure_opik


class AgentWrapper:
    """Thin composition wrapper around SmolAgents' MultiStepAgent that adds
    Opik tracing on every .run() call."""

    def __init__(self, agent: MultiStepAgent) -> None:
        self.__agent = agent

    @property
    def input_messages(self) -> list[dict]:
        return self.__agent.input_messages

    @property
    def agent_name(self) -> str:
        return self.__agent.agent_name

    @property
    def name(self) -> str | None:
        """Expose the name expected by SmolAgents' built-in Gradio UI."""
        return getattr(self.__agent, "name", None)

    @property
    def max_steps(self) -> int:
        return self.__agent.max_steps

    @classmethod
    def build(
        cls,
        collection_name: str,
        embedding_model_id: str,
        embedding_model_type: EmbeddingModelType,
        retriever_type: RetrieverType,
        *,
        mongodb_uri: str,
        database_name: str,
        openai_api_key: str,
        openai_model_id: str,
        retriever_k: int,
        retriever_device: str,
        max_steps: int,
    ) -> "AgentWrapper":
        retriever_tool = MongoDBRetrieverTool(
            collection_name=collection_name,
            embedding_model_id=embedding_model_id,
            embedding_model_type=embedding_model_type,
            retriever_type=retriever_type,
            k=retriever_k,
            device=retriever_device,
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            openai_api_key=openai_api_key,
        )
        summarizer_tool = SummarizerTool(
            model_id=f"openai/{openai_model_id}",
            api_key=openai_api_key,
        )

        model = LiteLLMModel(
            model_id=f"openai/{openai_model_id}",
            api_base="https://api.openai.com/v1",
            api_key=openai_api_key,
        )

        agent = ToolCallingAgent(
            tools=[what_can_i_do, retriever_tool, summarizer_tool],
            model=model,
            max_steps=max_steps,
            verbosity_level=2,
        )

        return cls(agent)

    @opik.track(
        name="Agent.run",
        capture_input=False,
        capture_output=False,
        ignore_arguments=["self"],
    )
    def run(self, task: str, **kwargs: Any) -> Any:
        result = self.__agent.run(task, **kwargs)

        model = self.__agent.model
        metadata = {
            "system_prompt": self.__agent.system_prompt,
            "tools": list(self.__agent.tools.keys()),
            "model_id": model.model_id,
            "input_token_count": getattr(model, "last_input_token_count", None),
            "output_token_count": getattr(model, "last_output_token_count", None),
            "task_character_count": len(task),
        }
        if hasattr(self.__agent, "step_number"):
            metadata["step_number"] = self.__agent.step_number

        opik_context.update_current_trace(tags=["agent"], metadata=metadata)

        return result


def get_agent(
    collection_name: str | None,
    embedding_model_id: str,
    embedding_model_type: EmbeddingModelType,
    retriever_type: RetrieverType,
    *,
    settings: OnlineSettings | None = None,
) -> AgentWrapper:
    app_settings = settings or get_settings()
    configure_opik(app_settings.opik)

    return AgentWrapper.build(
        collection_name=(collection_name or app_settings.mongodb.rag_collection_name),
        embedding_model_id=embedding_model_id,
        embedding_model_type=embedding_model_type,
        retriever_type=retriever_type,
        mongodb_uri=app_settings.mongodb.require_uri(),
        database_name=app_settings.mongodb.database_name,
        openai_api_key=app_settings.openai.require_api_key(),
        openai_model_id=app_settings.openai.model_id,
        retriever_k=app_settings.rag.retriever_k,
        retriever_device=app_settings.rag.device,
        max_steps=app_settings.agent.max_steps,
    )
