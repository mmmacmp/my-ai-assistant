from typing import Any

from my_ai_assistant.agents.tools import what_can_i_do
from my_ai_assistant.agents.tools.config import settings
from my_ai_assistant.agents.tools.mongodb_retriever import MongoDBRetrieverTool
from my_ai_assistant.agents.tools.summarizerr import SummarizerTool
import opik
from opik import opik_context
from smolagents import LiteLLMModel, MultiStepAgent, ToolCallingAgent




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
    def max_steps(self) -> int:
        return self.__agent.max_steps

    @classmethod
    def build(
        cls,
        collection_name: str,
        embedding_model_id: str,
        embedding_model_type,
        retriever_type,
    ) -> "AgentWrapper":
        retriever_tool = MongoDBRetrieverTool(
            collection_name=collection_name,
            embedding_model_id=embedding_model_id,
            embedding_model_type=embedding_model_type,
            retriever_type=retriever_type,
            k=settings.RETRIEVER_K,
            device=settings.RETRIEVER_DEVICE,
        )
        summarizer_tool = SummarizerTool(model_id=f"openai/{settings.OPENAI_MODEL_ID}")

        model = LiteLLMModel(
            model_id=f"openai/{settings.OPENAI_MODEL_ID}",
            api_base="https://api.openai.com/v1",
            api_key=settings.OPENAI_API_KEY,
        )

        agent = ToolCallingAgent(
            tools=[what_can_i_do, retriever_tool, summarizer_tool],
            model=model,
            max_steps=settings.AGENT_MAX_STEPS,
            verbosity_level=2,
        )

        return cls(agent)

    @opik.track(name="Agent.run")
    def run(self, task: str, **kwargs: Any) -> Any:
        result = self.__agent.run(task, **kwargs)

        model = self.__agent.model
        metadata = {
            "system_prompt": self.__agent.system_prompt,
            "tools": list(self.__agent.tools.keys()),
            "model_id": model.model_id,
            "input_token_count": getattr(model, "last_input_token_count", None),
            "output_token_count": getattr(model, "last_output_token_count", None),
        }
        if hasattr(self.__agent, "step_number"):
            metadata["step_number"] = self.__agent.step_number

        opik_context.update_current_trace(tags=["agent"], metadata=metadata)

        return result


def get_agent(
    collection_name: str,
    embedding_model_id: str,
    embedding_model_type,
    retriever_type,
) -> AgentWrapper:
    return AgentWrapper.build(
        collection_name=collection_name,
        embedding_model_id=embedding_model_id,
        embedding_model_type=embedding_model_type,
        retriever_type=retriever_type,
    )