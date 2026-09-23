"""Text summarization tool used by the assistant."""

from typing import cast

import opik
from litellm import completion
from litellm.types.utils import Choices, ModelResponse
from opik import opik_context
from smolagents.tools import Tool


class SummarizerTool(Tool):
    name = "summarizer"
    description = """Use this tool to summarize a piece of text. Especially useful when you have
    retrieved long document content and need a concise TL;DR before answering the user.""".strip()
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to summarize.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        model_id: str = "openai/gpt-4o-mini",
        max_characters: int = 512,
        mock: bool = False,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        if not mock and api_key is None:
            raise RuntimeError("An API key is required for real summarization")

        self.model_id = model_id
        self.max_characters = max_characters
        self.mock = mock
        self.api_key = api_key

    @opik.track(
        name="SummarizerTool.forward",
        type="tool",
        capture_input=False,
        capture_output=False,
        ignore_arguments=["self"],
    )
    def forward(self, text: str) -> str:
        opik_context.update_current_span(
            metadata={
                "model_id": self.model_id,
                "max_characters": self.max_characters,
                "input_character_count": len(text),
                "mock": self.mock,
            }
        )

        if self.mock:
            return text[: self.max_characters]

        prompt = f"Summarize this document in max {self.max_characters} characters:\n\n{text}".strip()

        response = cast(
            ModelResponse,
            completion(
                model=self.model_id,
                api_key=self.api_key,
                messages=[{"role": "user", "content": prompt}],
            ),
        )

        if not response.choices:
            return "The summarizer failed"

        choice = cast(Choices, response.choices[0])
        content = choice.message.content
        if content is None:
            return "The summarizer failed"

        return content
