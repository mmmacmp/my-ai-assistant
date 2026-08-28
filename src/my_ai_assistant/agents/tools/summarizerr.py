from typing import cast

from litellm import completion
from litellm.types.utils import Choices, ModelResponse
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
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.max_characters = max_characters
        self.mock = mock

    def forward(self, text: str) -> str:
        if self.mock:
            return text[: self.max_characters]

        prompt = f"Summarize this document in max {self.max_characters} characters:\n\n{text}".strip()

        response = cast(
            ModelResponse,
            completion(
                model=self.model_id,
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