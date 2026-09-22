"""Simple document summarization for retrieval-time context."""

import logging

from openai import OpenAI


logger = logging.getLogger(__name__)


class SimpleSummarizationAgent:
    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_characters: int = 128,
        mock: bool = False,
        max_concurrent_requests: int = 4,
        client: OpenAI | None = None,
    ) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be at least 1")
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be at least 1")
        if not mock and client is None and api_key is None:
            raise RuntimeError(
                "An API key or configured client is required for real summarization"
            )

        self.model_id = model_id
        self.max_characters = max_characters
        self.mock = mock
        self.max_concurrent_requests = max_concurrent_requests
        self.client = client
        if self.client is None and not self.mock:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, text: str, chunks: list[str]) -> list[str]:
        if self.mock:
            fake_summary = "This is a fake summary \n"
            return [fake_summary + chunk for chunk in chunks]

        try:
            summarization = self._summarize(text)
            if summarization is not None:
                return [summarization + "\n" + chunk for chunk in chunks]
            return chunks
        except Exception as error:
            logger.warning(
                "Document summarization failed: %s",
                error,
            )
            return chunks

    def _summarize(self, text: str) -> str | None:
        if self.client is None:
            raise RuntimeError("The OpenAI client is not configured")

        prompt = f"""You are a document summarization assistant.

Summarize the following document in a concise way that preserves the most
important information for future retrieval and search.

Requirements:
- Return only the summary in plain text.
- Do not include explanations, headings, labels, or commentary.
- The summary must not exceed {self.max_characters} characters.
- Prioritize facts, concepts, entities, relationships, and terminology.
- Remove repetition and do not add information absent from the document.

Document:
{text}"""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": prompt}],
            stream=False,
            temperature=0,
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return None
