from ast import Try
import os
import logging
from typing import Any

from openai import OpenAI

from my_ai_assistant.config.config import Settings

logger = logging.getLogger("__base__")
class SimpleSummarizationAgent:
    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        base_url: str | None = Settings.HUGGINGFACE_DEDICATED_ENDPOINT,
        api_key: str | None = Settings.HUGGINGFACE_ACCESS_TOKEN,
        max_characters: int = 128,
        mock: bool = False,
        max_concurrent_requests: int = 4,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.max_characters = max_characters
        self.mock = mock
        self.max_concurrent_requests = max_concurrent_requests
        self.client = OpenAI(
                                        # This is the default and can be omitted
                                        api_key=api_key
                                    )
        

    def __call__(self, text : str, chunks : list[str]) -> list[str]:
        if self.mock:
            fake_summary = '''This is a fake summary \n'''
            new_chunks = [fake_summary + chunk
                          for chunk in chunks]
            return new_chunks
        else:
            try:
                summarization = self._summarize(text)
                if summarization is not None:
                    new_chunks = [summarization + "\n" + chunk
                        for chunk in chunks]
                    return new_chunks
                return chunks
            except Exception as error:
                logger.warning(
                                "Document summarization failed: %s",
                                error,
                            )
                return chunks


    def _summarize(self, text : str) -> str|None:
        real_prompt = f'''You are a document summarization assistant.
        
                                    Summarize the following document in a concise way that preserves the most important information for future retrieval and search.
        
                                    Requirements:
        
                                    Return only the summary in plain text.
                                    Do not include explanations, headings, labels, or commentary.
                                    The summary must not exceed {self.max_characters} characters.
                                    Prioritize key entities, concepts, facts, topics, relationships, and terminology that would help retrieve this document later.
                                    Remove repetition, filler, and low-value details.
                                    Do not add information that is not present in the document.
        
                                    Document:
                                    {text}'''
        response = self.client.chat.completions.create(
                                                                    model=self.model_id,
                                                                    messages=[
                                                                        {
                                                                            "role": "system",
                                                                            "content": real_prompt,
                                                                        }
                                                                    ],
                                                                    stream=False,
                                                                    temperature=0,
                                                                )
        if response.choices and response.choices[0].message.content:
           return response.choices[0].message.content
        else: 
            return None


        