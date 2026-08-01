import asyncio

from my_ai_assistant.domain.document import Document


class SummarizationAgent:
    def __init__(self,
                 max_characters: int = 256,
                 max_concurrent_requests: int = 3,
                 mock: bool = True) -> None:
        if max_characters < 20:
            raise ValueError("max_characters must be at least 20")

        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requst must be at least 1")
        
        self.max_characters = max_characters
        self.max_concurrent_request = max_concurrent_requests
        self.mock = mock

    def __call__(self,
                 documents: list[Document],
                 temperature: float = 0.0,) -> list[Document]:
        
        return asyncio.run(
            self._summarize_batch(documents, temperature)
        )

    async def _summarize_batch(self, documents: list[Document], temperature: float) -> list[Document]:
        semaphores = asyncio.Semaphore(self.max_concurrent_request)

        tasks = [self._summarize_document(document, semaphores, temperature) for document in documents]

        return await asyncio.gather(*tasks)

    async def _summarize_document(self, document: Document, semaphore:asyncio.Semaphore, temperature: float) -> Document:
        async with semaphore:
            if not self.mock:
                raise NotImplementedError(
                    "Real LLM summarization is not implemented yet"
                )
            
            await asyncio.sleep(0)

            normalized_content = " ".join(
                document.content.split()
            )

            prefix = f"[MOCK temperature={temperature:.2f}] "

            available_characters = (
                self.max_characters - len(prefix)
            )

            mock_summary = (
                prefix
                + normalized_content[:available_characters]
            )

            return document.add_summary(mock_summary)

        




