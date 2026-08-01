

from my_ai_assistant.domain.document import Document


class HeuristicQualityAgent:

    def __call__(self,
                 documents: list[Document]
                 ) -> list[Document]:
        
        return [self._score_document(document) for document in documents]

    def _score_document(self,
                        document:Document
                        ) -> Document:
        
        content_len = len(document.content)
        urls_len = sum(len(url) for url in document.child_urls)

        if content_len == 0:
            return document.add_quality_score(0.0)

        url_content_ratio  = urls_len / content_len

        if url_content_ratio >= 0.7:
            return document.add_quality_score(0.0)
        elif url_content_ratio >= 0.5:
            return document.add_quality_score(0.2)
        else:
            return document


class QualityScoreAgent:
    """Score uncertain documents using an LLM or mock response."""

    def __init__(self, mock: bool = True) -> None:
        self.mock = mock

    def __call__(
        self,
        documents: list[Document],
    ) -> list[Document]:
        if not self.mock:
            raise NotImplementedError(
                "Real LLM scoring is not implemented yet."
            )

        return [
            document
            if document.content_quality_score is not None
            else document.add_quality_score(0.5)
            for document in documents
        ]