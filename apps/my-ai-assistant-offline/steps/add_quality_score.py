from typing import Annotated

from my_ai_assistant_offline.application.agents.quality import (
    HeuristicQualityAgent,
    QualityScoreAgent,
)
from my_ai_assistant_offline.config import get_settings
from my_ai_assistant_offline.domain.document import Document
from zenml import get_step_context, step


@step
def add_quality_score(
    documents: list[Document],
    model_id: str = "gpt-4o-mini",
    mock: bool = False,
    max_workers: int = 4,
    max_retries: int = 1,
) -> Annotated[list[Document], "scored_documents"]:
    settings = get_settings()
    heuristic_agent = HeuristicQualityAgent()
    heuristic_results = heuristic_agent(documents)

    already_scored_count = sum(
        document.content_quality_score is not None for document in documents
    )

    scored_by_heuristics_count = sum(
        original.content_quality_score is None
        and scored.content_quality_score is not None
        for original, scored in zip(documents, heuristic_results, strict=True)
    )

    documents_for_llm = [
        document
        for document in heuristic_results
        if document.content_quality_score is None
    ]

    scored_by_llm: list[Document] = []
    if documents_for_llm:
        llm_agent = QualityScoreAgent(
            model_id=model_id,
            mock=mock,
            max_concurrent_requests=max_workers,
            max_retries=max_retries,
            api_key=None if mock else settings.openai.require_api_key(),
        )
        scored_by_llm = llm_agent(documents_for_llm)

    scored_by_llm_id = {document.id: document for document in scored_by_llm}
    scored_documents = [
        scored_by_llm_id.get(document.id, document) for document in heuristic_results
    ]

    llm_scored_count = sum(
        document.content_quality_score is not None for document in scored_by_llm
    )

    get_step_context().add_output_metadata(
        output_name="scored_documents",
        metadata={
            "total_documents": len(scored_documents),
            "already_scored": already_scored_count,
            "scored_by_heuristics": scored_by_heuristics_count,
            "sent_to_llm": len(documents_for_llm),
            "scored_by_llm": llm_scored_count,
            "llm_mode": "mock" if mock else "real",
            "still_unscored": sum(
                document.content_quality_score is None for document in scored_documents
            ),
        },
    )

    return scored_documents
