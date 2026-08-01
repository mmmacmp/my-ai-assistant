


from typing import Annotated

from my_ai_assistant.agents.quality import (
    HeuristicQualityAgent,
    QualityScoreAgent,
)
from my_ai_assistant.domain.document import Document
from zenml import get_step_context, step

@step
def add_quality_score(documents: list[Document]
                      ) -> Annotated[list[Document], "scored_documents"]:
    heuristic_agent = HeuristicQualityAgent()
    heuristic_results = heuristic_agent(documents)

    scored_by_heuristics = [
        document
        for document in heuristic_results
        if document.content_quality_score is not None
    ]

    documents_for_llm = [
        document
        for document in heuristic_results
        if document.content_quality_score is None
    ]

    llm_agent = QualityScoreAgent(mock=True)
    scored_by_llm = llm_agent(documents_for_llm)

    scored_by_id = {
        document.id: document
        for document in scored_by_heuristics + scored_by_llm
    }

    scored_documents = [
        scored_by_id[document.id]
        for document in documents
    ]

    scored_count = sum(
        document.content_quality_score is not None
        for document in scored_documents
    )

    get_step_context().add_output_metadata(
    output_name="scored_documents",
    metadata={
        "total_documents": len(scored_documents),
        "scored_by_heuristics": len(scored_by_heuristics),
        "scored_by_mock_llm": len(scored_by_llm),
        "still_unscored": sum(
            document.content_quality_score is None
            for document in scored_documents
        ),
    },
)

    return scored_documents
    