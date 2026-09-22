from typing import Annotated

from zenml import get_step_context, step

from my_ai_assistant_offline.domain.document import Document


@step
def filter_by_quality(documents: list[Document], 
                      content_quality_score_threshold: float
                      ) -> Annotated[list[Document], "filtered_documets"]:

    filtered_docs = [doc
                     for doc in documents
                     if doc.content_quality_score is not None and doc.content_quality_score > content_quality_score_threshold]

    get_step_context().add_output_metadata(output_name="filtered_documets",
                                           metadata={
                                               "filtered_docs_count": len(filtered_docs),
                                               "quality_threshold": content_quality_score_threshold
                                           })

    return filtered_docs