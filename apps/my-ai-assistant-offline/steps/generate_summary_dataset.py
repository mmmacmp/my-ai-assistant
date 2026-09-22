from typing import Annotated

from zenml import step

from my_ai_assistant_offline.domain.dataset import InstructDataset
from my_ai_assistant_offline.domain.document import Document
from my_ai_assistant_offline.application.dataset.summarization_dataset import (
    SummarizationDatasetGenerator,
)


@step
def generate_summary_dataset(
    documents: list[Document],
    augmentation_loops: int = 4,
) -> Annotated[InstructDataset, "summary_dataset"]:
    generator = SummarizationDatasetGenerator(
        min_document_length=50,
        min_quality_score=0.3,
        max_summary_characters=256,
        max_summary_length_factor=2.0,
    )

    return generator.generate(
        documents=documents,
        augmentation_loops=augmentation_loops,
        val_split_ratio=0.1,
        test_split_ratio=0.1,
        seed=42,
    )