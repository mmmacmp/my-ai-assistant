from collections.abc import Callable

from my_ai_assistant_offline.application.agents.summarization import SummarizationAgent
from my_ai_assistant_offline.config import get_settings
from my_ai_assistant_offline.domain.dataset import (
    InstructDataset,
    InstructDatasetSample,
)
from my_ai_assistant_offline.domain.document import Document
from my_ai_assistant_offline.infrastructure.mongodb import MongoDBService


DocumentFilter = Callable[[Document], bool]


class SummarizationDatasetGenerator:
    def __init__(
        self,
        min_document_length: int = 50,
        min_quality_score: float = 0.3,
        max_summary_characters: int = 256,
        max_summary_length_factor: float = 2.0,
    ) -> None:
        if min_document_length < 1:
            raise ValueError("min_document_length must be at least 1")

        if not 0.0 <= min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be between 0 and 1")

        if max_summary_characters < 1:
            raise ValueError("max_summary_characters must be at least 1")

        if max_summary_length_factor <= 0:
            raise ValueError("max_summary_length_factor must be positive")

        self.min_document_length = min_document_length
        self.min_quality_score = min_quality_score
        self.max_summary_characters = max_summary_characters
        self.max_summary_length_factor = max_summary_length_factor

        self.pregeneration_filters: list[DocumentFilter] = [
            lambda document: (
                len(document.content.strip()) >= self.min_document_length
            ),
            lambda document: (
                document.content_quality_score is not None
                and document.content_quality_score >= self.min_quality_score
            ),
        ]

        self.postgeneration_filters: list[DocumentFilter] = [
            lambda document: (
                document.summary is not None
                and len(document.summary.strip())
                <= int(self.max_summary_characters * self.max_summary_length_factor)
            ),
        ]

    @staticmethod
    def filter_documents(
        documents: list[Document],
        filters: list[DocumentFilter],
    ) -> list[Document]:
        filtered_documents = documents

        for document_filter in filters:
            filtered_documents = [
                document for document in filtered_documents if document_filter(document)
            ]

        return filtered_documents

    def generate_augmented_summaries(
        self,
        documents: list[Document],
        augmentation_loops: int = 4,
    ) -> list[Document]:
        if augmentation_loops < 1:
            raise ValueError("augmentation_loops must be at least 1")

        agent = SummarizationAgent(
            max_characters=self.max_summary_characters,
            mock=True,
        )

        augmented_documents: list[Document] = []

        for loop_index in range(augmentation_loops):
            temperature = loop_index * 0.5 / augmentation_loops

            summarized_documents = agent(
                documents=documents,
                temperature=temperature,
            )

            augmented_documents.extend(summarized_documents)

        return augmented_documents

    def create_dataset_samples(
        self,
        documents: list[Document],
    ) -> list[InstructDatasetSample]:
        samples: list[InstructDatasetSample] = []

        for document in documents:
            if document.summary is None:
                raise ValueError(f"Document {document.id} has no summary")

            instruction = (
                "Summarize the following document clearly and concisely:"
                f"\n\n{document.content}"
            )

            sample = InstructDatasetSample(
                instruction=instruction,
                answer=document.summary,
            )

            samples.append(sample)

        return samples

    def generate(
        self,
        documents: list[Document],
        augmentation_loops: int = 4,
        val_split_ratio: float = 0.1,
        test_split_ratio: float = 0.1,
        seed: int | None = 42,
    ) -> InstructDataset:
        accepted_documents = self.filter_documents(
            documents=documents,
            filters=self.pregeneration_filters,
        )

        augmented_documents = self.generate_augmented_summaries(
            documents=accepted_documents,
            augmentation_loops=augmentation_loops,
        )

        valid_summaries = self.filter_documents(
            documents=augmented_documents,
            filters=self.postgeneration_filters,
        )

        samples = self.create_dataset_samples(
            documents=valid_summaries,
        )

        return InstructDataset.from_samples(
            samples=samples,
            val_split_ratio=val_split_ratio,
            test_split_ratio=test_split_ratio,
            seed=seed,
        )


if __name__ == "__main__":
    settings = get_settings()
    service = MongoDBService(
        model=Document,
        collection_name=settings.mongodb.raw_collection_name,
        mongodb_uri=settings.mongodb.require_uri(),
        database_name=settings.mongodb.database_name,
    )

    try:
        documents = service.fetch_documents()
    finally:
        service.close()

    generator = SummarizationDatasetGenerator(
        min_document_length=50,
        min_quality_score=0.3,
        max_summary_characters=256,
        max_summary_length_factor=2.0,
    )

    dataset = generator.generate(
        documents=documents,
        augmentation_loops=4,
        val_split_ratio=0.1,
        test_split_ratio=0.1,
        seed=42,
    )

    print("Train:", len(dataset.train))
    print("Validation:", len(dataset.validation))
    print("Test:", len(dataset.test))
