from zenml import pipeline

from my_ai_assistant_offline.config import get_settings
from steps.generate_summary_dataset import generate_summary_dataset
from steps.create_histograms import create_histograms
from steps.fetch_from_mongodb import (
    fetch_from_mongodb,
)
from steps.save_dataset import save_dataset


@pipeline
def dataset_generation(
    collection_name: str = "raw_documents",
) -> None:
    documents = fetch_from_mongodb(collection_name=collection_name)
    create_histograms(documents)
    summary_dataset = generate_summary_dataset(
        documents=documents,
        augmentation_loops=4,
    )

    save_dataset(
        dataset=summary_dataset,
        output_directory="data/processed",
    )


if __name__ == "__main__":
    settings = get_settings()
    dataset_generation(
        collection_name=settings.mongodb.raw_collection_name,
    )
