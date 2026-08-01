

from zenml import pipeline

from my_ai_assistant.steps.generate_summary_dataset import generate_summary_dataset
from my_ai_assistant.steps.create_histograms import create_histograms
from my_ai_assistant.steps.fetch_from_mongodb import (
    fetch_from_mongodb,
)
from my_ai_assistant.steps.save_dataset import save_dataset


@pipeline
def dataset_generation():
    documents = fetch_from_mongodb()
    create_histograms(documents)
    summary_dataset = generate_summary_dataset(
    documents=documents,
    augmentation_loops=4,
)

    dataset_paths = save_dataset(
        dataset=summary_dataset,
        output_directory="data/processed",
    )


if __name__ == "__main__":
    dataset_generation()