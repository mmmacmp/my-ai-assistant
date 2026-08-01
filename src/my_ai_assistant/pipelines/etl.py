


from pathlib import Path

from my_ai_assistant.steps.add_quality_score import add_quality_score
from my_ai_assistant.steps.crawl_links import crawl_links
from my_ai_assistant.steps.ingest_to_mongodb import ingest_to_mongodb
from my_ai_assistant.steps.save_documents import save_documents_to_disk
from zenml import pipeline

from my_ai_assistant.steps.extract_document import extract_documents


@pipeline
def etl_pipeline(data_directory: Path) -> None:
    parent_documents = extract_documents(data_directory)
    crawled_documents = crawl_links(
        documents=parent_documents,
        max_concurrent_requests=2,
    )
    scored_documents = add_quality_score(crawled_documents)
    # saved_paths = save_documents_to_disk(scored_documents)
    ingest_to_mongodb(
    documents=scored_documents,
    collection_name="raw_documents",
    clear_collection=True,
)


if __name__ == "__main__":
    etl_pipeline(data_directory=Path("data/raw"))

    # from zenml.client import Client
    # import json
    
    # artifact = Client().get_artifact_version(
    #     "bb8a5eda-9f41-4fbd-9b8c-26daa5875528"
    # )

    # documents = artifact.load()

    # json_data = [
    #     document.model_dump(mode="json")
    #     for document in documents
    # ]

    # output_path = Path("data/output/crawled_documents.json")
    # output_path.parent.mkdir(parents=True, exist_ok=True)

    # output_path.write_text(
    #     json.dumps(
    #         json_data,
    #         indent=2,
    #         ensure_ascii=False,
    #     ),
    #     encoding="utf-8",
    # )

    # print(f"Saved to: {output_path.resolve()}")