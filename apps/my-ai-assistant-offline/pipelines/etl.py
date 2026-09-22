from pathlib import Path

from steps.add_quality_score import add_quality_score
from steps.crawl_links import crawl_links
from steps.ingest_to_mongodb import ingest_to_mongodb
from zenml import pipeline

from my_ai_assistant_offline.config import get_settings
from steps.extract_document import read_notion_documents
from steps.save_documents import save_documents_to_disk


@pipeline
def etl_pipeline(
    data_directory: Path,
    quality_agent_model_id: str = "gpt-4o-mini",
    quality_agent_mock: bool = False,
    crawl_max_workers: int = 4,
    quality_max_workers: int = 1,
    raw_collection_name: str = "raw_documents",
) -> None:
    parent_documents = read_notion_documents(data_directory)
    crawled_documents = crawl_links(
        documents=parent_documents,
        max_concurrent_requests=crawl_max_workers,
    )
    scored_documents = add_quality_score(
        crawled_documents,
        model_id=quality_agent_model_id,
        mock=quality_agent_mock,
        max_workers=quality_max_workers,
    )
    save_documents_to_disk(scored_documents, "data/output/scored_documents", 20)
    ingest_to_mongodb(
        documents=scored_documents,
        collection_name=raw_collection_name,
        clear_collection=True,
    )


if __name__ == "__main__":
    settings = get_settings()
    etl_pipeline(
        data_directory=Path("data/notion"),
        quality_agent_model_id=settings.openai.model_id,
        raw_collection_name=settings.mongodb.raw_collection_name,
    )

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
