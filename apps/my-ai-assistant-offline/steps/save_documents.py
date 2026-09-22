

from pathlib import Path
from typing import Annotated

from my_ai_assistant_offline.domain.document import Document
from zenml import step

@step
def save_documents_to_disk(
    documents: list[Document],
    outputDir: str = "data/processed",
    limit: int | None = None,
) -> Annotated[list[str], "saved_document_paths"]:
    directory = Path(outputDir)
    directory.mkdir(parents=True, exist_ok=True)

    documents_to_save = documents if limit is None else documents[:limit]

    saved_paths: list[str] = []

    for document in documents_to_save:
        output_path = directory / f"{document.id}.json"
        output_path.write_text(
            document.model_dump_json(indent=2),
            encoding="utf-8",
        )
        saved_paths.append(output_path.as_posix())

    return saved_paths
