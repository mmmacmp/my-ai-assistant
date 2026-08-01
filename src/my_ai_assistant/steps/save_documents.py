

from pathlib import Path
from typing import Annotated

from my_ai_assistant.domain.document import Document
from zenml import step

@step
def save_documents_to_disk(documents: list[Document],
                           outputDir: str = "data/processed") -> Annotated[list[str], "saved_document_paths"]:
    dir = Path(outputDir)
    dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    for document in documents:
        output_path = dir / f"{document.id}.json"
        output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

        saved_paths.append(output_path.as_posix())
    return saved_paths
    