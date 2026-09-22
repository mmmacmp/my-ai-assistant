import hashlib
import logging
from pathlib import Path
import re

from zenml import step

from my_ai_assistant_offline.domain.document import Document, DocumentMetadata


MARKDOWN_LINK_PATTERN = re.compile(
    r"\[[^\]]+\]\((https?://[^)\s]+)\)"
)


def extract_document(file_path: Path) -> Document:
    if not file_path.is_file():
        raise FileNotFoundError(f"Document no found: {file_path}")

    if file_path.suffix.lower() != ".md":
        raise ValueError(f"Expected a markdown file: {file_path}")

    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Document is empty: {file_path}")

    urls = MARKDOWN_LINK_PATTERN.findall(content)

    title = _extract_title(content, file_path)

    source_id = hashlib.sha256(
        str(file_path.resolve()).encode("utf-8")
    ).hexdigest()[:32]

    metadata = DocumentMetadata(id=source_id, title=title, url=file_path.as_posix(), properties={
        "file_name": file_path.name,
        "file_type": "markdown"
    }  )

    return Document(id=source_id,
                    content=content,
                    metadata=metadata,
                    child_urls=list(dict.fromkeys(urls)))


def _extract_title(content: str, file_path: Path) -> str:
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("# "):
            return stripped_line.removeprefix("# ").strip()

    return file_path.stem.strip()


@step
def extract_documents(data_directory: Path) -> list[Document]:
    if not data_directory.is_dir():
        raise NotADirectoryError(
            f"Data directory not found: {data_directory}"
        )

    markdown_files = sorted(data_directory.rglob("*.md"))

    if not markdown_files:
        raise ValueError(
            f"No Markdown documents found in: {data_directory}"
        )

    return [
        extract_document(file_path)
        for file_path in markdown_files
    ]

@step
def read_notion_documents(
    data_directory: Path,
) -> list[Document]:
    json_files = sorted(
        data_directory.glob("database_*/*.json")
    )

    if not json_files:
        raise ValueError(
            f"No Notion JSON documents found in {data_directory}"
        )

    return [
        Document.model_validate_json(
            file_path.read_text(encoding="utf-8")
        )
        for file_path in json_files
    ]

if __name__ == "__main__":
    file_path = Path("data/raw/first_note.md")

    document = extract_document(file_path)

    print(document.metadata)
