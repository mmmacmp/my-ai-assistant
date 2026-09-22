from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    id: str
    url: str
    title: str
    properties: dict[str, Any]


class Document(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    metadata: DocumentMetadata
    parent_metadata: DocumentMetadata | None = None
    content_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    content: str
    summary: str | None = None
    child_urls: list[str] = Field(default_factory=list)

    def add_quality_score(self, score: float) -> "Document":
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Content quality score must be between 0 and 1."
            )

        return self.model_copy(
            update={"content_quality_score": score}
        )

    def add_summary(self, summary:str) -> "Document":
        cleaned_summary = summary.strip()

        if not cleaned_summary:
            raise ValueError("Summary cannot be empty")
        
        return self.model_copy(
            update={"summary": cleaned_summary}
        )