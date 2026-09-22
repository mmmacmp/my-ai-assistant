"""Create charts describing the document collection."""

from io import BytesIO
from typing import Annotated

import matplotlib.pyplot as plt
from PIL import Image
from zenml import ArtifactConfig, step

from my_ai_assistant_offline.domain.document import Document


@step
def create_histograms(
    documents: list[Document],
) -> Annotated[
    Image.Image,
    ArtifactConfig(name="histogram_chart"),
]:
    if not documents:
        raise ValueError("No documents were provided")

    content_lengths = [
        len(document.content)
        for document in documents
    ]

    quality_scores = [
        document.content_quality_score
        for document in documents
        if document.content_quality_score is not None
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    axes[0].hist(
        content_lengths,
        bins=min(10, len(content_lengths)),
    )
    axes[0].set_title("Document content lengths")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Documents")

    axes[1].hist(
        quality_scores,
        bins=5,
        range=(0.0, 1.0),
    )
    axes[1].set_title("Document quality scores")
    axes[1].set_xlabel("Quality score")
    axes[1].set_ylabel("Documents")

    figure.tight_layout()

    image_buffer = BytesIO()
    figure.savefig(
        image_buffer,
        format="png",
        dpi=150,
    )
    plt.close(figure)

    image_buffer.seek(0)
    histogram_image = Image.open(image_buffer).copy()
    image_buffer.close()

    return histogram_image