from pathlib import Path
from typing import Annotated

from zenml import step

from my_ai_assistant.domain.dataset import InstructDataset


@step
def save_dataset(
    dataset: InstructDataset,
    output_directory: str = "data/processed",
) -> Annotated[dict[str, str], "dataset_paths"]:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": dataset.train,
        "validation": dataset.validation,
        "test": dataset.test,
    }

    saved_paths: dict[str, str] = {}

    for split_name, samples in splits.items():
        file_path = output_path / f"{split_name}.jsonl"

        lines = [
            sample.model_dump_json()
            for sample in samples
        ]

        file_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        saved_paths[split_name] = file_path.as_posix()

    return saved_paths