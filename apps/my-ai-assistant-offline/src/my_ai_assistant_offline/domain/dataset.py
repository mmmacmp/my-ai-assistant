import random

from pydantic import BaseModel, Field

class InstructDatasetSample(BaseModel):
    instruction: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class InstructDataset(BaseModel):
    train: list[InstructDatasetSample]
    validation: list[InstructDatasetSample]
    test: list[InstructDatasetSample]

    val_split_ratio: float
    test_split_ratio: float
    seed: int | None = None

    @classmethod
    def from_samples(cls,
                     samples: list[InstructDatasetSample],
                     val_split_ratio: float,
                     test_split_ratio: float,
                     seed: int | None = None,) -> "InstructDataset":
        if not samples:
            raise ValueError("At least one sample is required")

        if not 0.0 <= val_split_ratio < 1.0:
            raise ValueError(
                "val_split_ratio must be between 0 and 1"
            )

        if not 0.0 <= test_split_ratio < 1.0:
            raise ValueError(
                "test_split_ratio must be between 0 and 1"
            )

        if val_split_ratio + test_split_ratio >= 1.0:
            raise ValueError(
                "Validation and test ratios must leave room for training"
            )

        shuffled_samples = samples.copy()

        random_generator = random.Random(seed)
        random_generator.shuffle(shuffled_samples)

        total_samples = len(shuffled_samples)

        train_end = int(total_samples * (1-val_split_ratio-test_split_ratio))

        validation_end = int(total_samples * (1 - test_split_ratio) )

        train_samples = shuffled_samples[:train_end]

        validation_samples = shuffled_samples[train_end:validation_end]

        test_samples = shuffled_samples[validation_end:]

        if not train_samples:
            raise ValueError("Train split is empty")

        if not validation_samples:
            raise ValueError("Validation split is empty")

        if not test_samples:
            raise ValueError("Test split is empty")

        return cls(
            train=train_samples,
            validation=validation_samples,
            test=test_samples,
            val_split_ratio=val_split_ratio,
            test_split_ratio=test_split_ratio,
            seed=seed,
        )

