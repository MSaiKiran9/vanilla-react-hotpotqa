"""HotpotQA dataset loading utilities."""

from __future__ import annotations

from datasets import load_dataset

from config import RANDOM_SEED, SAMPLE_SIZE


# Current Hugging Face dataset repository
DATASET_REPO = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"


def load_hotpotqa(split: str = "validation", sample_size: int | None = SAMPLE_SIZE):
    """Load HotpotQA and return a deterministic evaluation subset."""

    dataset = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split=split,
    )

    if sample_size is not None:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(sample_size))

    return dataset
