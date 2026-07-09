"""HotpotQA dataset loading utilities."""

from __future__ import annotations

from datasets import load_dataset

from config import DATASET_CONFIG, DATASET_NAME, RANDOM_SEED, SAMPLE_SIZE


def load_hotpotqa(split: str = "validation"):
    """Load HotpotQA and return a deterministic evaluation subset."""

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)

    if SAMPLE_SIZE is not None:
        sample_size = min(SAMPLE_SIZE, len(dataset))
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(sample_size))

    return dataset
