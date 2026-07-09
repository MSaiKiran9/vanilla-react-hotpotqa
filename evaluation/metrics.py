"""Answer-level metrics for HotpotQA evaluation."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lower text and remove articles, punctuation, and extra whitespace."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_scores(prediction: str, gold: str) -> tuple[float, float, float]:
    """Return precision, recall, and token F1."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1
