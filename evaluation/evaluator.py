"""Evaluation loop for the Vanilla ReAct baseline."""

from __future__ import annotations

from statistics import mean

from tqdm.auto import tqdm

from evaluation.metrics import exact_match, token_scores
from react.agent import ReActAgent
from retrieval.wikipedia import WikipediaRetriever
from utils.timer import Timer


def evaluate(model_name: str, loader, dataset, logger) -> dict[str, float | int | str]:
    """Evaluate one loaded model over a HotpotQA dataset."""

    wiki = WikipediaRetriever()
    agent = ReActAgent(loader, wiki)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    latencies = []
    iterations = []
    retrieval_failures = 0
    reasoning_failures = 0

    for example in tqdm(dataset, desc=f"Evaluating {model_name}"):
        question = example["question"]
        gold = example["answer"]

        with Timer() as timer:
            result = agent.answer(question)

        precision, recall, f1 = token_scores(result.answer, gold)
        accuracy = exact_match(result.answer, gold)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        latencies.append(timer.elapsed)
        iterations.append(result.iterations)
        retrieval_failures += result.retrieval_failures
        reasoning_failures += result.reasoning_failures

        logger.info(
            "%s | question=%r | prediction=%r | gold=%r | acc=%.3f | f1=%.3f",
            model_name,
            question,
            result.answer,
            gold,
            accuracy,
            f1,
        )

    total = max(len(dataset), 1)

    return {
        "Model": model_name,
        "Accuracy": mean(accuracies) if accuracies else 0.0,
        "Precision": mean(precisions) if precisions else 0.0,
        "Recall": mean(recalls) if recalls else 0.0,
        "Token F1": mean(f1s) if f1s else 0.0,
        "Average Latency": mean(latencies) if latencies else 0.0,
        "Average Iterations": mean(iterations) if iterations else 0.0,
        "Retrieval Failures": retrieval_failures,
        "Reasoning Failures": reasoning_failures,
        "Samples": total,
    }
