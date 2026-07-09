"""Run the full Vanilla ReAct HotpotQA evaluation."""

from __future__ import annotations

import argparse
import gc
import shutil

from config import MODELS, OUTPUT_DIR, OUTPUT_EXCEL, OUTPUT_FIGURE, SAMPLE_SIZE, set_seed
from evaluation.evaluator import evaluate
from evaluation.excel import save_evaluation
from evaluation.plots import save_model_comparison
from models.loader import ModelLoader
from utils.dataset import load_hotpotqa
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Vanilla ReAct on HotpotQA."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        help="Evaluate one configured model. Defaults to all models.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of validation examples to evaluate. Defaults to config.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    logger = get_logger("evaluation", "evaluation.log")
    clean_outputs()

    logger.info("Loading HotpotQA validation data")
    sample_size = SAMPLE_SIZE if args.sample_size is None else args.sample_size
    if sample_size is not None and sample_size <= 0:
        sample_size = None
    dataset = load_hotpotqa(sample_size=sample_size)
    logger.info("Loaded %d examples", len(dataset))

    results = []
    model_names = [args.model] if args.model else list(MODELS)

    for model_name in model_names:
        logger.info("Starting evaluation for %s", model_name)
        loader = ModelLoader(model_name)

        try:
            result = evaluate(model_name, loader, dataset, logger)
            results.append(result)
            save_evaluation(results)
            save_model_comparison(results)
            logger.info("Completed %s: %s", model_name, result)
        finally:
            loader.unload()
            gc.collect()

    logger.info("Saved outputs to %s", OUTPUT_DIR)


def clean_outputs() -> None:
    """Keep outputs/ limited to the requested report files."""

    allowed = {OUTPUT_EXCEL.resolve(), OUTPUT_FIGURE.resolve()}

    for path in OUTPUT_DIR.iterdir():
        if path.resolve() in allowed:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


if __name__ == "__main__":
    main()
