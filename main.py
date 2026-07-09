"""Run the full Vanilla ReAct HotpotQA evaluation."""

from __future__ import annotations

import gc
import shutil

from config import MODELS, OUTPUT_DIR, OUTPUT_EXCEL, OUTPUT_FIGURE, set_seed
from evaluation.evaluator import evaluate
from evaluation.excel import save_evaluation
from evaluation.plots import save_model_comparison
from models.loader import ModelLoader
from utils.dataset import load_hotpotqa
from utils.logger import get_logger


def main() -> None:
    set_seed()
    logger = get_logger("evaluation", "evaluation.log")
    clean_outputs()

    logger.info("Loading HotpotQA validation data")
    dataset = load_hotpotqa()
    logger.info("Loaded %d examples", len(dataset))

    results = []

    for model_name in MODELS:
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
