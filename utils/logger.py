"""Project logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from config import LOG_DIR, LOG_LEVEL


def get_logger(name: str, filename: str | None = None) -> logging.Logger:
    """Create a logger that writes to logs/*.log and stdout."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_name = filename or f"{name}.log"
    log_path = Path(LOG_DIR) / log_name
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
