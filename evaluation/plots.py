"""Publication-quality comparison plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURE_DPI, FIGURE_HEIGHT, FIGURE_WIDTH, OUTPUT_FIGURE


def save_model_comparison(results: list[dict]) -> None:
    """Generate a grouped bar chart for core metrics."""

    df = pd.DataFrame(results)
    metrics = ["Accuracy", "Precision", "Recall", "Token F1"]

    x = np.arange(len(df["Model"]))
    width = 0.18

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)

    for index, metric in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, df[metric], width, label=metric)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_title("Vanilla ReAct HotpotQA Model Comparison")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
