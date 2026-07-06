"""
feature_importance.py — Step 6: RF feature importance visualisation.

Produces a sorted horizontal bar chart of Random Forest feature importances
with all active feature column labels on the y-axis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from lsi_pipeline.config import FEATURE_COLUMNS

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def plot_feature_importance(
    rf_model: RandomForestClassifier,
    output_path: str | Path,
) -> None:
    """Plot a horizontal bar chart of RF Gini feature importances.

    Features are sorted in descending order of importance. The chart uses
    a dark theme consistent with the ROC curve plot.

    Parameters
    ----------
    rf_model:
        Fitted ``RandomForestClassifier`` instance.
    output_path:
        Destination path for ``feature_importance.png`` (300 DPI).
    """
    importances: np.ndarray = rf_model.feature_importances_
    std: np.ndarray = np.std(
        [tree.feature_importances_ for tree in rf_model.estimators_], axis=0
    )

    # Sort descending
    indices = np.argsort(importances)  # ascending; we'll reverse for plotting
    features_sorted = [FEATURE_COLUMNS[i] for i in indices]
    importances_sorted = importances[indices]
    std_sorted = std[indices]

    n = len(features_sorted)

    # Dynamic height: 0.45 inches per feature, minimum 5
    fig_height = max(5, n * 0.52)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # Colour gradient from low→high importance (teal → gold)
    colours = plt.cm.YlGnBu(np.linspace(0.35, 0.95, n))

    bars = ax.barh(
        range(n),
        importances_sorted,
        xerr=std_sorted,
        align="center",
        color=colours,
        edgecolor="#0f1117",
        linewidth=0.5,
        error_kw={"ecolor": "#ffffff", "alpha": 0.5, "capsize": 3, "lw": 1},
        height=0.65,
    )

    # Value labels on each bar
    for bar, val in zip(bars, importances_sorted):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#b0b0b0",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(features_sorted, fontsize=10, color="#e0e0e0")
    ax.set_xlabel("Mean Decrease in Gini Impurity", color="#c0c0c0", fontsize=11)
    ax.set_title(
        "Random Forest — Feature Importances\n"
        "Landslide Susceptibility Index (Aizawl, Mizoram)",
        color="#ffffff",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333550")

    ax.set_xlim(0, importances_sorted.max() * 1.18)
    ax.grid(True, axis="x", linestyle="--", alpha=0.2, color="#555770")

    # Subtitle with error bar note
    fig.text(
        0.5, 0.01,
        "Error bars show ±1 SD across trees",
        ha="center",
        fontsize=8,
        color="#707080",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Feature importance plot saved → %s", output_path)


def print_feature_importance_table(rf_model: RandomForestClassifier) -> None:
    """Print a ranked feature importance table to stdout.

    Parameters
    ----------
    rf_model:
        Fitted ``RandomForestClassifier`` instance.
    """
    importances = rf_model.feature_importances_
    ranked = sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True
    )

    print("\n" + "═" * 45)
    print("  FEATURE IMPORTANCE RANKING (RF Gini)")
    print("═" * 45)
    print(f"  {'Rank':<5}  {'Feature':<26}  {'Importance'}")
    print("  " + "─" * 41)
    for rank, (feat, imp) in enumerate(ranked, start=1):
        bar = "█" * int(imp * 60)
        print(f"  {rank:<5}  {feat:<26}  {imp:.4f}  {bar}")
    print("═" * 45 + "\n")
