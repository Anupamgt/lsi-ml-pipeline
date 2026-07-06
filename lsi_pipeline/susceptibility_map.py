"""
susceptibility_map.py — Step 7: Grid prediction and susceptibility zone mapping.

Accepts an optional raster grid CSV (aizawl_grid.csv), predicts landslide
probability for every pixel using the fitted RF model, classifies into 5
susceptibility zones using quantile bins, and exports both output CSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lsi_pipeline.config import (
    FEATURE_COLUMNS,
    N_ZONES,
    OUT_SUSC_CLASS,
    OUT_SUSC_SCORES,
    ZONE_LABELS,
)

logger = logging.getLogger(__name__)


def predict_susceptibility(
    grid_csv_path: str | Path,
    rf_model_path: str | Path,
    scaler_path: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict landslide probability and susceptibility zone for a pixel grid.

    Reads a grid CSV with columns ``[x, y, <all FEATURE_COLUMNS>]``,
    loads the fitted RF model and StandardScaler, predicts
    ``predict_proba[:, 1]`` for each pixel, classifies into 5 quantile
    zones (Very Low → Very High), and writes two output CSVs.

    Parameters
    ----------
    grid_csv_path:
        Path to ``aizawl_grid.csv``. Must contain ``x``, ``y``, and all
        columns listed in ``FEATURE_COLUMNS``.
    rf_model_path:
        Path to saved ``rf_model.pkl``.
    scaler_path:
        Path to saved ``scaler.pkl``.
    output_dir:
        Directory where output CSVs will be written.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_scores, df_classified)`` where:
        - ``df_scores``     : columns [x, y, lsi_prob]
        - ``df_classified`` : columns [x, y, lsi_prob, zone_id, zone_label]
    """
    grid_csv_path = Path(grid_csv_path)
    output_dir = Path(output_dir)

    # ── Load grid ─────────────────────────────────────────────────────────
    logger.info("Loading grid CSV: %s", grid_csv_path)
    df_grid = pd.read_csv(grid_csv_path)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df_grid.columns]
    if missing_cols:
        raise ValueError(
            f"Grid CSV is missing feature columns: {missing_cols}. "
            f"Available: {list(df_grid.columns)}"
        )

    X_grid = df_grid[FEATURE_COLUMNS].values

    # ── Load scaler and model ─────────────────────────────────────────────
    logger.info("Loading scaler: %s", scaler_path)
    scaler: StandardScaler = joblib.load(scaler_path)

    logger.info("Loading RF model: %s", rf_model_path)
    rf = joblib.load(rf_model_path)

    # ── Scale and predict ─────────────────────────────────────────────────
    logger.info("Predicting probabilities for %d grid pixels …", len(df_grid))
    X_grid_scaled = scaler.transform(X_grid)
    lsi_prob = rf.predict_proba(X_grid_scaled)[:, 1]

    # ── Assemble scores CSV ───────────────────────────────────────────────
    df_scores = pd.DataFrame({
        "x": df_grid["x"].values,
        "y": df_grid["y"].values,
        "lsi_prob": lsi_prob,
    })

    scores_path = output_dir / OUT_SUSC_SCORES
    output_dir.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(scores_path, index=False)
    logger.info("Susceptibility scores saved → %s", scores_path)

    # ── Classify into 5 zones (quantile bins) ────────────────────────────
    df_classified = df_scores.copy()
    df_classified["zone_id"], bin_edges = pd.qcut(
        lsi_prob,
        q=N_ZONES,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    actual_n_zones = int(df_classified["zone_id"].max()) + 1
    zone_label_map = {i: ZONE_LABELS[i] for i in range(actual_n_zones)}
    df_classified["zone_label"] = df_classified["zone_id"].map(zone_label_map)

    _print_zone_distribution(df_classified, bin_edges)

    classified_path = output_dir / OUT_SUSC_CLASS
    df_classified.to_csv(classified_path, index=False)
    logger.info("Susceptibility classified map saved → %s", classified_path)

    return df_scores, df_classified


def _print_zone_distribution(df: pd.DataFrame, bin_edges: np.ndarray) -> None:
    """Print a tabular summary of susceptibility zone distribution.

    Parameters
    ----------
    df:
        Classified susceptibility DataFrame (must have ``zone_label`` column).
    bin_edges:
        Array of quantile bin edges from ``pd.qcut``.
    """
    counts = df["zone_label"].value_counts().reindex(ZONE_LABELS, fill_value=0)
    total = len(df)

    print("\n" + "═" * 55)
    print("  SUSCEPTIBILITY ZONE DISTRIBUTION")
    print("═" * 55)
    print(f"  {'Zone':<12}  {'Count':>8}  {'Percent':>8}  {'Prob Range'}")
    print("  " + "─" * 50)
    for i, label in enumerate(ZONE_LABELS):
        cnt = counts.get(label, 0)
        pct = cnt / total * 100 if total > 0 else 0.0
        lo = bin_edges[i] if i < len(bin_edges) else 0.0
        hi = bin_edges[i + 1] if (i + 1) < len(bin_edges) else 1.0
        print(f"  {label:<12}  {cnt:>8}  {pct:>7.2f}%  [{lo:.3f}, {hi:.3f}]")
    print("═" * 55 + "\n")
