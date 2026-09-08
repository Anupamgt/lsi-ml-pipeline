#!/usr/bin/env python3
"""
generate_sample_data.py — Generate synthetic training / grid data for pipeline testing.

Emits all columns from ``FEATURE_COLUMNS`` (17 Sonker FR factors). Prefer
``extract_training_data.py`` for real raster-sampled training data.

Usage:
    python generate_sample_data.py --output data/landslide_training_data.csv
    python generate_sample_data.py --output data/landslide_training_data.csv --with-negatives
    python generate_sample_data.py --output data/aizawl_grid.csv --grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lsi_pipeline.config import FEATURE_COLUMNS, STUDY_BBOX

X_MIN = STUDY_BBOX["x_min"]
X_MAX = STUDY_BBOX["x_max"]
Y_MIN = STUDY_BBOX["y_min"]
Y_MAX = STUDY_BBOX["y_max"]

SEED = 42

# Synthetic FR ranges: positives skewed high, negatives low, grid spans full range.
FR_POS_LOW, FR_POS_HIGH = 1.2, 3.5
FR_NEG_LOW, FR_NEG_HIGH = 0.1, 0.9
FR_GRID_LOW, FR_GRID_HIGH = 0.0, 4.0


def _synthetic_fr_block(
    rng: np.random.Generator,
    n: int,
    low: float,
    high: float,
) -> dict[str, np.ndarray]:
    """Draw independent synthetic FR values for every active feature."""
    return {
        col: rng.uniform(low, high, n)
        for col in FEATURE_COLUMNS
    }


def generate_training_data(
    output_path: str,
    with_negatives: bool = False,
    n_positive: int = 22,
) -> None:
    """Generate synthetic landslide training data with all FEATURE_COLUMNS."""
    rng = np.random.default_rng(SEED)

    x_pos = rng.uniform(X_MIN, X_MAX, n_positive)
    y_pos = rng.uniform(Y_MIN, Y_MAX, n_positive)
    pos_fr = _synthetic_fr_block(rng, n_positive, FR_POS_LOW, FR_POS_HIGH)

    df_pos = pd.DataFrame({
        "x": x_pos,
        "y": y_pos,
        **pos_fr,
        "target": 1,
    })

    if with_negatives:
        x_neg = rng.uniform(X_MIN, X_MAX, n_positive)
        y_neg = rng.uniform(Y_MIN, Y_MAX, n_positive)
        neg_fr = _synthetic_fr_block(rng, n_positive, FR_NEG_LOW, FR_NEG_HIGH)
        df_neg = pd.DataFrame({
            "x": x_neg,
            "y": y_neg,
            **neg_fr,
            "target": 0,
        })
        df = pd.concat([df_pos, df_neg], ignore_index=True).sample(
            frac=1, random_state=SEED
        ).reset_index(drop=True)
    else:
        df = df_pos

    col_order = ["x", "y", *FEATURE_COLUMNS, "target"]
    df = df[col_order]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"✓ Training data written → {output_path}  "
        f"(shape: {df.shape}, features: {len(FEATURE_COLUMNS)})"
    )


def generate_grid_data(output_path: str, n_pixels: int = 500) -> None:
    """Generate a synthetic raster grid CSV: x, y + all FEATURE_COLUMNS (no target)."""
    rng = np.random.default_rng(SEED + 1)

    x = rng.uniform(X_MIN, X_MAX, n_pixels)
    y = rng.uniform(Y_MIN, Y_MAX, n_pixels)
    fr = _synthetic_fr_block(rng, n_pixels, FR_GRID_LOW, FR_GRID_HIGH)

    df = pd.DataFrame({"x": x, "y": y, **fr})
    col_order = ["x", "y", *FEATURE_COLUMNS]
    df = df[col_order]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"✓ Grid data written → {output_path}  "
        f"(shape: {df.shape}, features: {len(FEATURE_COLUMNS)})"
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate synthetic LSI test data.")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument(
        "--with-negatives", action="store_true",
        help="Include balanced non-landslide rows in training data",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Generate a raster grid CSV instead of training data",
    )
    parser.add_argument(
        "--n-pixels", type=int, default=500,
        help="Number of grid pixels (only used with --grid)",
    )
    args = parser.parse_args()

    if args.grid:
        generate_grid_data(args.output, n_pixels=args.n_pixels)
    else:
        generate_training_data(
            args.output, with_negatives=args.with_negatives
        )


if __name__ == "__main__":
    main()
