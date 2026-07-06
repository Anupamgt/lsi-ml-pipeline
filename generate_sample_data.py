#!/usr/bin/env python3
"""
generate_sample_data.py — Generate synthetic training data for pipeline testing.

Creates a realistic ``landslide_training_data.csv`` with 22 landslide points
within the Aizawl bounding box, using only the 3 currently available FR features.

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

# Match the study area from config
X_MIN, X_MAX = 520_000.0, 545_000.0
Y_MIN, Y_MAX = 2_580_000.0, 2_610_000.0

SEED = 42


def generate_training_data(
    output_path: str,
    with_negatives: bool = False,
    n_positive: int = 22,
) -> None:
    """Generate synthetic landslide training data.

    Parameters
    ----------
    output_path:
        Where to write the CSV.
    with_negatives:
        If True, also generate equal non-landslide rows (balanced dataset).
    n_positive:
        Number of landslide (target=1) points to generate.
    """
    rng = np.random.default_rng(SEED)

    # ── Landslide points (high FR values — zones 4-5 characteristics) ────
    x_pos = rng.uniform(X_MIN, X_MAX, n_positive)
    y_pos = rng.uniform(Y_MIN, Y_MAX, n_positive)

    # Higher FR values for landslide-prone areas
    slope_fr_pos = rng.uniform(1.5, 3.5, n_positive)
    aspect_fr_pos = rng.uniform(1.2, 2.8, n_positive)
    elevation_fr_pos = rng.uniform(1.8, 3.2, n_positive)

    df_pos = pd.DataFrame({
        "x": x_pos, "y": y_pos,
        "slope_fr": slope_fr_pos,
        "aspect_fr": aspect_fr_pos,
        "elevation_fr": elevation_fr_pos,
        "target": 1,
    })

    if with_negatives:
        x_neg = rng.uniform(X_MIN, X_MAX, n_positive)
        y_neg = rng.uniform(Y_MIN, Y_MAX, n_positive)
        slope_fr_neg = rng.uniform(0.1, 0.8, n_positive)
        aspect_fr_neg = rng.uniform(0.2, 0.9, n_positive)
        elevation_fr_neg = rng.uniform(0.1, 0.7, n_positive)

        df_neg = pd.DataFrame({
            "x": x_neg, "y": y_neg,
            "slope_fr": slope_fr_neg,
            "aspect_fr": aspect_fr_neg,
            "elevation_fr": elevation_fr_neg,
            "target": 0,
        })
        df = pd.concat([df_pos, df_neg], ignore_index=True).sample(
            frac=1, random_state=SEED
        ).reset_index(drop=True)
    else:
        df = df_pos

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Training data written → {output_path}  (shape: {df.shape})")


def generate_grid_data(output_path: str, n_pixels: int = 500) -> None:
    """Generate a synthetic raster grid CSV for susceptibility mapping.

    Parameters
    ----------
    output_path:
        Where to write the grid CSV.
    n_pixels:
        Number of grid pixels (simulates a coarse raster).
    """
    rng = np.random.default_rng(SEED + 1)

    x = rng.uniform(X_MIN, X_MAX, n_pixels)
    y = rng.uniform(Y_MIN, Y_MAX, n_pixels)
    slope_fr = rng.uniform(0.0, 4.0, n_pixels)
    aspect_fr = rng.uniform(0.0, 3.5, n_pixels)
    elevation_fr = rng.uniform(0.0, 4.0, n_pixels)

    df = pd.DataFrame({
        "x": x, "y": y,
        "slope_fr": slope_fr,
        "aspect_fr": aspect_fr,
        "elevation_fr": elevation_fr,
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Grid data written → {output_path}  (shape: {df.shape})")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate synthetic LSI test data.")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument(
        "--with-negatives", action="store_true",
        help="Include balanced non-landslide rows in training data"
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Generate a raster grid CSV instead of training data"
    )
    parser.add_argument(
        "--n-pixels", type=int, default=500,
        help="Number of grid pixels (only used with --grid)"
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
