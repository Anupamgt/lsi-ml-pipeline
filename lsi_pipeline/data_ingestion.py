"""
data_ingestion.py — Step 1 & 2: Load, validate, and optionally generate
non-landslide samples for the LSI training dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lsi_pipeline.config import (
    COORD_COLUMNS,
    FEATURE_COLUMNS,
    FR_LOW_MAX,
    FR_LOW_MIN,
    MIN_SAMPLE_BUFFER_M,
    RANDOM_STATE,
    STUDY_BBOX,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


# ─── STEP 1: Load & Validate ─────────────────────────────────────────────────


def load_and_validate(csv_path: str | Path) -> pd.DataFrame:
    """Load the training CSV and validate schema, nulls, and target values.

    Parameters
    ----------
    csv_path:
        Absolute or relative path to ``landslide_training_data.csv``.
        Required columns: ``x``, ``y``, all entries in ``FEATURE_COLUMNS``,
        and ``target`` (binary 0/1).

    Returns
    -------
    pd.DataFrame
        Validated DataFrame. Raises ``ValueError`` on schema or data issues.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded %s — shape %s", csv_path.name, df.shape)

    # ── Schema check ────────────────────────────────────────────────────────
    required_cols = COORD_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in input CSV: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── Null check on feature columns ───────────────────────────────────────
    null_counts = df[FEATURE_COLUMNS].isnull().sum()
    bad_cols = null_counts[null_counts > 0]
    if not bad_cols.empty:
        raise ValueError(f"Null values found in feature columns:\n{bad_cols}")

    # ── Binary target check ──────────────────────────────────────────────────
    unique_targets = set(df[TARGET_COLUMN].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"target column must contain only 0 and 1. Found: {unique_targets}"
        )

    _print_dataset_summary(df)
    return df


def _print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of the loaded dataset.

    Parameters
    ----------
    df:
        Validated training DataFrame.
    """
    n_total = len(df)
    n_pos = int((df[TARGET_COLUMN] == 1).sum())
    n_neg = int((df[TARGET_COLUMN] == 0).sum())
    ratio = n_pos / n_neg if n_neg > 0 else float("inf")

    print("\n" + "═" * 52)
    print("  DATASET SUMMARY")
    print("═" * 52)
    print(f"  Total samples       : {n_total}")
    print(f"  Landslide (1)       : {n_pos}")
    print(f"  Non-landslide (0)   : {n_neg}")
    print(f"  Class ratio (1:0)   : {ratio:.2f}")
    print(f"  Active features     : {len(FEATURE_COLUMNS)}")
    print(f"  Feature columns     : {FEATURE_COLUMNS}")
    print("═" * 52 + "\n")


# ─── STEP 2: Non-Landslide Point Generation ──────────────────────────────────


def generate_non_landslide_samples(
    df_positive: pd.DataFrame,
    n_samples: int | None = None,
    rng_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Randomly generate non-landslide pseudo-absence points.

    Points are sampled within the Aizawl study bounding box
    (``STUDY_BBOX``) with a minimum Euclidean distance of
    ``MIN_SAMPLE_BUFFER_M`` metres from all positive (landslide) points.

    Feature (FR) values for pseudo-absence points are drawn from
    ``U(FR_LOW_MIN, FR_LOW_MAX)`` to simulate low-risk zone characteristics.

    Parameters
    ----------
    df_positive:
        DataFrame containing only landslide (target=1) rows.
    n_samples:
        Number of non-landslide points to generate.
        Defaults to ``len(df_positive)`` for balanced classes.
    rng_seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated non-landslide samples with the same column schema as
        ``df_positive``.
    """
    if n_samples is None:
        n_samples = len(df_positive)

    rng = np.random.default_rng(rng_seed)
    pos_coords = df_positive[["x", "y"]].values  # (N, 2)

    x_min, x_max = STUDY_BBOX["x_min"], STUDY_BBOX["x_max"]
    y_min, y_max = STUDY_BBOX["y_min"], STUDY_BBOX["y_max"]

    sampled_xy: list[tuple[float, float]] = []
    max_attempts = n_samples * 200  # prevent infinite loop on tight domains
    attempt = 0

    logger.info(
        "Generating %d pseudo-absence points (buffer=%.0fm) …",
        n_samples,
        MIN_SAMPLE_BUFFER_M,
    )

    while len(sampled_xy) < n_samples and attempt < max_attempts:
        attempt += 1
        cand_x = rng.uniform(x_min, x_max)
        cand_y = rng.uniform(y_min, y_max)

        # Euclidean distances to all positive points
        diffs = pos_coords - np.array([[cand_x, cand_y]])
        dists = np.linalg.norm(diffs, axis=1)

        if dists.min() >= MIN_SAMPLE_BUFFER_M:
            sampled_xy.append((cand_x, cand_y))

    if len(sampled_xy) < n_samples:
        logger.warning(
            "Could only generate %d / %d pseudo-absence points within "
            "the bounding box with %.0f m buffer after %d attempts.",
            len(sampled_xy),
            n_samples,
            MIN_SAMPLE_BUFFER_M,
            max_attempts,
        )

    xs, ys = zip(*sampled_xy) if sampled_xy else ([], [])
    n_gen = len(sampled_xy)

    # Assign low-risk FR placeholder values
    fr_values = rng.uniform(FR_LOW_MIN, FR_LOW_MAX, size=(n_gen, len(FEATURE_COLUMNS)))

    rows: dict[str, list] = {"x": list(xs), "y": list(ys)}
    for i, col in enumerate(FEATURE_COLUMNS):
        rows[col] = fr_values[:, i].tolist()
    rows[TARGET_COLUMN] = [0] * n_gen

    df_neg = pd.DataFrame(rows)
    logger.info("Generated %d non-landslide samples.", n_gen)
    return df_neg


def build_full_dataset(
    df: pd.DataFrame, output_path: str | Path
) -> pd.DataFrame:
    """Ensure the dataset contains both classes; generate negatives if needed.

    If ``df`` contains only target=1 rows, pseudo-absence samples are
    generated with :func:`generate_non_landslide_samples` and the combined
    dataset is saved to ``output_path``.

    Parameters
    ----------
    df:
        Validated input DataFrame (may be positive-only or mixed).
    output_path:
        Path where ``landslide_full_dataset.csv`` will be saved.

    Returns
    -------
    pd.DataFrame
        Combined dataset with both classes, shuffled.
    """
    output_path = Path(output_path)

    has_negatives = (df[TARGET_COLUMN] == 0).any()

    if has_negatives:
        logger.info("Dataset already contains non-landslide samples — skipping generation.")
        df_full = df.copy()
    else:
        logger.info("Only landslide points found — generating pseudo-absence samples.")
        df_pos = df[df[TARGET_COLUMN] == 1].copy()
        df_neg = generate_non_landslide_samples(df_pos)
        df_full = pd.concat([df_pos, df_neg], ignore_index=True)

    # Shuffle
    df_full = df_full.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(output_path, index=False)
    logger.info("Full dataset saved → %s  (shape: %s)", output_path, df_full.shape)

    return df_full
