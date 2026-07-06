"""
tests/test_data_ingestion.py — Unit tests for data ingestion and validation.
"""

from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from lsi_pipeline.config import FEATURE_COLUMNS, TARGET_COLUMN
from lsi_pipeline.data_ingestion import (
    generate_non_landslide_samples,
    load_and_validate,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def minimal_csv(tmp_path):
    """Write a minimal valid training CSV with only target=1 rows."""
    cols = ["x", "y"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    n = 22
    rng = np.random.default_rng(0)
    data = {
        "x": rng.uniform(520_000, 545_000, n),
        "y": rng.uniform(2_580_000, 2_610_000, n),
    }
    for col in FEATURE_COLUMNS:
        data[col] = rng.uniform(1.5, 3.5, n)
    data[TARGET_COLUMN] = [1] * n

    df = pd.DataFrame(data)
    p = tmp_path / "training.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture()
def balanced_csv(tmp_path):
    """Write a balanced CSV with both target=0 and target=1 rows."""
    n = 22
    rng = np.random.default_rng(1)
    rows_pos = {
        "x": rng.uniform(520_000, 545_000, n),
        "y": rng.uniform(2_580_000, 2_610_000, n),
    }
    for col in FEATURE_COLUMNS:
        rows_pos[col] = rng.uniform(1.5, 3.5, n)
    rows_pos[TARGET_COLUMN] = [1] * n

    rows_neg = {
        "x": rng.uniform(520_000, 545_000, n),
        "y": rng.uniform(2_580_000, 2_610_000, n),
    }
    for col in FEATURE_COLUMNS:
        rows_neg[col] = rng.uniform(0.1, 0.6, n)
    rows_neg[TARGET_COLUMN] = [0] * n

    df = pd.concat([pd.DataFrame(rows_pos), pd.DataFrame(rows_neg)], ignore_index=True)
    p = tmp_path / "balanced.csv"
    df.to_csv(p, index=False)
    return str(p)


# ─── load_and_validate ────────────────────────────────────────────────────────


def test_load_valid_positive_only(minimal_csv):
    df = load_and_validate(minimal_csv)
    assert len(df) == 22
    assert set(df[TARGET_COLUMN].unique()) == {1}
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_load_valid_balanced(balanced_csv):
    df = load_and_validate(balanced_csv)
    assert len(df) == 44
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}


def test_load_missing_feature_column(tmp_path):
    """Missing a feature column should raise ValueError."""
    if not FEATURE_COLUMNS:
        pytest.skip("No feature columns configured.")
    cols = ["x", "y"] + FEATURE_COLUMNS[:-1] + [TARGET_COLUMN]  # drop last feature
    df = pd.DataFrame({c: [1.0] for c in cols})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_and_validate(str(p))


def test_load_null_in_feature(tmp_path):
    """NaN in a feature column should raise ValueError."""
    if not FEATURE_COLUMNS:
        pytest.skip("No feature columns configured.")
    data = {"x": [1.0], "y": [1.0]}
    for col in FEATURE_COLUMNS:
        data[col] = [np.nan]
    data[TARGET_COLUMN] = [1]
    df = pd.DataFrame(data)
    p = tmp_path / "null_feat.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Null values"):
        load_and_validate(str(p))


def test_load_invalid_target(tmp_path):
    """Non-binary target should raise ValueError."""
    data = {"x": [1.0], "y": [1.0]}
    for col in FEATURE_COLUMNS:
        data[col] = [1.0]
    data[TARGET_COLUMN] = [2]   # invalid
    df = pd.DataFrame(data)
    p = tmp_path / "bad_target.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="target column"):
        load_and_validate(str(p))


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_validate("/nonexistent/path.csv")


# ─── generate_non_landslide_samples ──────────────────────────────────────────


def test_generate_returns_correct_count():
    n = 5
    rng = np.random.default_rng(0)
    pos_data = {
        "x": rng.uniform(520_000, 545_000, n),
        "y": rng.uniform(2_580_000, 2_610_000, n),
    }
    for col in FEATURE_COLUMNS:
        pos_data[col] = rng.uniform(1.5, 3.0, n)
    pos_data[TARGET_COLUMN] = [1] * n
    df_pos = pd.DataFrame(pos_data)

    df_neg = generate_non_landslide_samples(df_pos, n_samples=n, rng_seed=42)
    assert len(df_neg) <= n  # may be fewer if bbox is tight
    assert (df_neg[TARGET_COLUMN] == 0).all()
    for col in FEATURE_COLUMNS:
        assert col in df_neg.columns


def test_generate_target_is_zero():
    rng = np.random.default_rng(0)
    n = 3
    pos_data = {
        "x": rng.uniform(520_000, 545_000, n),
        "y": rng.uniform(2_580_000, 2_610_000, n),
    }
    for col in FEATURE_COLUMNS:
        pos_data[col] = rng.uniform(1.0, 2.0, n)
    pos_data[TARGET_COLUMN] = [1] * n
    df_pos = pd.DataFrame(pos_data)

    df_neg = generate_non_landslide_samples(df_pos, n_samples=n, rng_seed=99)
    assert (df_neg[TARGET_COLUMN] == 0).all()
