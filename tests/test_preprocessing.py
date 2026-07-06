"""
tests/test_preprocessing.py — Unit tests for CV and scaler utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lsi_pipeline.config import FEATURE_COLUMNS, RF_PARAMS, LR_PARAMS, TARGET_COLUMN
from lsi_pipeline.preprocessing import (
    fit_full_scaler,
    fit_scaler_and_transform,
    stratified_kfold_cv,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_dataset():
    """Small balanced dataset with 2 separable classes."""
    rng = np.random.default_rng(42)
    n = 30
    X = np.zeros((n, len(FEATURE_COLUMNS)))
    y = np.zeros(n, dtype=int)

    half = n // 2
    # Class 1: high values
    X[:half] = rng.uniform(2.0, 4.0, (half, len(FEATURE_COLUMNS)))
    y[:half] = 1
    # Class 0: low values
    X[half:] = rng.uniform(0.0, 1.0, (n - half, len(FEATURE_COLUMNS)))

    # Build DataFrame with coords
    coords = rng.uniform(0, 10_000, (n, 2))  # small synthetic domain
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df[TARGET_COLUMN] = y
    return df, X, y


# ─── Scaler tests ─────────────────────────────────────────────────────────────


def test_fit_scaler_shapes(tmp_path, synthetic_dataset):
    _, X, _ = synthetic_dataset
    X_tr, X_te = X[:20], X[20:]
    scaler_path = tmp_path / "scaler.pkl"
    X_tr_s, X_te_s, scaler = fit_scaler_and_transform(X_tr, X_te, scaler_path)

    assert X_tr_s.shape == X_tr.shape
    assert X_te_s.shape == X_te.shape
    assert scaler_path.exists()


def test_fit_scaler_zero_mean(tmp_path, synthetic_dataset):
    _, X, _ = synthetic_dataset
    X_tr, X_te = X[:20], X[20:]
    scaler_path = tmp_path / "scaler2.pkl"
    X_tr_s, _, _ = fit_scaler_and_transform(X_tr, X_te, scaler_path)
    # Training set should be approximately zero-mean after StandardScaler
    np.testing.assert_allclose(X_tr_s.mean(axis=0), 0.0, atol=1e-10)


def test_fit_full_scaler_saves(tmp_path, synthetic_dataset):
    _, X, _ = synthetic_dataset
    scaler_path = tmp_path / "full_scaler.pkl"
    scaler = fit_full_scaler(X, scaler_path)
    assert scaler_path.exists()
    import joblib
    loaded = joblib.load(scaler_path)
    np.testing.assert_allclose(scaler.mean_, loaded.mean_)


# ─── K-Fold CV tests ──────────────────────────────────────────────────────────


def test_stratified_kfold_returns_valid_auc(synthetic_dataset):
    _, X, y = synthetic_dataset
    mean_auc, std_auc, per_fold = stratified_kfold_cv(
        X, y, RandomForestClassifier, RF_PARAMS, n_splits=5
    )
    assert 0.0 <= mean_auc <= 1.0
    assert std_auc >= 0.0
    assert len(per_fold) == 5
    assert all(0.0 <= a <= 1.0 for a in per_fold)


def test_stratified_kfold_lr(synthetic_dataset):
    _, X, y = synthetic_dataset
    mean_auc, std_auc, _ = stratified_kfold_cv(
        X, y, LogisticRegression, LR_PARAMS, n_splits=3
    )
    assert 0.0 <= mean_auc <= 1.0
