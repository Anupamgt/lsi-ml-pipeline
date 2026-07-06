"""
preprocessing.py — Step 3: Scaling and cross-validation strategy.

Implements:
  • fit_scaler_and_transform()   — StandardScaler fit/transform with save.
  • buffered_loocv()             — Spatial LOOCV with 1 km exclusion buffer.
  • stratified_kfold_cv()        — 5-fold stratified CV reporting mean±std AUC.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from lsi_pipeline.config import (
    BUFFER_RADIUS,
    FEATURE_COLUMNS,
    N_FOLDS,
    RANDOM_STATE,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


# ─── Scaler ──────────────────────────────────────────────────────────────────


def fit_scaler_and_transform(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on training data and transform both splits.

    The fitted scaler is serialised to ``scaler_path`` via joblib for
    later inference on the susceptibility grid.

    Parameters
    ----------
    X_train:
        Training feature matrix (n_train, n_features).
    X_test:
        Test feature matrix (n_test, n_features).
    scaler_path:
        Output path for the serialised scaler (``scaler.pkl``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, StandardScaler]
        ``(X_train_scaled, X_test_scaled, fitted_scaler)``
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved → %s", scaler_path)

    return X_train_scaled, X_test_scaled, scaler


def fit_full_scaler(X: np.ndarray, scaler_path: str | Path) -> StandardScaler:
    """Fit a StandardScaler on the complete dataset and save it.

    Used as the final production scaler after CV is complete.

    Parameters
    ----------
    X:
        Full feature matrix (n_samples, n_features).
    scaler_path:
        Output path for the serialised scaler (``scaler.pkl``).

    Returns
    -------
    StandardScaler
        The fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(X)
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Full-dataset scaler saved → %s", scaler_path)
    return scaler


# ─── Buffered LOOCV ──────────────────────────────────────────────────────────


def buffered_loocv(
    df: pd.DataFrame,
    model_cls,
    model_params: dict,
    buffer_radius: float = BUFFER_RADIUS,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Perform Buffered Leave-One-Out Cross-Validation (spatial LOOCV).

    For each sample *i*:

    1. Hold out sample *i* as the test point.
    2. Compute Euclidean distance from *i* to every other sample.
    3. Remove all samples within ``buffer_radius`` metres from the
       training set (spatial autocorrelation guard).
    4. Fit a fresh scaler on the remaining training samples.
    5. Transform the held-out point and predict its probability.

    All predicted probabilities are aggregated to compute a single AUC.

    Parameters
    ----------
    df:
        Full dataset including ``x``, ``y``, feature columns, and target.
    model_cls:
        Scikit-learn estimator class (e.g. ``RandomForestClassifier``).
    model_params:
        Keyword arguments passed to ``model_cls()``.
    buffer_radius:
        Exclusion radius in metres (default: ``BUFFER_RADIUS = 1000 m``).

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        ``(loocv_auc, y_true_all, y_prob_all)``
        The arrays can be used to plot the LOOCV ROC curve.
    """
    coords = df[["x", "y"]].values         # (N, 2)
    X = df[FEATURE_COLUMNS].values          # (N, F)
    y = df[TARGET_COLUMN].values            # (N,)
    N = len(df)

    y_true_all: list[int] = []
    y_prob_all: list[float] = []

    skipped_folds = 0

    for i in range(N):
        # Distances from point i to all other points
        diffs = coords - coords[i]          # (N, 2)
        dists = np.linalg.norm(diffs, axis=1)  # (N,)

        # Training mask: exclude the test point AND buffer zone
        train_mask = (dists > buffer_radius)  # excludes i (dist=0) automatically
        train_idx = np.where(train_mask)[0]

        if len(train_idx) == 0:
            logger.warning("Fold %d: no training samples outside buffer — skipping.", i)
            skipped_folds += 1
            continue

        # Check class diversity in training fold
        y_train_fold = y[train_idx]
        if len(np.unique(y_train_fold)) < 2:
            logger.warning(
                "Fold %d: training fold has only one class — skipping.", i
            )
            skipped_folds += 1
            continue

        X_train_fold = X[train_idx]
        X_test_fold = X[i : i + 1]  # shape (1, F)

        # Fit scaler on this fold's training data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # Fit model
        model = model_cls(**model_params)
        model.fit(X_train_scaled, y_train_fold)

        # Predict probability for held-out point
        prob = model.predict_proba(X_test_scaled)[0, 1]
        y_true_all.append(int(y[i]))
        y_prob_all.append(float(prob))

    y_true_arr = np.array(y_true_all)
    y_prob_arr = np.array(y_prob_all)

    if len(np.unique(y_true_arr)) < 2:
        logger.error("LOOCV: aggregated predictions contain only one class; AUC undefined.")
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true_arr, y_prob_arr)

    logger.info(
        "Buffered LOOCV complete — AUC: %.4f  (skipped %d / %d folds)",
        auc, skipped_folds, N,
    )
    return auc, y_true_arr, y_prob_arr


# ─── Stratified K-Fold CV ────────────────────────────────────────────────────


def stratified_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_cls,
    model_params: dict,
    n_splits: int = N_FOLDS,
) -> tuple[float, float, list[float]]:
    """Stratified K-Fold cross-validation with per-fold scaler fitting.

    Each fold independently fits a ``StandardScaler`` on the training
    partition and transforms the test partition, preventing data leakage.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features). Unscaled.
    y:
        Binary target vector (n_samples,).
    model_cls:
        Scikit-learn estimator class.
    model_params:
        Hyperparameters for ``model_cls``.
    n_splits:
        Number of CV folds (default: ``N_FOLDS = 5``).

    Returns
    -------
    tuple[float, float, list[float]]
        ``(mean_auc, std_auc, per_fold_aucs)``
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        model = model_cls(**model_params)
        model.fit(X_tr_scaled, y_tr)

        proba = model.predict_proba(X_te_scaled)[:, 1]
        auc = roc_auc_score(y_te, proba)
        fold_aucs.append(auc)
        logger.debug("  Fold %d AUC: %.4f", fold_idx, auc)

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs, ddof=1))

    logger.info(
        "Stratified %d-Fold CV — Mean AUC: %.4f ± %.4f",
        n_splits, mean_auc, std_auc,
    )
    return mean_auc, std_auc, fold_aucs
