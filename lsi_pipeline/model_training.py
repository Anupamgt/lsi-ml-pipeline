"""
model_training.py — Step 4: Train and persist ML models.

Trains Random Forest (primary), Logistic Regression (baseline), and
optionally Extreme Gradient Boosting (XGBoost) on the standardised dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lsi_pipeline.config import LR_PARAMS, RF_PARAMS, XGB_PARAMS

logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: str | Path,
) -> RandomForestClassifier:
    """Train a Random Forest Classifier and save it to disk."""
    logger.info("Training Random Forest — params: %s", RF_PARAMS)
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, output_path)
    logger.info("Random Forest saved → %s", output_path)
    return rf


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: str | Path,
) -> LogisticRegression:
    """Train a Logistic Regression classifier and save it to disk."""
    logger.info("Training Logistic Regression — params: %s", LR_PARAMS)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_train, y_train)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, output_path)
    logger.info("Logistic Regression saved → %s", output_path)
    return lr


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: str | Path,
    params: dict | None = None,
):
    """Train an XGBoost classifier and save it to disk.

    Uses ``scale_pos_weight`` derived from class imbalance when both classes
    are present. Requires the ``xgboost`` package.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for train_xgboost(). "
            "Install with: pip install xgboost"
        ) from exc

    fit_params = dict(params or XGB_PARAMS)
    y_train = np.asarray(y_train).astype(int)
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    fit_params.setdefault("scale_pos_weight", n_neg / n_pos)

    logger.info("Training XGBoost — params: %s", fit_params)
    xgb = XGBClassifier(**fit_params)
    xgb.fit(X_train, y_train)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, output_path)
    logger.info("XGBoost saved → %s", output_path)
    return xgb


def get_xgb_sklearn_params() -> dict:
    """Return XGB params suitable for ``buffered_loocv`` / k-fold factories.

    Adds a default ``scale_pos_weight=1.0``; callers with known imbalance
    may override.
    """
    params = dict(XGB_PARAMS)
    params.setdefault("scale_pos_weight", 1.0)
    return params
