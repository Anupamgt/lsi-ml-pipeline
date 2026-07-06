"""
model_training.py — Step 4: Train and persist ML models.

Trains a Random Forest Classifier (primary) and Logistic Regression
(baseline) on the full standardised dataset and saves both to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lsi_pipeline.config import LR_PARAMS, RF_PARAMS

logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: str | Path,
) -> RandomForestClassifier:
    """Train a Random Forest Classifier and save it to disk.

    Uses hyperparameters from ``config.RF_PARAMS``:
      - n_estimators=200, max_depth=None, class_weight='balanced',
        random_state=42, n_jobs=-1.

    Parameters
    ----------
    X_train:
        Scaled training features (n_samples, n_features).
    y_train:
        Binary training targets (n_samples,).
    output_path:
        Path to save ``rf_model.pkl`` via joblib.

    Returns
    -------
    RandomForestClassifier
        The fitted model.
    """
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
    """Train a Logistic Regression classifier and save it to disk.

    Uses hyperparameters from ``config.LR_PARAMS``:
      - solver='lbfgs', max_iter=1000, class_weight='balanced',
        random_state=42.

    Parameters
    ----------
    X_train:
        Scaled training features (n_samples, n_features).
    y_train:
        Binary training targets (n_samples,).
    output_path:
        Path to save ``lr_model.pkl`` via joblib.

    Returns
    -------
    LogisticRegression
        The fitted model.
    """
    logger.info("Training Logistic Regression — params: %s", LR_PARAMS)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_train, y_train)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, output_path)
    logger.info("Logistic Regression saved → %s", output_path)
    return lr
