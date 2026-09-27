"""
pu_bagging.py — Buffer Control Sampling + Positive-Unlabeled Bagging.

Practical NHESS-style helpers (Labbé et al.) for refining pseudo-absences:

1. **Buffer Control Sampling** — keep unlabeled points outside an inner
   exclusion buffer around positives (and optionally inside an outer ring).
2. **PU Bagging** — train an ensemble of classifiers, each on all positives
   plus a random subsample of unlabeled points; average predicted scores.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from lsi_pipeline.config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN

logger = logging.getLogger(__name__)


def pairwise_min_distance(
    query_xy: np.ndarray,
    reference_xy: np.ndarray,
) -> np.ndarray:
    """Minimum Euclidean distance from each query point to any reference point."""
    if len(reference_xy) == 0:
        return np.full(len(query_xy), np.inf)
    # Chunk to keep memory bounded for large pools
    out = np.empty(len(query_xy), dtype=np.float64)
    chunk = 5_000
    for start in range(0, len(query_xy), chunk):
        end = min(start + chunk, len(query_xy))
        batch = query_xy[start:end]
        dists = np.linalg.norm(
            batch[:, None, :] - reference_xy[None, :, :], axis=2
        )
        out[start:end] = dists.min(axis=1)
    return out


def buffer_control_mask(
    unlabeled_xy: np.ndarray,
    positive_xy: np.ndarray,
    inner_buffer_m: float = 500.0,
    outer_buffer_m: float | None = None,
) -> np.ndarray:
    """Boolean mask of unlabeled points that pass Buffer Control Sampling.

    Keeps points with ``distance >= inner_buffer_m`` from every positive.
    If ``outer_buffer_m`` is set, also requires ``distance <= outer_buffer_m``.
    """
    dists = pairwise_min_distance(unlabeled_xy, positive_xy)
    mask = dists >= float(inner_buffer_m)
    if outer_buffer_m is not None:
        mask &= dists <= float(outer_buffer_m)
    return mask


def apply_buffer_control_sampling(
    df: pd.DataFrame,
    inner_buffer_m: float = 500.0,
    outer_buffer_m: float | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter target=0 rows by Buffer Control Sampling; keep all positives.

    Parameters
    ----------
    df:
        Training DataFrame with ``x``, ``y``, ``target``, and feature columns.
    inner_buffer_m / outer_buffer_m:
        Buffer radii in metres (CRS units).
    feature_columns:
        Feature column names (defaults to ``config.FEATURE_COLUMNS``).
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    pos = df[df[TARGET_COLUMN] == 1]
    neg = df[df[TARGET_COLUMN] == 0]
    if neg.empty:
        logger.warning("No unlabeled/negative rows to filter.")
        return df.copy()

    pos_xy = pos[["x", "y"]].to_numpy(dtype=np.float64)
    neg_xy = neg[["x", "y"]].to_numpy(dtype=np.float64)
    keep = buffer_control_mask(neg_xy, pos_xy, inner_buffer_m, outer_buffer_m)
    neg_kept = neg.loc[keep]
    logger.info(
        "Buffer Control Sampling: kept %d / %d unlabeled "
        "(inner=%.0fm, outer=%s)",
        len(neg_kept),
        len(neg),
        inner_buffer_m,
        "none" if outer_buffer_m is None else f"{outer_buffer_m:.0f}m",
    )
    if neg_kept.empty:
        raise ValueError(
            "Buffer Control Sampling removed all unlabeled points. "
            "Relax inner_buffer_m or provide more candidates."
        )
    out = pd.concat([pos, neg_kept], ignore_index=True)
    return out.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


class PUBaggingClassifier(BaseEstimator, ClassifierMixin):
    """Positive-Unlabeled Bagging ensemble (Mordelet & Vert / Labbé-style).

    Each bag trains ``base_estimator`` on all positives plus a random
    subsample of unlabeled points. Prediction averages bag probabilities.
    """

    def __init__(
        self,
        base_estimator: Any | None = None,
        n_bags: int = 10,
        sample_ratio: float = 1.0,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_bags = n_bags
        self.sample_ratio = sample_ratio
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PUBaggingClassifier":
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        pos_idx = np.where(y == 1)[0]
        unl_idx = np.where(y == 0)[0]
        if len(pos_idx) == 0:
            raise ValueError("PU Bagging requires at least one positive sample.")
        if len(unl_idx) == 0:
            raise ValueError("PU Bagging requires unlabeled (target=0) samples.")

        n_draw = max(1, int(round(self.sample_ratio * len(pos_idx))))
        n_draw = min(n_draw, len(unl_idx))
        rng = np.random.default_rng(self.random_state)

        base = self.base_estimator
        if base is None:
            base = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )

        self.estimators_: list[Any] = []
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        for bag in range(self.n_bags):
            drawn = rng.choice(unl_idx, size=n_draw, replace=False)
            idx = np.concatenate([pos_idx, drawn])
            rng.shuffle(idx)
            est = clone(base)
            # Ensure distinct seed per bag when estimator supports it
            if hasattr(est, "set_params") and "random_state" in est.get_params():
                est.set_params(random_state=self.random_state + bag + 1)
            est.fit(X[idx], y[idx])
            self.estimators_.append(est)

        logger.info(
            "PU Bagging fitted: %d bags, %d positives, %d unlabeled/draw",
            self.n_bags,
            len(pos_idx),
            n_draw,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("PUBaggingClassifier has not been fitted.")
        X = np.asarray(X)
        probs = np.mean(
            [est.predict_proba(X) for est in self.estimators_], axis=0
        )
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score_unlabeled(
        self,
        X_unlabeled: np.ndarray,
    ) -> np.ndarray:
        """Return mean positive-class probability for unlabeled rows."""
        return self.predict_proba(X_unlabeled)[:, 1]


def refine_unlabeled_with_pu_bagging(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_bags: int = 10,
    sample_ratio: float = 1.0,
    keep_fraction: float = 0.5,
    base_estimator: Any | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Drop the most landslide-like unlabeled points via PU Bagging scores.

    Keeps the bottom ``keep_fraction`` of unlabeled points by predicted
    landslide probability (most reliable absences), plus all positives.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    pos = df[df[TARGET_COLUMN] == 1].copy()
    neg = df[df[TARGET_COLUMN] == 0].copy()
    if neg.empty:
        return df.copy()

    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = PUBaggingClassifier(
        base_estimator=base_estimator,
        n_bags=n_bags,
        sample_ratio=sample_ratio,
        random_state=random_state,
    )
    clf.fit(X_s, y)

    neg_mask = y == 0
    scores = clf.predict_proba(X_s[neg_mask])[:, 1]
    n_keep = max(1, int(round(keep_fraction * len(scores))))
    # Lowest landslide probability = most reliable absences
    order = np.argsort(scores)
    keep_local = order[:n_keep]
    neg_kept = neg.iloc[keep_local]

    logger.info(
        "PU Bagging refine: kept %d / %d unlabeled (lowest P(landslide))",
        len(neg_kept),
        len(neg),
    )
    out = pd.concat([pos, neg_kept], ignore_index=True)
    return out.sample(frac=1, random_state=random_state).reset_index(drop=True)


def make_pu_bagging_estimator_factory(
    base_params: dict | None = None,
    n_bags: int = 10,
    sample_ratio: float = 1.0,
) -> Callable[..., PUBaggingClassifier]:
    """Return a constructor compatible with ``buffered_loocv`` / k-fold helpers."""

    params = base_params or {
        "n_estimators": 100,
        "class_weight": "balanced",
        "n_jobs": -1,
    }

    def factory(**kwargs: Any) -> PUBaggingClassifier:
        rs = kwargs.get("random_state", RANDOM_STATE)
        base = RandomForestClassifier(**{**params, "random_state": rs})
        return PUBaggingClassifier(
            base_estimator=base,
            n_bags=n_bags,
            sample_ratio=sample_ratio,
            random_state=rs,
        )

    return factory
