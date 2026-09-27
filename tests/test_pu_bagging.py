"""Smoke / unit tests for Buffer Control Sampling and PU Bagging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsi_pipeline.config import FEATURE_COLUMNS, TARGET_COLUMN
from lsi_pipeline.pu_bagging import (
    PUBaggingClassifier,
    apply_buffer_control_sampling,
    buffer_control_mask,
    pairwise_min_distance,
    refine_unlabeled_with_pu_bagging,
)


@pytest.fixture()
def toy_df():
    rng = np.random.default_rng(0)
    n = 20
    # Positives clustered near origin
    pos = {
        "x": rng.uniform(0, 100, n // 2),
        "y": rng.uniform(0, 100, n // 2),
        TARGET_COLUMN: np.ones(n // 2, dtype=int),
    }
    # Negatives: half near positives, half far away
    neg_near = {
        "x": rng.uniform(0, 80, n // 4),
        "y": rng.uniform(0, 80, n // 4),
    }
    neg_far = {
        "x": rng.uniform(2000, 3000, n // 4),
        "y": rng.uniform(2000, 3000, n // 4),
    }
    for col in FEATURE_COLUMNS:
        pos[col] = rng.uniform(1.5, 3.0, n // 2)
        neg_near[col] = rng.uniform(0.2, 1.0, n // 4)
        neg_far[col] = rng.uniform(0.2, 1.0, n // 4)
    neg_near[TARGET_COLUMN] = np.zeros(n // 4, dtype=int)
    neg_far[TARGET_COLUMN] = np.zeros(n // 4, dtype=int)
    return pd.concat(
        [pd.DataFrame(pos), pd.DataFrame(neg_near), pd.DataFrame(neg_far)],
        ignore_index=True,
    )


def test_pairwise_min_distance():
    q = np.array([[0.0, 0.0], [3.0, 4.0]])
    r = np.array([[0.0, 0.0]])
    d = pairwise_min_distance(q, r)
    np.testing.assert_allclose(d, [0.0, 5.0])


def test_buffer_control_mask():
    unl = np.array([[0.0, 0.0], [1000.0, 0.0]])
    pos = np.array([[0.0, 0.0]])
    mask = buffer_control_mask(unl, pos, inner_buffer_m=500.0)
    assert mask.tolist() == [False, True]


def test_apply_buffer_control_sampling(toy_df):
    out = apply_buffer_control_sampling(toy_df, inner_buffer_m=500.0)
    assert (out[TARGET_COLUMN] == 1).sum() == (toy_df[TARGET_COLUMN] == 1).sum()
    assert (out[TARGET_COLUMN] == 0).sum() < (toy_df[TARGET_COLUMN] == 0).sum()
    assert (out[TARGET_COLUMN] == 0).sum() >= 1


def test_pu_bagging_fit_predict(toy_df):
    X = toy_df[FEATURE_COLUMNS].values
    y = toy_df[TARGET_COLUMN].values
    clf = PUBaggingClassifier(n_bags=3, sample_ratio=1.0, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(toy_df), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    preds = clf.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_refine_unlabeled(toy_df):
    out = refine_unlabeled_with_pu_bagging(
        toy_df, n_bags=3, keep_fraction=0.5, random_state=0
    )
    assert (out[TARGET_COLUMN] == 1).sum() == (toy_df[TARGET_COLUMN] == 1).sum()
    assert (out[TARGET_COLUMN] == 0).sum() <= (toy_df[TARGET_COLUMN] == 0).sum()
    assert len(out) >= (toy_df[TARGET_COLUMN] == 1).sum() + 1
