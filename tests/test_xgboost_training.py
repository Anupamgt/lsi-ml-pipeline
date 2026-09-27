"""Smoke test for XGBoost training helper."""

from __future__ import annotations

import numpy as np
import pytest

from lsi_pipeline.config import FEATURE_COLUMNS


@pytest.fixture()
def tiny_xy():
    rng = np.random.default_rng(1)
    n = 24
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    X[:12] += 1.5
    y = np.array([1] * 12 + [0] * 12)
    return X, y


def test_train_xgboost_saves(tmp_path, tiny_xy):
    pytest.importorskip("xgboost")
    from lsi_pipeline.model_training import train_xgboost

    X, y = tiny_xy
    out = tmp_path / "xgb_model.pkl"
    model = train_xgboost(X, y, out)
    assert out.exists()
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
