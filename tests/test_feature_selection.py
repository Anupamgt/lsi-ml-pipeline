"""Unit tests for correlation filter, CART ranking, and GA selection."""

from __future__ import annotations

import numpy as np
import pytest

from lsi_pipeline.feature_selection import (
    cart_feature_ranking,
    correlation_filter,
    genetic_algorithm_select,
    select_features,
)


@pytest.fixture()
def separable_xy():
    rng = np.random.default_rng(42)
    n = 40
    # 5 informative-ish features + 1 duplicate of feature 0
    X = rng.normal(size=(n, 6))
    X[:20, 0] += 2.0
    X[20:, 0] -= 2.0
    X[:, 5] = X[:, 0] + rng.normal(0, 0.01, n)  # nearly perfect correlate
    y = np.array([1] * 20 + [0] * 20)
    names = [f"f{i}" for i in range(6)]
    return X, y, names


def test_correlation_filter_drops_duplicate(separable_xy):
    X, _, names = separable_xy
    kept, dropped = correlation_filter(X, names, threshold=0.9)
    assert "f5" in dropped or "f0" in dropped
    assert len(kept) + len(dropped) == len(names)
    assert len(kept) < len(names)


def test_cart_ranking_length(separable_xy):
    X, y, names = separable_xy
    ranking = cart_feature_ranking(X, y, names)
    assert len(ranking) == len(names)
    assert all(isinstance(s, float) for _, s in ranking)


def test_ga_selects_subset(separable_xy):
    X, y, names = separable_xy
    # Use tiny GA for speed
    selected, fitness = genetic_algorithm_select(
        X, y, names,
        population=8,
        generations=4,
        evaluator="rf",
        random_state=0,
    )
    assert 1 <= len(selected) <= len(names)
    assert fitness < 1e5
    assert all(s in names for s in selected)


def test_select_features_none(separable_xy):
    X, y, names = separable_xy
    result = select_features(X, y, names, method="none")
    assert result.selected_features == names
    assert result.method == "none"


def test_select_features_cart(separable_xy):
    X, y, names = separable_xy
    result = select_features(X, y, names, method="cart", corr_threshold=0.9)
    assert result.method == "cart"
    assert len(result.selected_features) < len(names) or result.dropped_correlated
    assert result.cart_ranking


def test_select_features_ga(separable_xy):
    X, y, names = separable_xy
    result = select_features(
        X, y, names,
        method="ga",
        corr_threshold=0.9,
        ga_population=6,
        ga_generations=3,
        ga_evaluator="rf",
    )
    assert result.method == "ga"
    assert result.ga_best_fitness is not None
    assert len(result.selected_features) >= 1
