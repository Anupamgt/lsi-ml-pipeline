"""
feature_selection.py — Correlation filter, CART ranking, Genetic Algorithm.

NHESS-style (Labbé et al.) practical feature selection:

1. Drop highly correlated features (|r| ≥ threshold).
2. Rank remaining features with a CART (DecisionTree) impurity importance.
3. Optional Genetic Algorithm wrapper: fitness = CV error + n_features/100,
   evaluated with Random Forest or XGBoost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from lsi_pipeline.config import RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    dropped_correlated: list[str]
    cart_ranking: list[tuple[str, float]]
    ga_best_fitness: float | None = None
    method: str = "none"


def correlation_filter(
    X: np.ndarray,
    feature_names: Sequence[str],
    threshold: float = 0.85,
) -> tuple[list[str], list[str]]:
    """Remove one of each pair with |Pearson r| ≥ threshold.

    Keeps the first feature in each correlated pair (stable, deterministic).
    """
    names = list(feature_names)
    if X.shape[1] != len(names):
        raise ValueError("X columns must match feature_names length.")
    if X.shape[1] < 2:
        return names, []

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    keep = np.ones(len(names), dtype=bool)
    dropped: list[str] = []

    for i in range(len(names)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(names)):
            if not keep[j]:
                continue
            if abs(corr[i, j]) >= threshold:
                keep[j] = False
                dropped.append(names[j])
                logger.debug(
                    "Corr filter: drop %s (|r|=%.3f with %s)",
                    names[j],
                    corr[i, j],
                    names[i],
                )

    kept = [n for n, k in zip(names, keep) if k]
    logger.info(
        "Correlation filter (|r|≥%.2f): kept %d / %d (dropped %d)",
        threshold,
        len(kept),
        len(names),
        len(dropped),
    )
    return kept, dropped


def cart_feature_ranking(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    random_state: int = RANDOM_STATE,
) -> list[tuple[str, float]]:
    """Rank features by CART (DecisionTreeClassifier) impurity importance."""
    names = list(feature_names)
    tree = DecisionTreeClassifier(
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
    )
    tree.fit(X, y)
    importances = tree.feature_importances_
    ranking = sorted(
        zip(names, importances.tolist()),
        key=lambda t: t[1],
        reverse=True,
    )
    logger.info(
        "CART ranking top-5: %s",
        ", ".join(f"{n}={v:.3f}" for n, v in ranking[:5]),
    )
    return ranking


def _make_evaluator(
    evaluator: str,
    random_state: int,
) -> Any:
    """Build a cheap CV evaluator for GA fitness."""
    evaluator = evaluator.lower()
    if evaluator == "rf":
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if evaluator in ("xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for GA evaluator='xgb'. "
                "Install with: pip install xgboost"
            ) from exc
        return XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    if evaluator == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                probability=False,
                class_weight="balanced",
                random_state=random_state,
            ),
        )
    raise ValueError(f"Unknown GA evaluator: {evaluator!r} (use rf|xgb|svm)")


def _fitness(
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    n_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> float:
    """fitness = mean CV error + n_features/100  (lower is better)."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 1e6
    X_sub = X[:, idx]
    # Need at least 2 samples per class for stratified CV
    if len(np.unique(y)) < 2 or X_sub.shape[0] < n_splits * 2:
        return 1e6
    cv = StratifiedKFold(
        n_splits=min(n_splits, int(y.sum()), int((1 - y).sum()) or 1),
        shuffle=True,
        random_state=random_state,
    )
    try:
        scores = cross_val_score(
            estimator, X_sub, y, cv=cv, scoring="roc_auc", n_jobs=1
        )
        error = 1.0 - float(np.nanmean(scores))
    except Exception:  # noqa: BLE001 — GA must tolerate bad subsets
        error = 1.0
    return error + (len(idx) / 100.0)


def genetic_algorithm_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    population: int = 20,
    generations: int = 15,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.7,
    evaluator: str = "rf",
    random_state: int = RANDOM_STATE,
) -> tuple[list[str], float]:
    """Binary GA feature selection. Fitness = CV error + n_features/100."""
    names = list(feature_names)
    n_feat = len(names)
    if n_feat == 0:
        return [], 1e6
    if n_feat == 1:
        return names, 0.0

    rng = np.random.default_rng(random_state)
    est = _make_evaluator(evaluator, random_state)

    # Population of binary masks; ensure ≥1 feature selected
    pop = rng.integers(0, 2, size=(population, n_feat), dtype=int)
    for i in range(population):
        if pop[i].sum() == 0:
            pop[i, rng.integers(0, n_feat)] = 1

    best_mask = pop[0].copy()
    best_fit = _fitness(best_mask.astype(bool), X, y, est, random_state=random_state)

    for gen in range(generations):
        fits = np.array([
            _fitness(ind.astype(bool), X, y, est, random_state=random_state + gen)
            for ind in pop
        ])
        order = np.argsort(fits)
        if fits[order[0]] < best_fit:
            best_fit = float(fits[order[0]])
            best_mask = pop[order[0]].copy()

        # Elitism + tournament selection
        elite_n = max(1, population // 5)
        new_pop = [pop[i].copy() for i in order[:elite_n]]

        while len(new_pop) < population:
            # Tournament of size 3
            cands = rng.choice(population, size=3, replace=False)
            parent_a = pop[cands[np.argmin(fits[cands])]]
            cands = rng.choice(population, size=3, replace=False)
            parent_b = pop[cands[np.argmin(fits[cands])]]

            child = parent_a.copy()
            if rng.random() < crossover_rate:
                cut = int(rng.integers(1, n_feat))
                child = np.concatenate([parent_a[:cut], parent_b[cut:]])

            mut = rng.random(n_feat) < mutation_rate
            child[mut] = 1 - child[mut]
            if child.sum() == 0:
                child[rng.integers(0, n_feat)] = 1
            new_pop.append(child)

        pop = np.asarray(new_pop[:population], dtype=int)
        logger.debug(
            "GA gen %d/%d best_fitness=%.4f n_feat=%d",
            gen + 1,
            generations,
            best_fit,
            int(best_mask.sum()),
        )

    selected = [n for n, m in zip(names, best_mask) if m]
    logger.info(
        "GA selected %d / %d features (fitness=%.4f, evaluator=%s): %s",
        len(selected),
        n_feat,
        best_fit,
        evaluator,
        selected,
    )
    return selected, best_fit


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    method: str = "none",
    corr_threshold: float = 0.85,
    ga_population: int = 20,
    ga_generations: int = 15,
    ga_mutation_rate: float = 0.15,
    ga_crossover_rate: float = 0.7,
    ga_evaluator: str = "rf",
    random_state: int = RANDOM_STATE,
) -> FeatureSelectionResult:
    """Run the configured feature-selection pipeline.

    Methods
    -------
    none:
        Return all features unchanged.
    cart:
        Correlation filter → keep features ranked by CART (all survivors).
    ga:
        Correlation filter → Genetic Algorithm on remaining features.
    """
    method = (method or "none").lower()
    names = list(feature_names)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)

    if method == "none":
        return FeatureSelectionResult(
            selected_features=names,
            dropped_correlated=[],
            cart_ranking=[],
            method="none",
        )

    kept, dropped = correlation_filter(X, names, threshold=corr_threshold)
    col_index = {n: i for i, n in enumerate(names)}
    X_kept = X[:, [col_index[n] for n in kept]]

    ranking = cart_feature_ranking(X_kept, y, kept, random_state=random_state)

    if method == "cart":
        # Keep CART-ranked survivors (all after corr filter), ordered by importance
        selected = [n for n, _ in ranking]
        return FeatureSelectionResult(
            selected_features=selected,
            dropped_correlated=dropped,
            cart_ranking=ranking,
            method="cart",
        )

    if method == "ga":
        selected, best_fit = genetic_algorithm_select(
            X_kept,
            y,
            kept,
            population=ga_population,
            generations=ga_generations,
            mutation_rate=ga_mutation_rate,
            crossover_rate=ga_crossover_rate,
            evaluator=ga_evaluator,
            random_state=random_state,
        )
        return FeatureSelectionResult(
            selected_features=selected,
            dropped_correlated=dropped,
            cart_ranking=ranking,
            ga_best_fitness=best_fit,
            method="ga",
        )

    raise ValueError(f"Unknown feature-selection method: {method!r} (none|cart|ga)")


def subset_dataframe(
    df: pd.DataFrame,
    selected_features: Sequence[str],
) -> pd.DataFrame:
    """Return df with coords + selected features + target (if present)."""
    cols = ["x", "y", *selected_features]
    if "target" in df.columns:
        cols.append("target")
    return df[cols].copy()
