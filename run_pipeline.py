#!/usr/bin/env python3
"""
run_pipeline.py — Main entry point for the LSI ML Pipeline.

Orchestrates all 7 steps end-to-end via argparse CLI.

Usage examples:
  # Run with only 3 currently available features (no grid):
  python run_pipeline.py --input data/landslide_training_data.csv

  # Run with pre-built full dataset + susceptibility grid:
  python run_pipeline.py \\
      --input data/landslide_training_data.csv \\
      --grid  data/aizawl_grid.csv

  # Specify custom output directory:
  python run_pipeline.py \\
      --input data/landslide_training_data.csv \\
      --output /path/to/outputs/

  # Skip LOOCV (fast mode, 5-fold only):
  python run_pipeline.py --input data/landslide_training_data.csv --skip-loocv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lsi_pipeline import config
from lsi_pipeline.config import (
    BUFFER_RADIUS,
    FEATURE_COLUMNS,
    LR_PARAMS,
    N_FOLDS,
    OUT_FI_PLOT,
    OUT_LR_MODEL,
    OUT_METRICS,
    OUT_RF_MODEL,
    OUT_ROC_PLOT,
    OUT_SCALER,
    OUT_FULL_DATASET,
    RF_PARAMS,
    TARGET_COLUMN,
)
from lsi_pipeline.data_ingestion import build_full_dataset, load_and_validate
from lsi_pipeline.evaluation import (
    compute_full_dataset_metrics,
    plot_roc_comparison,
    write_metrics_report,
)
from lsi_pipeline.feature_importance import (
    plot_feature_importance,
    print_feature_importance_table,
)
from lsi_pipeline.model_training import train_logistic_regression, train_random_forest
from lsi_pipeline.preprocessing import (
    buffered_loocv,
    fit_full_scaler,
    stratified_kfold_cv,
)
from lsi_pipeline.susceptibility_map import predict_susceptibility


# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("lsi_pipeline.run")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "Landslide Susceptibility Index (LSI) — ML Training & Validation Pipeline\n"
            "IIT Patna | Aizawl, Mizoram | Random Forest + Logistic Regression"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="CSV_PATH",
        help="Path to landslide_training_data.csv "
             "(cols: x, y, [FR features], target)",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        metavar="OUTPUT_DIR",
        help="Directory for all output files (default: ./outputs/)",
    )
    parser.add_argument(
        "--grid", "-g",
        default=None,
        metavar="GRID_CSV",
        help="Optional: aizawl_grid.csv for susceptibility map prediction "
             "(cols: x, y, [FR features])",
    )
    parser.add_argument(
        "--skip-loocv",
        action="store_true",
        default=False,
        help="Skip buffered LOOCV (faster, uses only 5-fold CV as primary metric). "
             "Useful for large datasets or quick sanity checks.",
    )
    parser.add_argument(
        "--buffer-radius",
        type=float,
        default=BUFFER_RADIUS,
        metavar="METRES",
        help=f"Spatial exclusion buffer for LOOCV (default: {BUFFER_RADIUS:.0f} m)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ─── Pipeline steps ───────────────────────────────────────────────────────────


def step1_load(input_path: str) -> "pd.DataFrame":
    """Step 1: Load and validate the training CSV."""
    logger.info("━━ STEP 1 — Data Ingestion ━━")
    from lsi_pipeline.data_ingestion import load_and_validate  # local import for clarity
    return load_and_validate(input_path)


def step2_build_dataset(df, output_dir: Path) -> "pd.DataFrame":
    """Step 2: Build full dataset (generate pseudo-absences if needed)."""
    logger.info("━━ STEP 2 — Dataset Construction ━━")
    return build_full_dataset(df, output_dir / OUT_FULL_DATASET)


def step3_prepare_arrays(df) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Step 3a: Extract numpy arrays from the full dataset."""
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    coords = df[["x", "y"]].values
    return X, y, coords


def step4_run_cv(
    df,
    X: np.ndarray,
    y: np.ndarray,
    skip_loocv: bool,
    buffer_radius: float,
) -> dict:
    """Step 4: Run LOOCV and stratified K-fold CV for both models.

    Parameters
    ----------
    df:
        Full dataset DataFrame (required for spatial LOOCV).
    X:
        Feature matrix.
    y:
        Target vector.
    skip_loocv:
        If True, LOOCV is skipped and NaN placeholders are returned.
    buffer_radius:
        Spatial exclusion buffer (metres) for LOOCV.

    Returns
    -------
    dict
        Keys: rf_loocv_auc, rf_loocv_true, rf_loocv_prob,
              rf_kfold_mean, rf_kfold_std, rf_kfold_folds,
              lr_loocv_auc, lr_loocv_true, lr_loocv_prob,
              lr_kfold_mean, lr_kfold_std, lr_kfold_folds.
    """
    logger.info("━━ STEP 3/5 — Cross-Validation ━━")

    results: dict = {}

    # Override BUFFER_RADIUS from CLI arg
    if buffer_radius != BUFFER_RADIUS:
        logger.info("Using CLI buffer radius: %.0f m", buffer_radius)

    # ── 5-Fold Stratified CV ─────────────────────────────────────────────
    logger.info("Running Stratified %d-Fold CV — Random Forest …", N_FOLDS)
    rf_kf_mean, rf_kf_std, rf_kf_folds = stratified_kfold_cv(
        X, y, RandomForestClassifier, RF_PARAMS, n_splits=N_FOLDS
    )
    results.update({
        "rf_kfold_mean": rf_kf_mean,
        "rf_kfold_std": rf_kf_std,
        "rf_kfold_folds": rf_kf_folds,
    })

    logger.info("Running Stratified %d-Fold CV — Logistic Regression …", N_FOLDS)
    lr_kf_mean, lr_kf_std, lr_kf_folds = stratified_kfold_cv(
        X, y, LogisticRegression, LR_PARAMS, n_splits=N_FOLDS
    )
    results.update({
        "lr_kfold_mean": lr_kf_mean,
        "lr_kfold_std": lr_kf_std,
        "lr_kfold_folds": lr_kf_folds,
    })

    # ── Buffered LOOCV ────────────────────────────────────────────────────
    if skip_loocv:
        logger.warning("--skip-loocv flag set: skipping buffered LOOCV.")
        nan_arr = np.array([])
        results.update({
            "rf_loocv_auc": float("nan"),
            "rf_loocv_true": nan_arr,
            "rf_loocv_prob": nan_arr,
            "lr_loocv_auc": float("nan"),
            "lr_loocv_true": nan_arr,
            "lr_loocv_prob": nan_arr,
        })
    else:
        logger.info("Running Buffered LOOCV (r=%.0fm) — Random Forest …", buffer_radius)
        t0 = time.time()
        rf_loocv_auc, rf_loocv_true, rf_loocv_prob = buffered_loocv(
            df, RandomForestClassifier, RF_PARAMS, buffer_radius=buffer_radius
        )
        logger.info("RF LOOCV done in %.1f s", time.time() - t0)
        results.update({
            "rf_loocv_auc": rf_loocv_auc,
            "rf_loocv_true": rf_loocv_true,
            "rf_loocv_prob": rf_loocv_prob,
        })

        logger.info("Running Buffered LOOCV (r=%.0fm) — Logistic Regression …", buffer_radius)
        t0 = time.time()
        lr_loocv_auc, lr_loocv_true, lr_loocv_prob = buffered_loocv(
            df, LogisticRegression, LR_PARAMS, buffer_radius=buffer_radius
        )
        logger.info("LR LOOCV done in %.1f s", time.time() - t0)
        results.update({
            "lr_loocv_auc": lr_loocv_auc,
            "lr_loocv_true": lr_loocv_true,
            "lr_loocv_prob": lr_loocv_prob,
        })

    return results


def step5_train_final_models(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
) -> "tuple":
    """Step 5: Fit scaler and train final models on full dataset.

    Parameters
    ----------
    X:
        Full feature matrix (unscaled).
    y:
        Full target vector.
    output_dir:
        Output directory for artefacts.

    Returns
    -------
    tuple
        (X_scaled, rf_model, lr_model, scaler)
    """
    logger.info("━━ STEP 4 — Final Model Training ━━")

    scaler = fit_full_scaler(X, output_dir / OUT_SCALER)
    X_scaled = scaler.transform(X)

    rf_model = train_random_forest(X_scaled, y, output_dir / OUT_RF_MODEL)
    lr_model = train_logistic_regression(X_scaled, y, output_dir / OUT_LR_MODEL)

    return X_scaled, rf_model, lr_model, scaler


def step6_evaluate_and_report(
    rf_model, lr_model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    cv_results: dict,
    output_dir: Path,
    buffer_radius: float,
    skip_loocv: bool,
) -> None:
    """Step 6: Compute full-dataset metrics, plot ROC, write report."""
    logger.info("━━ STEP 5 — Evaluation & Reporting ━━")

    rf_metrics = compute_full_dataset_metrics(rf_model, X_scaled, y, "Random Forest")
    lr_metrics = compute_full_dataset_metrics(lr_model, X_scaled, y, "Logistic Regression")

    # ── ROC Plot ─────────────────────────────────────────────────────────
    if not skip_loocv and len(cv_results["rf_loocv_true"]) > 0:
        plot_roc_comparison(
            rf_loocv_true=cv_results["rf_loocv_true"],
            rf_loocv_prob=cv_results["rf_loocv_prob"],
            rf_loocv_auc=cv_results["rf_loocv_auc"],
            rf_kfold_mean=cv_results["rf_kfold_mean"],
            rf_kfold_std=cv_results["rf_kfold_std"],
            lr_loocv_true=cv_results["lr_loocv_true"],
            lr_loocv_prob=cv_results["lr_loocv_prob"],
            lr_loocv_auc=cv_results["lr_loocv_auc"],
            lr_kfold_mean=cv_results["lr_kfold_mean"],
            lr_kfold_std=cv_results["lr_kfold_std"],
            output_path=output_dir / OUT_ROC_PLOT,
        )
    else:
        logger.warning("LOOCV skipped — ROC plot uses full-dataset proba (not CV-based).")
        _plot_roc_fulldataset_fallback(
            rf_model, lr_model, X_scaled, y,
            cv_results, output_dir / OUT_ROC_PLOT
        )

    # ── Metrics Report ────────────────────────────────────────────────────
    write_metrics_report(
        rf_loocv_auc=cv_results["rf_loocv_auc"],
        rf_kfold_mean=cv_results["rf_kfold_mean"],
        rf_kfold_std=cv_results["rf_kfold_std"],
        rf_kfold_folds=cv_results["rf_kfold_folds"],
        rf_metrics=rf_metrics,
        lr_loocv_auc=cv_results["lr_loocv_auc"],
        lr_kfold_mean=cv_results["lr_kfold_mean"],
        lr_kfold_std=cv_results["lr_kfold_std"],
        lr_kfold_folds=cv_results["lr_kfold_folds"],
        lr_metrics=lr_metrics,
        buffer_radius=buffer_radius,
        n_folds=N_FOLDS,
        n_features=len(FEATURE_COLUMNS),
        output_path=output_dir / OUT_METRICS,
    )


def _plot_roc_fulldataset_fallback(
    rf_model, lr_model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    cv_results: dict,
    output_path: Path,
) -> None:
    """Fallback ROC plot using full-dataset probabilities when LOOCV is skipped."""
    rf_prob = rf_model.predict_proba(X_scaled)[:, 1]
    lr_prob = lr_model.predict_proba(X_scaled)[:, 1]
    from sklearn.metrics import roc_auc_score
    plot_roc_comparison(
        rf_loocv_true=y,
        rf_loocv_prob=rf_prob,
        rf_loocv_auc=roc_auc_score(y, rf_prob),
        rf_kfold_mean=cv_results["rf_kfold_mean"],
        rf_kfold_std=cv_results["rf_kfold_std"],
        lr_loocv_true=y,
        lr_loocv_prob=lr_prob,
        lr_loocv_auc=roc_auc_score(y, lr_prob),
        lr_kfold_mean=cv_results["lr_kfold_mean"],
        lr_kfold_std=cv_results["lr_kfold_std"],
        output_path=output_path,
    )


def step7_feature_importance(rf_model, output_dir: Path) -> None:
    """Step 7: Feature importance plot and table."""
    logger.info("━━ STEP 6 — Feature Importance ━━")
    print_feature_importance_table(rf_model)
    plot_feature_importance(rf_model, output_dir / OUT_FI_PLOT)


def step8_susceptibility_map(
    grid_path: str | None,
    output_dir: Path,
) -> None:
    """Step 8: Susceptibility map prediction (optional)."""
    if grid_path is None:
        logger.info("No --grid CSV provided — skipping susceptibility map generation.")
        return
    logger.info("━━ STEP 7 — Susceptibility Map ━━")
    predict_susceptibility(
        grid_csv_path=grid_path,
        rf_model_path=output_dir / OUT_RF_MODEL,
        scaler_path=output_dir / OUT_SCALER,
        output_dir=output_dir,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Main pipeline orchestrator."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    print("\n" + "╔" + "═" * 60 + "╗")
    print("║  LSI ML PIPELINE — Landslide Susceptibility Index         ║")
    print("║  IIT Patna | Aizawl, Mizoram | UTM CRS                   ║")
    print("╚" + "═" * 60 + "╝\n")
    print(f"  Active features  : {len(FEATURE_COLUMNS)}")
    print(f"  Feature list     : {FEATURE_COLUMNS}")
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  LOOCV buffer     : {args.buffer_radius:.0f} m")
    print(f"  Skip LOOCV       : {args.skip_loocv}")
    print()

    # Steps
    df_raw = step1_load(args.input)
    df_full = step2_build_dataset(df_raw, output_dir)

    X, y, coords = step3_prepare_arrays(df_full)

    cv_results = step4_run_cv(
        df_full, X, y,
        skip_loocv=args.skip_loocv,
        buffer_radius=args.buffer_radius,
    )

    X_scaled, rf_model, lr_model, scaler = step5_train_final_models(X, y, output_dir)

    step6_evaluate_and_report(
        rf_model, lr_model, X_scaled, y,
        cv_results, output_dir,
        buffer_radius=args.buffer_radius,
        skip_loocv=args.skip_loocv,
    )

    step7_feature_importance(rf_model, output_dir)
    step8_susceptibility_map(args.grid, output_dir)

    elapsed = time.time() - total_start
    print(f"\n✓  Pipeline complete in {elapsed:.1f} s")
    print(f"   All outputs written to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
