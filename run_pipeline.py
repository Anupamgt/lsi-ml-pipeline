#!/usr/bin/env python3
"""
run_pipeline.py — Main entry point for the LSI ML Pipeline.

Orchestrates ingestion → optional PU/buffer sampling → feature selection →
CV (Buffered LOOCV + k-fold) → RF / LR / XGBoost training → reports / map.

Usage examples:
  python run_pipeline.py --input data/landslide_training_data.csv
  python run_pipeline.py --study-config configs/aizawl.yaml \\
      --input data/landslide_training_data.csv --grid data/aizawl_grid.csv \\
      --output results/aizawl_17fr --feature-selection cart --pu-bagging
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

from lsi_pipeline.study_area import apply_study_config, load_study_config


# ─── Logging setup ────────────────────────────────────────────────────────────

def _configure_stdio_utf8() -> None:
    """Avoid Windows cp1252 crashes on box-drawing / Unicode log chars."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


_configure_stdio_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("lsi_pipeline.run")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "Landslide Susceptibility Index (LSI) — ML Training & Validation Pipeline\n"
            "IIT Patna | Multi-study-area | RF + LR + XGBoost | NHESS-style upgrades"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--study-config",
        default="configs/aizawl.yaml",
        metavar="YAML",
        help="Study-area YAML (paths, bbox, CRS, buffers). Default: configs/aizawl.yaml",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        metavar="CSV_PATH",
        help="Training CSV (default: paths.training_csv from study config)",
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
        help="Optional grid CSV for susceptibility map prediction",
    )
    parser.add_argument(
        "--skip-loocv",
        action="store_true",
        default=False,
        help="Skip buffered LOOCV (uses only k-fold CV as primary metric).",
    )
    parser.add_argument(
        "--buffer-radius",
        type=float,
        default=None,
        metavar="METRES",
        help="Spatial exclusion buffer for LOOCV (default: from study config)",
    )
    parser.add_argument(
        "--feature-selection",
        choices=("none", "cart", "ga"),
        default="none",
        help="Feature selection: none | cart (corr+CART) | ga (corr+GA).",
    )
    parser.add_argument(
        "--pu-bagging",
        action="store_true",
        default=False,
        help="Apply Buffer Control Sampling + PU Bagging unlabeled refinement.",
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        default=False,
        help="Skip XGBoost training/evaluation.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ─── Pipeline steps ───────────────────────────────────────────────────────────


def step1_load(input_path: str):
    logger.info("━━ STEP 1 — Data Ingestion ━━")
    from lsi_pipeline.data_ingestion import load_and_validate
    return load_and_validate(input_path)


def step2_build_dataset(df, output_dir: Path):
    logger.info("━━ STEP 2 — Dataset Construction ━━")
    from lsi_pipeline.config import OUT_FULL_DATASET
    from lsi_pipeline.data_ingestion import build_full_dataset
    return build_full_dataset(df, output_dir / OUT_FULL_DATASET)


def step2b_pu_refine(df, study_cfg, enabled: bool):
    """Optional Buffer Control Sampling + PU Bagging refinement."""
    if not enabled:
        return df
    logger.info("━━ STEP 2b — Buffer Control + PU Bagging ━━")
    from lsi_pipeline.pu_bagging import (
        apply_buffer_control_sampling,
        refine_unlabeled_with_pu_bagging,
    )

    pu = study_cfg.pu_bagging
    df = apply_buffer_control_sampling(
        df,
        inner_buffer_m=pu.inner_buffer_m,
        outer_buffer_m=pu.outer_buffer_m,
    )
    df = refine_unlabeled_with_pu_bagging(
        df,
        n_bags=pu.n_bags,
        sample_ratio=pu.sample_ratio,
        keep_fraction=0.75,
    )
    return df


def step2c_feature_selection(df, method: str, study_cfg, output_dir: Path):
    """Run feature selection and update config.FEATURE_COLUMNS in-place."""
    import lsi_pipeline.config as cfg
    from lsi_pipeline.config import OUT_FEATURE_SELECTION, TARGET_COLUMN
    from lsi_pipeline.feature_selection import select_features

    logger.info("━━ STEP 2c — Feature Selection (%s) ━━", method)
    X = df[cfg.FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    fs = study_cfg.feature_selection
    result = select_features(
        X, y, cfg.FEATURE_COLUMNS,
        method=method,
        corr_threshold=fs.corr_threshold,
        ga_population=fs.ga_population,
        ga_generations=fs.ga_generations,
        ga_mutation_rate=fs.ga_mutation_rate,
        ga_crossover_rate=fs.ga_crossover_rate,
        ga_evaluator=fs.ga_evaluator,
        random_state=cfg.RANDOM_STATE,
    )
    cfg.FEATURE_COLUMNS = list(result.selected_features)

    report_path = output_dir / OUT_FEATURE_SELECTION
    lines = [
        f"method: {result.method}",
        f"selected ({len(result.selected_features)}): {result.selected_features}",
        f"dropped_correlated: {result.dropped_correlated}",
        f"cart_ranking: {result.cart_ranking}",
        f"ga_best_fitness: {result.ga_best_fitness}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Feature selection report → %s", report_path)
    return df, result


def step3_prepare_arrays(df):
    import lsi_pipeline.config as cfg
    X = df[cfg.FEATURE_COLUMNS].values
    y = df[cfg.TARGET_COLUMN].values
    coords = df[["x", "y"]].values
    return X, y, coords


def step4_run_cv(df, X, y, skip_loocv: bool, buffer_radius: float, run_xgb: bool) -> dict:
    """Run LOOCV and stratified K-fold CV for RF, LR, and optionally XGB."""
    from lsi_pipeline.config import LR_PARAMS, N_FOLDS, RF_PARAMS
    from lsi_pipeline.model_training import get_xgb_sklearn_params
    from lsi_pipeline.preprocessing import buffered_loocv, stratified_kfold_cv

    logger.info("━━ STEP 3/5 — Cross-Validation ━━")
    results: dict = {}

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

    if run_xgb:
        from xgboost import XGBClassifier
        xgb_params = get_xgb_sklearn_params()
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        xgb_params["scale_pos_weight"] = n_neg / n_pos
        logger.info("Running Stratified %d-Fold CV — XGBoost …", N_FOLDS)
        xgb_kf_mean, xgb_kf_std, xgb_kf_folds = stratified_kfold_cv(
            X, y, XGBClassifier, xgb_params, n_splits=N_FOLDS
        )
        results.update({
            "xgb_kfold_mean": xgb_kf_mean,
            "xgb_kfold_std": xgb_kf_std,
            "xgb_kfold_folds": xgb_kf_folds,
            "_xgb_params": xgb_params,
        })

    nan_arr = np.array([])
    if skip_loocv:
        logger.warning("--skip-loocv flag set: skipping buffered LOOCV.")
        results.update({
            "rf_loocv_auc": float("nan"),
            "rf_loocv_true": nan_arr,
            "rf_loocv_prob": nan_arr,
            "lr_loocv_auc": float("nan"),
            "lr_loocv_true": nan_arr,
            "lr_loocv_prob": nan_arr,
        })
        if run_xgb:
            results.update({
                "xgb_loocv_auc": float("nan"),
                "xgb_loocv_true": nan_arr,
                "xgb_loocv_prob": nan_arr,
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

        if run_xgb:
            from xgboost import XGBClassifier
            xgb_params = results["_xgb_params"]
            logger.info("Running Buffered LOOCV (r=%.0fm) — XGBoost …", buffer_radius)
            t0 = time.time()
            xgb_loocv_auc, xgb_loocv_true, xgb_loocv_prob = buffered_loocv(
                df, XGBClassifier, xgb_params, buffer_radius=buffer_radius
            )
            logger.info("XGB LOOCV done in %.1f s", time.time() - t0)
            results.update({
                "xgb_loocv_auc": xgb_loocv_auc,
                "xgb_loocv_true": xgb_loocv_true,
                "xgb_loocv_prob": xgb_loocv_prob,
            })

    return results


def step5_train_final_models(X, y, output_dir: Path, run_xgb: bool):
    from lsi_pipeline.config import (
        OUT_LR_MODEL,
        OUT_RF_MODEL,
        OUT_SCALER,
        OUT_XGB_MODEL,
    )
    from lsi_pipeline.model_training import (
        train_logistic_regression,
        train_random_forest,
        train_xgboost,
    )
    from lsi_pipeline.preprocessing import fit_full_scaler

    logger.info("━━ STEP 4 — Final Model Training ━━")
    scaler = fit_full_scaler(X, output_dir / OUT_SCALER)
    X_scaled = scaler.transform(X)
    rf_model = train_random_forest(X_scaled, y, output_dir / OUT_RF_MODEL)
    lr_model = train_logistic_regression(X_scaled, y, output_dir / OUT_LR_MODEL)
    xgb_model = None
    if run_xgb:
        xgb_model = train_xgboost(X_scaled, y, output_dir / OUT_XGB_MODEL)
    return X_scaled, rf_model, lr_model, xgb_model, scaler


def step6_evaluate_and_report(
    rf_model, lr_model, xgb_model,
    X_scaled, y, cv_results, output_dir: Path,
    buffer_radius: float, skip_loocv: bool, study_name: str,
    selected_features: list[str], extra_notes: list[str],
) -> None:
    from lsi_pipeline.config import FEATURE_COLUMNS, N_FOLDS, OUT_METRICS, OUT_ROC_PLOT
    from lsi_pipeline.evaluation import (
        compute_full_dataset_metrics,
        plot_roc_comparison,
        write_metrics_report,
    )

    logger.info("━━ STEP 5 — Evaluation & Reporting ━━")
    rf_metrics = compute_full_dataset_metrics(rf_model, X_scaled, y, "Random Forest")
    lr_metrics = compute_full_dataset_metrics(lr_model, X_scaled, y, "Logistic Regression")
    xgb_metrics = None
    if xgb_model is not None:
        xgb_metrics = compute_full_dataset_metrics(xgb_model, X_scaled, y, "XGBoost")

    xgb_kwargs = {}
    if xgb_model is not None:
        xgb_kwargs = {
            "xgb_loocv_true": cv_results.get("xgb_loocv_true"),
            "xgb_loocv_prob": cv_results.get("xgb_loocv_prob"),
            "xgb_loocv_auc": cv_results.get("xgb_loocv_auc"),
            "xgb_kfold_mean": cv_results.get("xgb_kfold_mean"),
            "xgb_kfold_std": cv_results.get("xgb_kfold_std"),
        }

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
            study_name=study_name,
            **xgb_kwargs,
        )
    else:
        logger.warning("LOOCV skipped — ROC plot uses full-dataset proba (not CV-based).")
        _plot_roc_fulldataset_fallback(
            rf_model, lr_model, xgb_model, X_scaled, y,
            cv_results, output_dir / OUT_ROC_PLOT, study_name,
        )

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
        xgb_loocv_auc=cv_results.get("xgb_loocv_auc"),
        xgb_kfold_mean=cv_results.get("xgb_kfold_mean"),
        xgb_kfold_std=cv_results.get("xgb_kfold_std"),
        xgb_kfold_folds=cv_results.get("xgb_kfold_folds"),
        xgb_metrics=xgb_metrics,
        selected_features=selected_features,
        study_name=study_name,
        extra_notes=extra_notes,
    )


def _plot_roc_fulldataset_fallback(
    rf_model, lr_model, xgb_model,
    X_scaled, y, cv_results, output_path, study_name,
) -> None:
    from sklearn.metrics import roc_auc_score
    from lsi_pipeline.evaluation import plot_roc_comparison

    rf_prob = rf_model.predict_proba(X_scaled)[:, 1]
    lr_prob = lr_model.predict_proba(X_scaled)[:, 1]
    kwargs = {}
    if xgb_model is not None:
        xgb_prob = xgb_model.predict_proba(X_scaled)[:, 1]
        kwargs = {
            "xgb_loocv_true": y,
            "xgb_loocv_prob": xgb_prob,
            "xgb_loocv_auc": roc_auc_score(y, xgb_prob),
            "xgb_kfold_mean": cv_results.get("xgb_kfold_mean"),
            "xgb_kfold_std": cv_results.get("xgb_kfold_std"),
        }
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
        study_name=study_name,
        **kwargs,
    )


def step7_feature_importance(rf_model, output_dir: Path) -> None:
    from lsi_pipeline.config import OUT_FI_PLOT
    from lsi_pipeline.feature_importance import (
        plot_feature_importance,
        print_feature_importance_table,
    )
    logger.info("━━ STEP 6 — Feature Importance ━━")
    print_feature_importance_table(rf_model)
    plot_feature_importance(rf_model, output_dir / OUT_FI_PLOT)


def step8_susceptibility_map(grid_path: str | None, output_dir: Path) -> None:
    if grid_path is None:
        logger.info("No --grid CSV provided — skipping susceptibility map generation.")
        return
    from lsi_pipeline.config import OUT_RF_MODEL, OUT_SCALER
    from lsi_pipeline.susceptibility_map import predict_susceptibility
    logger.info("━━ STEP 7 — Susceptibility Map ━━")
    predict_susceptibility(
        grid_csv_path=grid_path,
        rf_model_path=output_dir / OUT_RF_MODEL,
        scaler_path=output_dir / OUT_SCALER,
        output_dir=output_dir,
    )


def _write_run_metadata(
    output_dir: Path,
    study_cfg,
    args,
    selected_features: list[str],
    elapsed: float,
) -> None:
    from lsi_pipeline.config import OUT_RUN_META
    lines = [
        f"study_area: {study_cfg.name}",
        f"study_config: {study_cfg.config_path}",
        f"crs: {study_cfg.crs}",
        f"input: {args.input}",
        f"grid: {args.grid}",
        f"feature_selection: {args.feature_selection}",
        f"pu_bagging: {args.pu_bagging}",
        f"skip_loocv: {args.skip_loocv}",
        f"skip_xgboost: {args.skip_xgboost}",
        f"buffer_radius_m: {args.buffer_radius}",
        f"n_features: {len(selected_features)}",
        f"features: {selected_features}",
        f"elapsed_s: {elapsed:.1f}",
        "",
        "Model pickles (rf_model.pkl, lr_model.pkl, xgb_model.pkl, scaler.pkl)",
        "are gitignored. Regenerate with:",
        f"  python run_pipeline.py --study-config {args.study_config} \\",
        f"    --input {args.input} --output {args.output}"
        + (f" --grid {args.grid}" if args.grid else ""),
    ]
    (output_dir / OUT_RUN_META).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Main pipeline orchestrator."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    study_cfg = apply_study_config(load_study_config(args.study_config))

    # Resolve defaults from study config after apply
    import lsi_pipeline.config as cfg

    if args.input is None:
        args.input = str(study_cfg.training_csv)
    if args.buffer_radius is None:
        args.buffer_radius = study_cfg.buffer_radius
    if args.grid is None and study_cfg.grid_csv.exists():
        # Do not auto-enable grid; only use explicit --grid.
        pass

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_xgb = not args.skip_xgboost
    if run_xgb:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            logger.warning("xgboost not installed — skipping XGBoost. pip install xgboost")
            run_xgb = False

    total_start = time.time()

    print("\n" + "╔" + "═" * 60 + "╗")
    print("║  LSI ML PIPELINE — Landslide Susceptibility Index         ║")
    print(f"║  Study: {study_cfg.name:<20} CRS: {study_cfg.crs:<18} ║")
    print("╚" + "═" * 60 + "╝\n")
    print(f"  Active features  : {len(cfg.FEATURE_COLUMNS)}")
    print(f"  Feature list     : {cfg.FEATURE_COLUMNS}")
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  LOOCV buffer     : {args.buffer_radius:.0f} m")
    print(f"  Feature select   : {args.feature_selection}")
    print(f"  PU bagging       : {args.pu_bagging}")
    print(f"  XGBoost          : {run_xgb}")
    print(f"  Skip LOOCV       : {args.skip_loocv}")
    print()

    df_raw = step1_load(args.input)
    df_full = step2_build_dataset(df_raw, output_dir)
    df_full = step2b_pu_refine(df_full, study_cfg, enabled=args.pu_bagging)
    df_full, fs_result = step2c_feature_selection(
        df_full, args.feature_selection, study_cfg, output_dir
    )

    X, y, _coords = step3_prepare_arrays(df_full)

    cv_results = step4_run_cv(
        df_full, X, y,
        skip_loocv=args.skip_loocv,
        buffer_radius=args.buffer_radius,
        run_xgb=run_xgb,
    )

    X_scaled, rf_model, lr_model, xgb_model, _scaler = step5_train_final_models(
        X, y, output_dir, run_xgb=run_xgb
    )

    extra_notes = [
        f"feature_selection={fs_result.method}",
        f"pu_bagging={args.pu_bagging}",
    ]
    step6_evaluate_and_report(
        rf_model, lr_model, xgb_model, X_scaled, y,
        cv_results, output_dir,
        buffer_radius=args.buffer_radius,
        skip_loocv=args.skip_loocv,
        study_name=study_cfg.name,
        selected_features=list(cfg.FEATURE_COLUMNS),
        extra_notes=extra_notes,
    )

    step7_feature_importance(rf_model, output_dir)
    step8_susceptibility_map(args.grid, output_dir)

    elapsed = time.time() - total_start
    _write_run_metadata(output_dir, study_cfg, args, list(cfg.FEATURE_COLUMNS), elapsed)

    print(f"\n✓  Pipeline complete in {elapsed:.1f} s")
    print(f"   All outputs written to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
