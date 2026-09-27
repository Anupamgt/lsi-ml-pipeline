"""
evaluation.py — Step 5: Validation metrics, ROC plotting, and report generation.

Provides:
  • compute_full_dataset_metrics()   — Accuracy / Precision / Recall / F1 / CM
  • plot_roc_comparison()            — Overlaid ROC curves with CV annotation
  • write_metrics_report()           — Formatted text report (box drawing chars)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # non-interactive backend for headless execution

logger = logging.getLogger(__name__)


# ─── Full-dataset metrics ─────────────────────────────────────────────────────


def compute_full_dataset_metrics(
    model,
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Compute classification metrics on the full (scaled) dataset."""
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    print(f"\n{'─' * 50}")
    print(f"  {model_name} — Full-Dataset Metrics (training fit)")
    print(f"{'─' * 50}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1-Score   : {metrics['f1']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    return metrics


# ─── ROC Curve Plot ───────────────────────────────────────────────────────────


def plot_roc_comparison(
    rf_loocv_true: np.ndarray,
    rf_loocv_prob: np.ndarray,
    rf_loocv_auc: float,
    rf_kfold_mean: float,
    rf_kfold_std: float,
    lr_loocv_true: np.ndarray,
    lr_loocv_prob: np.ndarray,
    lr_loocv_auc: float,
    lr_kfold_mean: float,
    lr_kfold_std: float,
    output_path: str | Path,
    xgb_loocv_true: np.ndarray | None = None,
    xgb_loocv_prob: np.ndarray | None = None,
    xgb_loocv_auc: float | None = None,
    xgb_kfold_mean: float | None = None,
    xgb_kfold_std: float | None = None,
    study_name: str = "Aizawl, Mizoram",
) -> None:
    """Plot overlaid LOOCV ROC curves for RF, LR, and optionally XGBoost."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    fpr_rf, tpr_rf, _ = roc_curve(rf_loocv_true, rf_loocv_prob)
    ax.plot(
        fpr_rf, tpr_rf,
        color="#4fc3f7",
        lw=2.5,
        label=f"Random Forest — LOOCV AUC = {rf_loocv_auc:.3f}",
    )

    fpr_lr, tpr_lr, _ = roc_curve(lr_loocv_true, lr_loocv_prob)
    ax.plot(
        fpr_lr, tpr_lr,
        color="#ef9a9a",
        lw=2.5,
        linestyle="--",
        label=f"Logistic Regression — LOOCV AUC = {lr_loocv_auc:.3f}",
    )

    if (
        xgb_loocv_true is not None
        and xgb_loocv_prob is not None
        and xgb_loocv_auc is not None
        and len(xgb_loocv_true) > 0
    ):
        fpr_xgb, tpr_xgb, _ = roc_curve(xgb_loocv_true, xgb_loocv_prob)
        ax.plot(
            fpr_xgb, tpr_xgb,
            color="#a5d6a7",
            lw=2.5,
            linestyle="-.",
            label=f"XGBoost — LOOCV AUC = {xgb_loocv_auc:.3f}",
        )

    ax.plot(
        [0, 1], [0, 1],
        color="#555770",
        lw=1.2,
        linestyle=":",
        label="Random Classifier (AUC = 0.500)",
    )

    cv_lines = [
        "5-Fold Stratified CV (stability)",
        f"  RF : {rf_kfold_mean:.3f} ± {rf_kfold_std:.3f}",
        f"  LR : {lr_kfold_mean:.3f} ± {lr_kfold_std:.3f}",
    ]
    if xgb_kfold_mean is not None and xgb_kfold_std is not None:
        cv_lines.append(f"  XGB: {xgb_kfold_mean:.3f} ± {xgb_kfold_std:.3f}")
    cv_text = "\n".join(cv_lines)

    ax.text(
        0.57, 0.15,
        cv_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#1e2130",
            edgecolor="#4fc3f7",
            alpha=0.9,
        ),
        color="#e0e0e0",
        family="monospace",
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", color="#c0c0c0", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="#c0c0c0", fontsize=12)
    ax.set_title(
        f"ROC Curves — Buffered LOOCV\nLandslide Susceptibility Index ({study_name})",
        color="#ffffff",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333550")

    ax.legend(
        loc="lower right",
        framealpha=0.85,
        facecolor="#1e2130",
        edgecolor="#4fc3f7",
        fontsize=9.5,
        labelcolor="#e0e0e0",
    )
    ax.grid(True, linestyle="--", alpha=0.2, color="#555770")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("ROC comparison plot saved → %s", output_path)


# ─── Metrics Report ───────────────────────────────────────────────────────────


def _cm_to_str(cm: np.ndarray) -> list[str]:
    return [
        f"  [[{cm[0, 0]:>4}  {cm[0, 1]:>4}]",
        f"   [{cm[1, 0]:>4}  {cm[1, 1]:>4}]]",
    ]


def _append_model_section(
    lines: list[str],
    title: str,
    loocv_auc: float,
    kfold_mean: float,
    kfold_std: float,
    kfold_folds: list[float],
    metrics: dict,
    buffer_radius: float,
    n_folds: int,
    box_line,
    W: int,
) -> None:
    hor = "─" * W
    folds_str = "  ".join(f"{v:.3f}" for v in kfold_folds)
    cm_lines = _cm_to_str(metrics["confusion_matrix"])
    lines.extend([
        f"├{hor}┤",
        box_line(title),
        f"├{hor}┤",
        box_line(f"LOOCV AUC  (buffered, r={buffer_radius/1000:.0f}km) : {loocv_auc:.4f}"),
        box_line(f"{n_folds}-Fold CV AUC (mean ± std)     : {kfold_mean:.4f} ± {kfold_std:.4f}"),
        box_line(f"  Per-fold  : {folds_str}"),
        box_line(),
        box_line("Full-Dataset Metrics (training fit — not for citation):"),
        box_line(f"  Accuracy  : {metrics['accuracy']:.4f}"),
        box_line(f"  Precision : {metrics['precision']:.4f}"),
        box_line(f"  Recall    : {metrics['recall']:.4f}"),
        box_line(f"  F1-Score  : {metrics['f1']:.4f}"),
        box_line(f"  ROC-AUC   : {metrics['roc_auc']:.4f}"),
        box_line("  Confusion Matrix (TN  FP / FN  TP):"),
        box_line(cm_lines[0]),
        box_line(cm_lines[1]),
    ])


def write_metrics_report(
    rf_loocv_auc: float,
    rf_kfold_mean: float,
    rf_kfold_std: float,
    rf_kfold_folds: list[float],
    rf_metrics: dict,
    lr_loocv_auc: float,
    lr_kfold_mean: float,
    lr_kfold_std: float,
    lr_kfold_folds: list[float],
    lr_metrics: dict,
    buffer_radius: float,
    n_folds: int,
    n_features: int,
    output_path: str | Path,
    xgb_loocv_auc: float | None = None,
    xgb_kfold_mean: float | None = None,
    xgb_kfold_std: float | None = None,
    xgb_kfold_folds: list[float] | None = None,
    xgb_metrics: dict | None = None,
    selected_features: list[str] | None = None,
    study_name: str = "aizawl",
    extra_notes: list[str] | None = None,
) -> None:
    """Write a formatted metrics report to ``model_metrics_report.txt``."""
    W = 52
    hor = "─" * W

    def box_line(text: str = "", fill: str = " ") -> str:
        padded = f" {text}"
        return f"│{padded:<{W}}│"

    lines = [
        f"┌{'─' * W}┐",
        box_line("LSI ML PIPELINE — MODEL METRICS REPORT"),
        box_line(f"Study area       : {study_name}"),
        box_line(f"Active features   : {n_features}"),
        box_line(f"LOOCV buffer      : {buffer_radius:.0f} m"),
        box_line(f"Stratified K-Fold : {n_folds} folds"),
    ]
    if selected_features:
        lines.append(
            box_line(
                f"Feature list: {len(selected_features)} cols "
                "(see feature_selection.txt)"
            )
        )

    if extra_notes:
        for note in extra_notes:
            lines.append(box_line(note[: W - 1]))

    _append_model_section(
        lines, "RANDOM FOREST (PRIMARY MODEL)",
        rf_loocv_auc, rf_kfold_mean, rf_kfold_std, rf_kfold_folds, rf_metrics,
        buffer_radius, n_folds, box_line, W,
    )
    _append_model_section(
        lines, "LOGISTIC REGRESSION (BASELINE MODEL)",
        lr_loocv_auc, lr_kfold_mean, lr_kfold_std, lr_kfold_folds, lr_metrics,
        buffer_radius, n_folds, box_line, W,
    )

    if (
        xgb_metrics is not None
        and xgb_loocv_auc is not None
        and xgb_kfold_mean is not None
        and xgb_kfold_std is not None
        and xgb_kfold_folds is not None
    ):
        _append_model_section(
            lines, "XGBOOST (NHESS UPGRADE)",
            xgb_loocv_auc, xgb_kfold_mean, xgb_kfold_std, xgb_kfold_folds, xgb_metrics,
            buffer_radius, n_folds, box_line, W,
        )

    lines.append(f"└{'─' * W}┘")

    report_str = "\n".join(lines)
    print("\n" + report_str + "\n")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_str, encoding="utf-8")
    logger.info("Metrics report saved → %s", output_path)
