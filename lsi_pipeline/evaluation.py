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
    """Compute classification metrics on the full (scaled) dataset.

    Intended as a supplementary diagnostic alongside the CV scores.
    Because the model is evaluated on its own training data (when no
    held-out split exists), these numbers reflect training fit, NOT
    generalisation. Cite CV AUC as the primary result.

    Parameters
    ----------
    model:
        A fitted scikit-learn estimator with ``predict`` and
        ``predict_proba`` methods.
    X_scaled:
        Scaled feature matrix (n_samples, n_features).
    y_true:
        True binary labels (n_samples,).
    model_name:
        Label for logging output.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix.
    """
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
) -> None:
    """Plot overlaid LOOCV ROC curves for RF and LR with 5-fold CV annotation.

    The primary curves are derived from aggregated LOOCV predictions.
    A text box displays the 5-fold stratified CV AUC mean ± std for each
    model as a stability reference.

    Parameters
    ----------
    rf_loocv_true:
        Aggregated true labels from RF buffered LOOCV.
    rf_loocv_prob:
        Aggregated predicted probabilities from RF buffered LOOCV.
    rf_loocv_auc:
        RF buffered LOOCV AUC scalar.
    rf_kfold_mean:
        RF 5-fold CV mean AUC.
    rf_kfold_std:
        RF 5-fold CV AUC std deviation.
    lr_loocv_true:
        Aggregated true labels from LR buffered LOOCV.
    lr_loocv_prob:
        Aggregated predicted probabilities from LR buffered LOOCV.
    lr_loocv_auc:
        LR buffered LOOCV AUC scalar.
    lr_kfold_mean:
        LR 5-fold CV mean AUC.
    lr_kfold_std:
        LR 5-fold CV AUC std deviation.
    output_path:
        Destination path for ``roc_comparison.png`` (300 DPI).
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # ── Plot RF ROC ──────────────────────────────────────────────────────────
    fpr_rf, tpr_rf, _ = roc_curve(rf_loocv_true, rf_loocv_prob)
    ax.plot(
        fpr_rf, tpr_rf,
        color="#4fc3f7",
        lw=2.5,
        label=f"Random Forest — LOOCV AUC = {rf_loocv_auc:.3f}",
    )

    # ── Plot LR ROC ──────────────────────────────────────────────────────────
    fpr_lr, tpr_lr, _ = roc_curve(lr_loocv_true, lr_loocv_prob)
    ax.plot(
        fpr_lr, tpr_lr,
        color="#ef9a9a",
        lw=2.5,
        linestyle="--",
        label=f"Logistic Regression — LOOCV AUC = {lr_loocv_auc:.3f}",
    )

    # ── Diagonal reference ───────────────────────────────────────────────────
    ax.plot(
        [0, 1], [0, 1],
        color="#555770",
        lw=1.2,
        linestyle=":",
        label="Random Classifier (AUC = 0.500)",
    )

    # ── CV annotation text box ───────────────────────────────────────────────
    cv_text = (
        "5-Fold Stratified CV (stability)\n"
        f"  RF : {rf_kfold_mean:.3f} ± {rf_kfold_std:.3f}\n"
        f"  LR : {lr_kfold_mean:.3f} ± {lr_kfold_std:.3f}"
    )
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

    # ── Styling ──────────────────────────────────────────────────────────────
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", color="#c0c0c0", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="#c0c0c0", fontsize=12)
    ax.set_title(
        "ROC Curves — Buffered LOOCV\nLandslide Susceptibility Index (Aizawl, Mizoram)",
        color="#ffffff",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333550")

    legend = ax.legend(
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
) -> None:
    """Write a formatted metrics report to ``model_metrics_report.txt``.

    Uses box-drawing characters for alignment consistent with the
    project specification.

    Parameters
    ----------
    rf_loocv_auc:
        Random Forest buffered LOOCV AUC.
    rf_kfold_mean:
        RF 5-fold CV mean AUC.
    rf_kfold_std:
        RF 5-fold CV AUC std deviation.
    rf_kfold_folds:
        Per-fold AUC list for RF.
    rf_metrics:
        Dict from :func:`compute_full_dataset_metrics` for RF.
    lr_loocv_auc:
        Logistic Regression buffered LOOCV AUC.
    lr_kfold_mean:
        LR 5-fold CV mean AUC.
    lr_kfold_std:
        LR 5-fold CV AUC std deviation.
    lr_kfold_folds:
        Per-fold AUC list for LR.
    lr_metrics:
        Dict from :func:`compute_full_dataset_metrics` for LR.
    buffer_radius:
        Spatial exclusion radius used in LOOCV (metres).
    n_folds:
        Number of stratified CV folds.
    n_features:
        Number of active feature columns.
    output_path:
        Destination path for the report text file.
    """
    W = 52  # box width (inner)
    hor = "─" * W

    def box_line(text: str = "", fill: str = " ") -> str:
        padded = f" {text}"
        return f"│{padded:<{W}}│"

    def section_header(text: str) -> str:
        return f"├{hor}┤\n" + box_line(text)

    def cm_to_str(cm: np.ndarray) -> list[str]:
        return [
            f"  [[{cm[0,0]:>4}  {cm[0,1]:>4}]",
            f"   [{cm[1,0]:>4}  {cm[1,1]:>4}]]",
        ]

    rf_cm_lines = cm_to_str(rf_metrics["confusion_matrix"])
    lr_cm_lines = cm_to_str(lr_metrics["confusion_matrix"])

    rf_folds_str = "  ".join(f"{v:.3f}" for v in rf_kfold_folds)
    lr_folds_str = "  ".join(f"{v:.3f}" for v in lr_kfold_folds)

    lines = [
        f"┌{'─' * W}┐",
        box_line("LSI ML PIPELINE — MODEL METRICS REPORT"),
        box_line(f"Active features   : {n_features}"),
        box_line(f"LOOCV buffer      : {buffer_radius:.0f} m"),
        box_line(f"Stratified K-Fold : {n_folds} folds"),
        f"├{'─' * W}┤",
        box_line("RANDOM FOREST (PRIMARY MODEL)"),
        f"├{'─' * W}┤",
        box_line(f"LOOCV AUC  (buffered, r={buffer_radius/1000:.0f}km) : {rf_loocv_auc:.4f}"),
        box_line(f"{n_folds}-Fold CV AUC (mean ± std)     : {rf_kfold_mean:.4f} ± {rf_kfold_std:.4f}"),
        box_line(f"  Per-fold  : {rf_folds_str}"),
        box_line(),
        box_line("Full-Dataset Metrics (training fit — not for citation):"),
        box_line(f"  Accuracy  : {rf_metrics['accuracy']:.4f}"),
        box_line(f"  Precision : {rf_metrics['precision']:.4f}"),
        box_line(f"  Recall    : {rf_metrics['recall']:.4f}"),
        box_line(f"  F1-Score  : {rf_metrics['f1']:.4f}"),
        box_line(f"  ROC-AUC   : {rf_metrics['roc_auc']:.4f}"),
        box_line("  Confusion Matrix (TN  FP / FN  TP):"),
        box_line(rf_cm_lines[0]),
        box_line(rf_cm_lines[1]),
        f"├{'─' * W}┤",
        box_line("LOGISTIC REGRESSION (BASELINE MODEL)"),
        f"├{'─' * W}┤",
        box_line(f"LOOCV AUC  (buffered, r={buffer_radius/1000:.0f}km) : {lr_loocv_auc:.4f}"),
        box_line(f"{n_folds}-Fold CV AUC (mean ± std)     : {lr_kfold_mean:.4f} ± {lr_kfold_std:.4f}"),
        box_line(f"  Per-fold  : {lr_folds_str}"),
        box_line(),
        box_line("Full-Dataset Metrics (training fit — not for citation):"),
        box_line(f"  Accuracy  : {lr_metrics['accuracy']:.4f}"),
        box_line(f"  Precision : {lr_metrics['precision']:.4f}"),
        box_line(f"  Recall    : {lr_metrics['recall']:.4f}"),
        box_line(f"  F1-Score  : {lr_metrics['f1']:.4f}"),
        box_line(f"  ROC-AUC   : {lr_metrics['roc_auc']:.4f}"),
        box_line("  Confusion Matrix (TN  FP / FN  TP):"),
        box_line(lr_cm_lines[0]),
        box_line(lr_cm_lines[1]),
        f"└{'─' * W}┘",
    ]

    report_str = "\n".join(lines)
    print("\n" + report_str + "\n")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_str, encoding="utf-8")
    logger.info("Metrics report saved → %s", output_path)
