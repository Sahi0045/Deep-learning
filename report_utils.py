"""
report_utils.py - Shared evaluation, plotting, and benchmark-reporting utilities.

Used by both train.py and evaluate.py so that metrics, plots, and the unified
model-comparison report are produced consistently regardless of entry point.

Public API
----------
compute_metrics(...)        -> dict of accuracy / F1 / AUC / per-class accuracy / report string
plot_training_history(...)  -> saves accuracy/loss/top-K curves
plot_confusion_matrix(...)  -> saves a row-normalised confusion-matrix heatmap
plot_per_class_accuracy(...)-> saves a per-class accuracy bar chart
plot_roc_curves(...)        -> saves one-vs-rest ROC curves with AUC
write_text_report(...)      -> writes a human-readable metrics report to disk
write_comparison_report(...)-> writes model_comparison.csv, .md, and .png
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    """Compute accuracy, weighted/macro F1, mean one-vs-rest AUC, and per-class accuracy."""
    acc = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class: List[float] = []
    for cls_idx in range(len(class_names)):
        mask = y_true == cls_idx
        per_class.append(float(accuracy_score(y_true[mask], y_pred[mask])) if mask.sum() else 0.0)

    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    # label_binarize returns shape (N, 1) for the binary case; expand for uniform handling
    if y_true_bin.ndim == 1:
        y_true_bin = np.vstack([1 - y_true_bin, y_true_bin]).T

    aucs: List[float] = []
    for cls_idx in range(len(class_names)):
        if y_pred_probs.shape[1] <= cls_idx:
            continue
        if y_true_bin[:, cls_idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, cls_idx], y_pred_probs[:, cls_idx])
        aucs.append(float(auc(fpr, tpr)))
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    report_str = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=list(class_names), zero_division=0,
    )

    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "mean_auc": mean_auc,
        "per_class_accuracy": per_class,
        "report_str": report_str,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _history_dict(history) -> Dict[str, list]:
    """Return a plain dict of metric histories from a keras History or a merged dict."""
    if isinstance(history, dict):
        return history
    return dict(getattr(history, "history", {}))


def plot_training_history(history, out_path: str, title: str = "Training History") -> None:
    """Save accuracy, loss, and (if present) top-K accuracy curves side by side."""
    hist = _history_dict(history)
    if not hist:
        logger.warning("History is empty; skipping training-history plot.")
        return

    epochs = range(1, len(next(iter(hist.values()))) + 1)
    has_val = any(k.startswith("val_") for k in hist)

    # Discover the top-K metric name (top_k_accuracy, top3_accuracy, ...) robustly.
    topk_train = next((k for k in hist if k.startswith("top") and "val" not in k), None)
    topk_val = ("val_" + topk_train) if topk_train and ("val_" + topk_train) in hist else None

    n_panels = 3 if topk_train else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, y=1.02)

    def _curve(ax, key, label, style):
        if key in hist:
            ax.plot(epochs, hist[key], label=label, linewidth=2, linestyle=style)

    # Accuracy
    ax = axes[0]
    _curve(ax, "accuracy", "Train", "-")
    _curve(ax, "val_accuracy", "Val", "--")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    if has_val:
        ax.legend()
    ax.grid(alpha=0.3)

    # Loss
    ax = axes[1]
    _curve(ax, "loss", "Train", "-")
    _curve(ax, "val_loss", "Val", "--")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    if has_val:
        ax.legend()
    ax.grid(alpha=0.3)

    # Top-K accuracy (optional)
    if topk_train:
        ax = axes[2]
        ax.plot(epochs, hist[topk_train], label="Train", linewidth=2)
        if topk_val:
            ax.plot(epochs, hist[topk_val], label="Val", linewidth=2, linestyle="--")
        ax.set_title("Top-K Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Top-K Accuracy")
        if has_val:
            ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training history plot saved to %s", out_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
    title: str = "Normalised Confusion Matrix",
) -> None:
    """Save a row-normalised confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_n = np.nan_to_num(cm.astype("float") / cm.sum(axis=1, keepdims=True))

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n), max(6, 1.4 * n)))
    sns.heatmap(
        cm_n, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out_path)


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
    title: str = "Per-Class Accuracy",
) -> None:
    """Save a bar chart of accuracy per class."""
    per_class = []
    for cls_idx in range(len(class_names)):
        mask = y_true == cls_idx
        per_class.append(float(accuracy_score(y_true[mask], y_pred[mask])) if mask.sum() else 0.0)

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(class_names)), 5))
    colors = plt.cm.RdYlGn(np.clip(np.array(per_class), 0, 1))
    bars = ax.bar(class_names, per_class, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(
        float(np.mean(per_class)) if per_class else 0.0,
        color="navy", linewidth=1.5, linestyle="--",
        label=f"Mean: {np.mean(per_class):.2%}" if per_class else "Mean: n/a",
    )
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    for bar, acc in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Per-class accuracy chart saved to %s", out_path)


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
    title: str = "ROC Curves (One-vs-Rest)",
) -> None:
    """Save one-vs-rest ROC curves with per-class and mean AUC."""
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    if y_true_bin.ndim == 1:
        y_true_bin = np.vstack([1 - y_true_bin, y_true_bin]).T

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(class_names), 1)))
    aucs: List[float] = []
    for idx, (cls_name, color) in enumerate(zip(class_names, colors)):
        if y_pred_probs.shape[1] <= idx or y_true_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_pred_probs[:, idx])
        roc_auc = float(auc(fpr, tpr))
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{cls_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} | Mean AUC = {np.mean(aucs):.3f}" if aucs else title)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to %s", out_path)


# ---------------------------------------------------------------------------
# Text + comparison reports
# ---------------------------------------------------------------------------

def write_text_report(
    metrics: Dict[str, object],
    class_names: Sequence[str],
    out_path: str,
    header: str = "Evaluation Results",
    extra_lines: Optional[List[str]] = None,
) -> None:
    """Write a human-readable metrics report (precision/recall/F1 + summary) to disk."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{header}\n{'=' * len(header)}\n\n")
        f.write(f"Accuracy      : {metrics['accuracy']:.4f}  ({metrics['accuracy'] * 100:.2f}%)\n")
        f.write(f"Weighted F1   : {metrics['weighted_f1']:.4f}\n")
        f.write(f"Macro F1      : {metrics['macro_f1']:.4f}\n")
        f.write(f"Mean AUC      : {metrics['mean_auc']:.4f}\n\n")
        if extra_lines:
            f.write("\n".join(extra_lines) + "\n\n")
        f.write("Per-class accuracy:\n")
        for cls, acc in zip(class_names, metrics["per_class_accuracy"]):
            f.write(f"  {cls:<18} {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(str(metrics["report_str"]))
    logger.info("Text report saved to %s", out_path)


def _fmt(x, default="-") -> str:
    if x is None:
        return default
    if isinstance(x, float):
        return f"{x:.4f}" if not np.isnan(x) else default
    return str(x)


def write_comparison_report(
    rows: List[Dict[str, object]],
    out_dir: str,
    class_names: Sequence[str],
    per_class_counts: Optional[Dict[str, Dict[str, int]]] = None,
    dataset_name: str = "",
) -> Dict[str, str]:
    """
    Write the unified model-comparison artefacts.

    Produces:
      - model_comparison.csv  (machine-readable results table)
      - model_comparison.md   (human-readable report incl. dataset + per-class counts)
      - model_comparison.png  (grouped bar chart of the four headline metrics)

    Each ``row`` should contain at least: model_type, name, params, train_time_s,
    best_val_accuracy, accuracy, weighted_f1, macro_f1, mean_auc, status.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "model_comparison.csv")
    md_path = os.path.join(out_dir, "model_comparison.md")
    png_path = os.path.join(out_dir, "model_comparison.png")

    fields = [
        "model_type", "name", "params", "train_time_s", "best_val_accuracy",
        "accuracy", "weighted_f1", "macro_f1", "mean_auc", "status",
    ]

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    logger.info("Comparison CSV saved to %s", csv_path)

    # MD
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Report\n\n")
        if dataset_name:
            f.write(f"**Dataset:** {dataset_name}\n\n")
        if per_class_counts:
            f.write("## Dataset summary (per-class sample counts)\n\n")
            f.write("| Class | Train | Val | Test | Total |\n")
            f.write("|-------|------|-----|------|-------|\n")
            for cls in class_names:
                tr = per_class_counts.get("train", {}).get(cls, 0)
                va = per_class_counts.get("val", {}).get(cls, 0)
                te = per_class_counts.get("test", {}).get(cls, 0)
                f.write(f"| {cls} | {tr} | {va} | {te} | {tr + va + te} |\n")
            f.write("\n")
        f.write("## Results\n\n")
        f.write("| Model | Type | Params | Train (s) | Best Val Acc | Test Acc | Weighted F1 | Macro F1 | Mean AUC | Status |\n")
        f.write("|-------|------|--------|-----------|--------------|----------|-------------|----------|----------|--------|\n")
        for r in rows:
            f.write(
                f"| {r.get('name', '')} | {r.get('model_type', '')} | "
                f"{r.get('params', '')} | {_fmt(r.get('train_time_s'), '0')} | "
                f"{_fmt(r.get('best_val_accuracy'))} | {_fmt(r.get('accuracy'))} | "
                f"{_fmt(r.get('weighted_f1'))} | {_fmt(r.get('macro_f1'))} | "
                f"{_fmt(r.get('mean_auc'))} | {r.get('status', '')} |\n"
            )
        # winner
        ok = [r for r in rows if r.get("status") == "ok"]
        if ok:
            best = max(ok, key=lambda r: float(r.get("accuracy", 0)))
            f.write(f"\n**Best test accuracy:** {best.get('name')} "
                    f"({float(best.get('accuracy', 0)) * 100:.2f}%)\n")
            best_f1 = max(ok, key=lambda r: float(r.get("weighted_f1", 0)))
            f.write(f"**Best weighted F1:** {best_f1.get('name')} "
                    f"({float(best_f1.get('weighted_f1', 0)):.4f})\n")
    logger.info("Comparison Markdown saved to %s", md_path)

    # PNG - grouped bar chart of the four headline metrics across models
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        labels = [str(r.get("name", r.get("model_type", ""))) for r in ok_rows]
        metrics_names = ["accuracy", "weighted_f1", "macro_f1", "mean_auc"]
        metric_labels = ["Accuracy", "Weighted F1", "Macro F1", "Mean AUC"]
        x = np.arange(len(labels))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(8, 2 * len(labels)), 6))
        for i, (mkey, mlabel) in enumerate(zip(metrics_names, metric_labels)):
            vals = []
            for r in ok_rows:
                v = r.get(mkey)
                vals.append(float(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else 0.0)
            bars = ax.bar(x + (i - 1.5) * width, vals, width, label=mlabel)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
        ax.set_title("Model Comparison"); ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info("Comparison chart saved to %s", png_path)
    else:
        logger.warning("No successful model rows; skipping comparison chart.")

    return {"csv": csv_path, "md": md_path, "png": png_path}
