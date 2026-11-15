"""
evaluate.py - Standalone evaluation and multi-model comparison.

Evaluates one or more trained models on the held-out split and produces, per
model, a classification report, normalised confusion matrix, per-class accuracy
chart, and ROC curves. When more than one model is evaluated (``--model_type
all`` or by default, all checkpoints found in checkpoints/), it also writes a
unified comparison report:

    results/model_comparison.csv
    results/model_comparison.md
    results/model_comparison.png

The script is model-type aware: 2D models (custom_cnn, efficientnet_v2) are
scored on extracted frames; video models (cnn_lstm, conv3d) are scored on
spatio-temporal clips.

Usage
-----
  python3 evaluate.py                          # evaluate all checkpoints present
  python3 evaluate.py --model_type custom_cnn
  python3 evaluate.py --model_type all --split test
  python3 evaluate.py --model checkpoints/best_custom_cnn.keras --model_type custom_cnn
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import tensorflow as tf
from tensorflow import keras

from config import cfg
from data_pipeline import get_datasets, get_video_datasets
from model import VIDEO_TYPES  # noqa: F401  (re-exported for downstream tools)
from report_utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_roc_curves,
    write_comparison_report,
    write_text_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KNOWN_TYPES = ["custom_cnn", "efficientnet_v2", "cnn_lstm", "conv3d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate / compare violence-detection models")
    parser.add_argument(
        "--model_type", default="all",
        choices=["all", "custom_cnn", "efficientnet_v2", "cnn_lstm", "conv3d"],
        help="Which model family(ies) to evaluate. 'all' evaluates every checkpoint found.",
    )
    parser.add_argument("--model", default=None,
                        help="Path to a specific .keras checkpoint (overrides best_<mtype>.keras).")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which held-out split to evaluate on (default: test).")
    parser.add_argument("--no_augment", action="store_true",
                        help="No effect on evaluation; kept for parity with train.py.")
    return parser.parse_args()


def is_video(mtype: str) -> bool:
    return mtype in VIDEO_TYPES


def checkpoint_for(mtype: str) -> str:
    return os.path.join(cfg.data.model_dir, f"best_{mtype}.keras")


def load_datasets_for(mtype: str) -> dict:
    """Load the dataset matching a model family (frames for 2D, clips for video)."""
    if is_video(mtype):
        return get_video_datasets(augment=False)
    return get_datasets(augment=False)


def select_split(datasets: dict, split: str) -> tuple:
    """Return (X, y) arrays for the requested split."""
    if split == "val":
        return datasets["X_val"], datasets["y_val"]
    return datasets["X_test"], datasets["y_test"]


def evaluate_one(
    mtype: str,
    model_path: str,
    split: str,
) -> Optional[dict]:
    """Evaluate a single checkpoint; return a comparison row or None on failure."""
    if not os.path.exists(model_path):
        logger.warning("[%s] checkpoint not found: %s — skipping.", mtype, model_path)
        return None

    logger.info("\n" + "=" * 60)
    logger.info("[%s] Loading model: %s", mtype, model_path)
    model = keras.models.load_model(model_path, compile=False)
    model.summary(print_fn=logger.info)

    try:
        datasets = load_datasets_for(mtype)
    except FileNotFoundError as e:
        logger.error("[%s] Data unavailable: %s", mtype, e)
        return None

    class_names = datasets["class_names"]
    X, y = select_split(datasets, split)
    logger.info("[%s] Predicting on '%s' split (%d samples)...", mtype, split, len(X))

    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y, axis=1)

    metrics = compute_metrics(y_true, y_pred, y_pred_probs, class_names)
    logger.info("[%s] Accuracy=%.4f  Weighted F1=%.4f  Macro F1=%.4f  Mean AUC=%.4f",
                mtype, metrics["accuracy"], metrics["weighted_f1"],
                metrics["macro_f1"], metrics["mean_auc"])

    # Per-model artefacts (suffixed by split to avoid clobbering cross-split runs)
    tag = f"{mtype}_{split}"
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        os.path.join(cfg.data.results_dir, f"confusion_matrix_{tag}.png"),
        title=f"Normalised Confusion Matrix — {mtype} ({split})",
    )
    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        os.path.join(cfg.data.results_dir, f"per_class_accuracy_{tag}.png"),
        title=f"Per-Class Accuracy — {mtype} ({split})",
    )
    plot_roc_curves(
        y_true, y_pred_probs, class_names,
        os.path.join(cfg.data.results_dir, f"roc_curves_{tag}.png"),
        title=f"ROC Curves — {mtype} ({split})",
    )
    write_text_report(
        metrics, class_names,
        os.path.join(cfg.data.results_dir, f"classification_report_{tag}.txt"),
        header=f"Evaluation — {mtype} ({split} split)",
        extra_lines=[f"Model type   : {mtype}",
                     f"Checkpoint   : {model_path}",
                     f"Parameters   : {model.count_params():,}",
                     f"Split        : {split}"],
    )

    return {
        "model_type": mtype,
        "name": model.name,
        "params": model.count_params(),
        "train_time_s": "",
        "best_val_accuracy": "",
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_f1": metrics["macro_f1"],
        "mean_auc": metrics["mean_auc"],
        "status": "ok",
        "per_class_counts": datasets.get("per_class_counts"),
        "class_names": class_names,
    }


def main() -> None:
    args = parse_args()

    logger.info("TensorFlow %s | Keras %s", tf.__version__, keras.__version__)

    types: List[str]
    explicit_model = args.model
    if args.model_type == "all":
        types = [t for t in KNOWN_TYPES if os.path.exists(checkpoint_for(t))]
        if not types and explicit_model:
            logger.error("No checkpoints found in %s and no --model given.", cfg.data.model_dir)
            sys.exit(1)
        if not types:
            logger.error("No checkpoints found in %s. Run train.py first.", cfg.data.model_dir)
            sys.exit(1)
    else:
        types = [args.model_type]

    rows: List[dict] = []
    first_counts: Optional[dict] = None
    first_classes: Optional[list] = None

    for mtype in types:
        model_path = explicit_model if explicit_model else checkpoint_for(mtype)
        row = evaluate_one(mtype, model_path, args.split)
        if row is None:
            continue
        rows.append(row)
        if first_counts is None and row.get("per_class_counts"):
            first_counts = row["per_class_counts"]
            first_classes = row.get("class_names")

    if not rows:
        logger.error("No models were evaluated. Exiting.")
        sys.exit(1)

    class_names = first_classes or cfg.data.classes
    write_comparison_report(
        rows,
        out_dir=cfg.data.results_dir,
        class_names=class_names,
        per_class_counts=first_counts,
        dataset_name=f"{cfg.data.dataset_name} ({args.split} split)",
    )

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete: %d model(s) evaluated on '%s' split.", len(rows), args.split)
    logger.info("Comparison report: %s/model_comparison.{{csv,md,png}}", cfg.data.results_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
