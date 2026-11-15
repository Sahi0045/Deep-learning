"""
train.py - Multi-model training and benchmarking pipeline.

Supports four model families and a unified ``--model_type all`` benchmark:

  custom_cnn       4-block 2D Residual CNN baseline (single frames)
  efficientnet_v2  EfficientNetV2B0 transfer learning (single frames)
  cnn_lstm         Spatio-temporal CNN + BiLSTM (video clips)
  conv3d           Volumetric 3D CNN (video clips)
  all              Train and compare every model family above

Key features
------------
- ``--model_type all`` trains each family and writes a unified comparison report
  (results/model_comparison.{csv,md,png}).
- EfficientNetV2 transfer learning uses fast, feature-cached head training by
  default: the frozen backbone's pooled features are computed once, the head is
  trained on the cache, and the model is reassembled into a self-contained,
  loadable ``.keras`` file. End-to-end fine-tuning is optional (``--fine_tune``).
- Per-class sample counts are logged and recorded in the comparison report.
- Per-model artefacts: checkpoints/best_<mtype>.keras, training-history plot,
  confusion matrix, ROC curves, and a text classification report.

Usage
-----
  python3 train.py --model_type custom_cnn --epochs 25
  python3 train.py --model_type efficientnet_v2 --fine_tune
  python3 train.py --model_type all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import tensorflow as tf
from tensorflow import keras

from config import cfg
from data_pipeline import build_tf_dataset, get_datasets, get_video_datasets
from model import (
    VIDEO_TYPES,
    build_classification_head,
    build_feature_extractor,
    build_model_by_name,
    compile_model,
    reassemble_transfer_model,
    unfreeze_backbone,
)
from report_utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    write_comparison_report,
    write_text_report,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.data.logs_dir, "training.log")),
    ],
)
logger = logging.getLogger(__name__)

ALL_TYPES = ["custom_cnn", "efficientnet_v2", "cnn_lstm", "conv3d"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train / benchmark violence-detection models")
    parser.add_argument(
        "--model_type", default="custom_cnn",
        choices=["custom_cnn", "efficientnet_v2", "transfer", "cnn_lstm", "conv3d", "3d_cnn", "all"],
        help="Which architecture(s) to train. 'all' benchmarks every family.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="2D training epochs (default: cfg).")
    parser.add_argument("--batch_size", type=int, default=None, help="2D batch size (default: cfg).")
    parser.add_argument("--video_epochs", type=int, default=None, help="Video training epochs.")
    parser.add_argument("--video_batch_size", type=int, default=None, help="Video batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--optimizer", default=None, choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation.")
    parser.add_argument(
        "--cache_features", dest="cache_features", action="store_true", default=True,
        help="Cache frozen-backbone features for fast transfer head training (default: on).",
    )
    parser.add_argument(
        "--no_cache_features", dest="cache_features", action="store_false",
        help="Train EfficientNetV2 end-to-end on the frozen backbone (slower).",
    )
    parser.add_argument("--fine_tune", action="store_true",
                        help="After cached head training, fine-tune the EfficientNetV2 backbone end-to-end.")
    parser.add_argument("--fine_tune_epochs", type=int, default=15, help="Fine-tuning epochs.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_video(mtype: str) -> bool:
    return mtype in VIDEO_TYPES


def resolve_types(model_type: str) -> List[str]:
    if model_type == "all":
        return list(ALL_TYPES)
    if model_type == "transfer":
        return ["efficientnet_v2"]
    if model_type == "3d_cnn":
        return ["conv3d"]
    return [model_type]


def _epochs_for(mtype: str, args: argparse.Namespace) -> int:
    if is_video(mtype):
        return args.video_epochs if args.video_epochs is not None else cfg.training.video_epochs
    return args.epochs if args.epochs is not None else cfg.training.epochs


def _batch_for(mtype: str, args: argparse.Namespace) -> int:
    if is_video(mtype):
        return args.video_batch_size if args.video_batch_size is not None else cfg.training.video_batch_size
    return args.batch_size if args.batch_size is not None else cfg.training.batch_size


def _lr(args: argparse.Namespace) -> float:
    return args.lr if args.lr is not None else cfg.training.learning_rate


def _optimizer(args: argparse.Namespace) -> str:
    return args.optimizer if args.optimizer is not None else cfg.training.optimizer


def load_datasets(mtype: str, args: argparse.Namespace) -> dict:
    augment = not args.no_augment
    if is_video(mtype):
        return get_video_datasets(augment=augment)
    return get_datasets(augment=augment)


def build_callbacks(model_path: str, log_subdir: str, csv_name: str) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor=cfg.training.checkpoint_metric,
            save_best_only=cfg.training.save_best_only,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.training.lr_decay_factor,
            patience=cfg.training.lr_patience,
            min_lr=cfg.training.min_lr,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(cfg.data.logs_dir, log_subdir),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
        keras.callbacks.CSVLogger(
            os.path.join(cfg.data.logs_dir, csv_name), append=False,
        ),
    ]


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Balanced class weights from one-hot training labels."""
    from sklearn.utils.class_weight import compute_class_weight
    y_int = np.argmax(y_train, axis=1)
    classes = np.unique(y_int)
    weights = compute_class_weight("balanced", classes=classes, y=y_int)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _log_per_class_counts(datasets: dict, mtype: str) -> None:
    counts = datasets.get("per_class_counts", {})
    if not counts:
        return
    logger.info("[%s] Per-class sample counts:", mtype)
    for cls, n in counts.get("train", {}).items():
        logger.info("  %-18s train=%4d  val=%4d  test=%4d  total=%4d",
                    cls, n,
                    counts.get("val", {}).get(cls, 0),
                    counts.get("test", {}).get(cls, 0),
                    counts.get("total", {}).get(cls, 0))


def _merge_histories(h1: dict, h2: dict) -> dict:
    merged: dict = {}
    for k, v in h1.items():
        merged[k] = list(v) + list(h2.get(k, []))
    for k, v in h2.items():
        if k not in merged:
            merged[k] = list(v)
    return merged


# ---------------------------------------------------------------------------
# Training paths
# ---------------------------------------------------------------------------

def train_transfer_cached(
    datasets: dict,
    args: argparse.Namespace,
    mtype: str,
) -> Tuple[keras.Model, dict, float]:
    """Fast EfficientNetV2 transfer training: cache frozen-backbone features, train head, reassemble."""
    epochs = _epochs_for(mtype, args)
    batch_size = _batch_for(mtype, args)
    lr = _lr(args)
    opt = _optimizer(args)
    image_size = cfg.data.image_size
    c = cfg.data.num_channels
    num_classes = datasets["num_classes"]
    class_names = datasets["class_names"]

    X_train, y_train = datasets["X_train"], datasets["y_train"]
    X_val, y_val = datasets["X_val"], datasets["y_val"]

    logger.info("[%s] Caching frozen-backbone features (train=%d, val=%d)...",
                mtype, len(X_train), len(X_val))
    extractor, feat_dim = build_feature_extractor(image_size, c)
    logger.info("[%s] Feature dimension: %d", mtype, feat_dim)

    Ftr = extractor.predict(X_train, batch_size=batch_size, verbose=0)
    Fval = extractor.predict(X_val, batch_size=batch_size, verbose=0)

    head = build_classification_head(
        feat_dim, num_classes,
        fc_dropout=cfg.model.fc_dropout, l2_lambda=cfg.model.l2_lambda,
    )
    head = compile_model(head, learning_rate=lr, optimizer=opt)

    cw = compute_class_weights(y_train)
    logger.info("[%s] Class weights: %s", mtype,
                {class_names[k]: f"{v:.2f}" for k, v in cw.items()})

    head_ckpt = os.path.join(cfg.data.model_dir, f"_head_{mtype}.keras")
    callbacks = build_callbacks(head_ckpt, log_subdir=mtype, csv_name=f"training_log_{mtype}_head.csv")
    # Cache phase uses no augmentation (features are precomputed).
    logger.info("[%s] Training head on cached features (epochs=%d)...", mtype, epochs)
    t0 = time.time()
    history = head.fit(
        Ftr, y_train, validation_data=(Fval, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks,
        class_weight=cw, verbose=1,
    )
    cached_time = time.time() - t0

    # Reassemble into a self-contained end-to-end model from the best head weights.
    w, b = head.get_layer("predictions").get_weights()
    model = reassemble_transfer_model(
        image_size, c, num_classes, (w, b),
        fc_dropout=cfg.model.fc_dropout, l2_lambda=cfg.model.l2_lambda,
    )
    model = compile_model(model, learning_rate=lr, optimizer=opt)

    history_dict = dict(history.history)

    # Optional end-to-end fine-tuning (with augmentation) on the reassembled model.
    if args.fine_tune:
        logger.info("[%s] Fine-tuning backbone end-to-end (epochs=%d)...", mtype, args.fine_tune_epochs)
        unfreeze_backbone(model)
        model = compile_model(model, learning_rate=lr * 0.1, optimizer=opt)
        augment = not args.no_augment
        train_ds = build_tf_dataset(X_train, y_train, batch_size, augment=augment, shuffle=True, seed=cfg.data.random_seed)
        val_ds = build_tf_dataset(X_val, y_val, batch_size, augment=False, shuffle=False)
        ft_path = os.path.join(cfg.data.model_dir, f"best_{mtype}.keras")
        ft_callbacks = build_callbacks(ft_path, log_subdir=f"{mtype}_ft", csv_name=f"training_log_{mtype}_ft.csv")
        t1 = time.time()
        history_ft = model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.fine_tune_epochs, callbacks=ft_callbacks,
            class_weight=cw, verbose=1,
        )
        cached_time += time.time() - t1
        history_dict = _merge_histories(history_dict, dict(history_ft.history))

    # Cleanup the temporary head checkpoint.
    if os.path.exists(head_ckpt):
        os.remove(head_ckpt)

    return model, history_dict, cached_time


def train_standard(
    mtype: str,
    datasets: dict,
    args: argparse.Namespace,
) -> Tuple[keras.Model, dict, float]:
    """End-to-end training for custom_cnn, cnn_lstm, or conv3d (or EfficientNetV2 if caching is off)."""
    epochs = _epochs_for(mtype, args)
    batch_size = _batch_for(mtype, args)
    lr = _lr(args)
    opt = _optimizer(args)

    model = build_model_by_name(mtype)
    model = compile_model(model, learning_rate=lr, optimizer=opt)
    model.summary(print_fn=logger.info)
    logger.info("[%s] Total parameters: %s", mtype, f"{model.count_params():,}")

    cw = compute_class_weights(datasets["y_train"])
    logger.info("[%s] Class weights: %s", mtype,
                {datasets["class_names"][k]: f"{v:.2f}" for k, v in cw.items()})

    ckpt_path = os.path.join(cfg.data.model_dir, f"best_{mtype}.keras")
    callbacks = build_callbacks(ckpt_path, log_subdir=mtype, csv_name=f"training_log_{mtype}.csv")

    logger.info("[%s] Training (epochs=%d, batch=%d, lr=%.6f, optimizer=%s)...",
                mtype, epochs, batch_size, lr, opt)
    t0 = time.time()
    history = model.fit(
        datasets["train"], validation_data=datasets["val"],
        epochs=epochs, callbacks=callbacks, class_weight=cw, verbose=1,
    )
    train_time = time.time() - t0
    return model, dict(history.history), train_time


# ---------------------------------------------------------------------------
# Per-model evaluation + artefacts
# ---------------------------------------------------------------------------

def evaluate_and_record(
    mtype: str,
    model: keras.Model,
    datasets: dict,
    history: dict,
    train_time: float,
) -> dict:
    """Evaluate on the test split, save per-model artefacts, return a comparison row."""
    class_names = datasets["class_names"]
    X_test, y_test = datasets["X_test"], datasets["y_test"]

    logger.info("[%s] Predicting on test set (%d samples)...", mtype, len(X_test))
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    metrics = compute_metrics(y_true, y_pred, y_pred_probs, class_names)
    logger.info("[%s] Test accuracy=%.4f  weighted_f1=%.4f  macro_f1=%.4f  mean_auc=%.4f",
                mtype, metrics["accuracy"], metrics["weighted_f1"],
                metrics["macro_f1"], metrics["mean_auc"])

    # Per-model artefacts
    tag = mtype
    plot_training_history(
        history,
        os.path.join(cfg.data.results_dir, f"training_history_{tag}.png"),
        title=f"Training History — {tag}",
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        os.path.join(cfg.data.results_dir, f"confusion_matrix_{tag}.png"),
        title=f"Normalised Confusion Matrix — {tag}",
    )
    plot_roc_curves(
        y_true, y_pred_probs, class_names,
        os.path.join(cfg.data.results_dir, f"roc_curves_{tag}.png"),
    )
    write_text_report(
        metrics, class_names,
        os.path.join(cfg.data.results_dir, f"classification_report_{tag}.txt"),
        header=f"Evaluation — {tag}",
        extra_lines=[f"Model type       : {mtype}",
                     f"Parameters       : {model.count_params():,}",
                     f"Train time (s)   : {train_time:.1f}"],
    )

    best_val_acc = max(history.get("val_accuracy", [0.0])) if history else 0.0
    return {
        "model_type": mtype,
        "name": model.name,
        "params": model.count_params(),
        "train_time_s": round(train_time, 1),
        "best_val_accuracy": float(best_val_acc),
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_f1": metrics["macro_f1"],
        "mean_auc": metrics["mean_auc"],
        "status": "ok",
    }


def save_model(mtype: str, model: keras.Model) -> None:
    """Persist the self-contained full model (best + final)."""
    best_path = os.path.join(cfg.data.model_dir, f"best_{mtype}.keras")
    final_path = os.path.join(cfg.data.model_dir, f"final_{mtype}.keras")
    # For the cached path the full model was never checkpointed; save it now.
    model.save(best_path)
    model.save(final_path)
    logger.info("[%s] Saved best -> %s", mtype, best_path)
    logger.info("[%s] Saved final -> %s", mtype, final_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_one(mtype: str, args: argparse.Namespace) -> Optional[dict]:
    """Train a single model family end-to-end and return its comparison row (or None on failure)."""
    logger.info("\n" + "=" * 70)
    logger.info("Training model family: %s", mtype)
    logger.info("=" * 70)
    try:
        datasets = load_datasets(mtype, args)
    except FileNotFoundError as e:
        logger.error("[%s] Data unavailable: %s", mtype, e)
        return None

    logger.info("[%s] Train=%d  Val=%d  Test=%d  (classes=%s)",
                mtype, datasets["train_size"], datasets["val_size"],
                datasets["test_size"], datasets["class_names"])
    logger.info("[%s] Leakage check: %s", mtype, datasets.get("leakage_check"))
    _log_per_class_counts(datasets, mtype)

    try:
        if mtype == "efficientnet_v2" and args.cache_features:
            model, history, train_time = train_transfer_cached(datasets, args, mtype)
        else:
            model, history, train_time = train_standard(mtype, datasets, args)
        save_model(mtype, model)
        row = evaluate_and_record(mtype, model, datasets, history, train_time)
        row["per_class_counts"] = datasets.get("per_class_counts")
        row["class_names"] = datasets["class_names"]
        return row
    except Exception as e:  # keep the benchmark running if one model fails
        logger.exception("[%s] Training failed: %s", mtype, e)
        return {
            "model_type": mtype, "name": mtype, "params": "",
            "train_time_s": 0, "best_val_accuracy": "",
            "accuracy": "", "weighted_f1": "", "macro_f1": "", "mean_auc": "",
            "status": f"failed: {type(e).__name__}",
        }


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Violence-Detection Training Pipeline")
    logger.info("=" * 70)
    logger.info("TensorFlow %s | Keras %s", tf.__version__, keras.__version__)
    logger.info("GPUs available: %d", len(tf.config.list_physical_devices("GPU")))

    types = resolve_types(args.model_type)
    logger.info("Model families to train: %s", types)

    rows: List[dict] = []
    first_counts: Optional[dict] = None
    first_classes: Optional[list] = None
    for mtype in types:
        row = train_one(mtype, args)
        if row is None:
            continue
        rows.append(row)
        if first_counts is None and row.get("per_class_counts"):
            first_counts = row["per_class_counts"]
            first_classes = row.get("class_names")

    if not rows:
        logger.error("No models were trained. Exiting.")
        sys.exit(1)

    # Unified comparison report (use the per-class counts from the first successful run).
    class_names = first_classes or cfg.data.classes
    write_comparison_report(
        rows,
        out_dir=cfg.data.results_dir,
        class_names=class_names,
        per_class_counts=first_counts,
        dataset_name=cfg.data.dataset_name,
    )

    logger.info("\n" + "=" * 70)
    logger.info("Benchmark complete. %d/%d model families trained.", sum(1 for r in rows if r["status"] == "ok"), len(rows))
    logger.info("Comparison report: %s/model_comparison.{{csv,md,png}}", cfg.data.results_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
