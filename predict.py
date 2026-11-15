"""
predict.py - Inference for the violence/action-detection models.

Model-type aware:
  - 2D models (custom_cnn, efficientnet_v2) classify a single image or every
    image in a directory.
  - Video models (cnn_lstm, conv3d) classify a video clip: evenly-spaced frames
    are extracted into a (T, H, W, 3) volume and classified once.

By default the best checkpoint for the requested family is loaded from
checkpoints/best_<model_type>.keras; pass --model to point elsewhere.

Usage
-----
  # Single image (2D model)
  python3 predict.py --model_type custom_cnn --image path/to/img.jpg --top_k 3

  # Whole directory (2D model)
  python3 predict.py --model_type efficientnet_v2 --image path/to/folder/

  # Video clip (video model)
  python3 predict.py --model_type cnn_lstm --video path/to/clip.mp4 --top_k 3

  # Overlay chart alongside each prediction
  python3 predict.py --model_type custom_cnn --image img.jpg --save_overlay
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import glob
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow import keras

from config import cfg
from model import VIDEO_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Violence/action-detection inference")
    parser.add_argument("--model_type", default="custom_cnn",
                        choices=["custom_cnn", "efficientnet_v2", "cnn_lstm", "conv3d"],
                        help="Model family whose checkpoint to load.")
    parser.add_argument("--model", default=None,
                        help="Path to a .keras checkpoint (default: checkpoints/best_<model_type>.keras).")
    # Exactly one of --image / --video must be given; --image may be a file or dir.
    parser.add_argument("--image", default=None, help="An image file or a directory of images.")
    parser.add_argument("--video", default=None, help="A video file to classify (video models).")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to show.")
    parser.add_argument("--save_overlay", action="store_true",
                        help="Save a prediction-overlay PNG alongside each input.")
    return parser.parse_args()


def is_video(mtype: str) -> bool:
    return mtype in VIDEO_TYPES


def default_checkpoint(mtype: str) -> str:
    return os.path.join(cfg.data.model_dir, f"best_{mtype}.keras")


def load_model(model_type: str, model_path: str) -> keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run "
            f"`train.py --model_type {model_type}` first."
        )
    logger.info("Loading %s model from: %s", model_type, model_path)
    return keras.models.load_model(model_path, compile=False)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, image_size: int) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
    return (img.astype(np.float32) / 255.0)[None, ...]  # (1, H, W, 3)


def extract_clip(video_path: str, seq_len: int, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``seq_len`` evenly-spaced frames (middle 80%%) into a (1, T, H, W, 3) clip."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, int(total * 0.10))
    end = max(start + seq_len, int(total * 0.90))
    indices = np.linspace(start, min(end, max(total - 1, 0)), seq_len, dtype=int)

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from: {video_path}")
    while len(frames) < seq_len:
        frames.append(frames[-1])
    clip = np.stack(frames[:seq_len], axis=0)[None, ...]  # (1, T, H, W, 3)
    preview = (frames[0] * 255.0).astype(np.uint8)
    return clip, preview


# ---------------------------------------------------------------------------
# Prediction + display
# ---------------------------------------------------------------------------

def predict(model: keras.Model, batch: np.ndarray, class_names: List[str], top_k: int) -> List[Tuple[str, float]]:
    probs = model.predict(batch, verbose=0)[0]
    top_k = min(top_k, len(class_names))
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]


def print_predictions(label: str, predictions: List[Tuple[str, float]]) -> None:
    top_class, top_conf = predictions[0]
    print(f"\n{label}")
    print(f"  -> Predicted: {top_class.upper()}  ({top_conf:.1%} confidence)")
    for rank, (cls, conf) in enumerate(predictions, start=1):
        bar = "#" * int(conf * 20)
        print(f"    {rank}. {cls:<16} {conf:6.1%}  {bar}")


def save_overlay(image_rgb: np.ndarray, predictions: List[Tuple[str, float]], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Action Classification", fontsize=14, fontweight="bold")
    axes[0].imshow(image_rgb); axes[0].axis("off"); axes[0].set_title("Input")
    labels = [p[0] for p in predictions][::-1]
    scores = [p[1] for p in predictions][::-1]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(labels)))[::-1]
    bars = axes[1].barh(labels, scores, color=colors)
    axes[1].set_xlim(0, 1); axes[1].set_xlabel("Confidence"); axes[1].set_title("Top Predictions")
    for bar, s in zip(bars, scores):
        axes[1].text(min(s + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                     f"{s:.1%}", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Overlay saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if bool(args.image) == bool(args.video):
        parser.error("Provide exactly one of --image (file/dir for 2D models) or --video (for video models).")

    model_path = args.model or default_checkpoint(args.model_type)
    model = load_model(args.model_type, model_path)
    class_names = list(cfg.data.classes)
    image_size = cfg.data.image_size

    if is_video(args.model_type):
        if not args.video:
            parser.error(f"--video is required for model_type '{args.model_type}'.")
        logger.info("Extracting %d frames from %s ...", cfg.data.sequence_length, args.video)
        clip, preview = extract_clip(args.video, cfg.data.sequence_length, image_size)
        predictions = predict(model, clip, class_names, args.top_k)
        print_predictions(f"Video: {os.path.basename(args.video)}", predictions)
        if args.save_overlay:
            out = os.path.join(cfg.data.results_dir,
                               f"{os.path.splitext(os.path.basename(args.video))[0]}_prediction.png")
            save_overlay(preview, predictions, out)
        return

    # 2D image / directory path
    if not args.image:
        parser.error(f"--image is required for model_type '{args.model_type}'.")
    if os.path.isdir(args.image):
        image_paths = []
        for ext in IMAGE_EXTS:
            image_paths.extend(glob.glob(os.path.join(args.image, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(args.image, f"*{ext.upper()}")))
        image_paths = sorted(set(image_paths))
        if not image_paths:
            logger.error("No images found in directory: %s", args.image)
            sys.exit(1)
    else:
        image_paths = [args.image]

    logger.info("Running inference on %d image(s) with %s ...", len(image_paths), args.model_type)
    for img_path in image_paths:
        try:
            batch = preprocess_image(img_path, image_size)
        except FileNotFoundError as e:
            logger.error("%s - skipping.", e)
            continue
        predictions = predict(model, batch, class_names, args.top_k)
        print_predictions(f"Image: {os.path.basename(img_path)}", predictions)
        if args.save_overlay:
            base = os.path.splitext(os.path.basename(img_path))[0]
            out = os.path.join(cfg.data.results_dir, f"{base}_prediction.png")
            bgr = cv2.imread(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else (batch[0] * 255).astype(np.uint8)
            save_overlay(rgb, predictions, out)
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
