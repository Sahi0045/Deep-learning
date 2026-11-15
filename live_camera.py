"""
live_camera.py - Real-time violence / action detection via webcam.

Opens the default camera, grabs frames, runs the trained CNN every N frames,
and overlays the top prediction + confidence bar on the live feed.

Features:
  - Live prediction overlay with animated confidence bars
  - Color-coded alert (RED for violent actions, GREEN for non-bullying)
  - FPS counter
  - Screenshot capture with 's' key
  - Graceful fallback if no camera is available (demo mode with synthetic frame)

Usage:
    python3 live_camera.py [--model path/to/model.keras] [--camera 0] [--interval 5]

Controls:
    q / ESC  — quit
    s        — save screenshot to results/
    p        — pause / resume
    +/-      — increase / decrease prediction interval
"""

import argparse
import datetime
import logging
import os
import sys
import time
import collections

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress TF noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras

from config import cfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Classes considered violent — trigger RED alert (the two assault classes)
VIOLENT_CLASSES = {
    "unarmed_assault", "armed_assault",
}

# Color palette (BGR)
COLOR_GREEN  = (50,  220, 50)
COLOR_RED    = (30,  30,  230)
COLOR_ORANGE = (30,  160, 230)
COLOR_WHITE  = (240, 240, 240)
COLOR_BLACK  = (10,  10,  10)
COLOR_CYAN   = (220, 200, 50)
COLOR_GRAY   = (120, 120, 120)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Run training first:  python3 train.py --epochs 40"
        )
    logger.info("Loading model: %s", model_path)
    model = keras.models.load_model(model_path)
    logger.info("Model ready. Input shape: %s", model.input_shape)
    return model


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    """
    Convert a BGR OpenCV frame to a normalised RGB batch for inference.

    Args:
        frame:      Raw BGR frame from cv2.VideoCapture.
        image_size: Target square resolution.

    Returns:
        Float32 array of shape (1, image_size, image_size, 3).
    """
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
    normed  = resized.astype(np.float32) / 255.0
    return np.expand_dims(normed, axis=0)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_overlay(
    frame:       np.ndarray,
    predictions: list,          # [(class_name, confidence), ...]
    fps:         float,
    paused:      bool,
    frame_count: int,
) -> np.ndarray:
    """
    Draw the full HUD overlay on the frame:
      - Top-left: title + FPS
      - Bottom-left: top-3 confidence bars
      - Top-right: alert badge for violent classes
      - Bottom-right: controls hint

    Args:
        frame:       BGR frame to draw on (modified in-place copy returned).
        predictions: Sorted (class, confidence) list from model.
        fps:         Current frames-per-second.
        paused:      Whether inference is paused.
        frame_count: Running frame count for animation pulse.

    Returns:
        Annotated BGR frame.
    """
    h, w = frame.shape[:2]
    out  = frame.copy()

    top_class, top_conf = predictions[0]
    is_violent = top_class in VIOLENT_CLASSES
    alert_color = COLOR_RED if is_violent else COLOR_GREEN

    # ── Semi-transparent panel (bottom) ─────────────────────────────────
    panel_h = 180
    panel   = out[h - panel_h:h, 0:w].copy()
    cv2.rectangle(panel, (0, 0), (w, panel_h), COLOR_BLACK, -1)
    out[h - panel_h:h, 0:w] = cv2.addWeighted(
        panel, 0.70, out[h - panel_h:h, 0:w], 0.30, 0
    )

    # ── Top header bar ───────────────────────────────────────────────────
    cv2.rectangle(out, (0, 0), (w, 50), COLOR_BLACK, -1)
    cv2.putText(out, "Violence / Action Detector", (12, 34),
                FONT, 0.75, COLOR_CYAN, 1, cv2.LINE_AA)
    fps_text = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_text, FONT_SMALL, 0.6, 1)
    cv2.putText(out, fps_text, (w - fw - 12, 32),
                FONT_SMALL, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

    # PAUSED indicator
    if paused:
        cv2.putText(out, "[ PAUSED ]", (w // 2 - 55, 32),
                    FONT, 0.65, COLOR_ORANGE, 1, cv2.LINE_AA)

    # ── Alert badge (top-right) ──────────────────────────────────────────
    badge_label = ("⚠ VIOLENCE DETECTED" if is_violent else "✓ NON-VIOLENT")
    badge_color = alert_color
    # Pulsing effect for violent alerts
    if is_violent:
        alpha = 0.5 + 0.5 * abs(np.sin(frame_count * 0.15))
        badge_color = tuple(int(c * alpha) for c in COLOR_RED)

    cv2.rectangle(out, (w - 270, 60), (w - 10, 100), badge_color, -1)
    cv2.rectangle(out, (w - 270, 60), (w - 10, 100), COLOR_WHITE, 1)
    cv2.putText(out, badge_label, (w - 262, 88),
                FONT_SMALL, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── Main prediction label ────────────────────────────────────────────
    label_text = f"{top_class.upper()}  {top_conf:.0%}"
    cv2.putText(out, label_text, (14, h - panel_h + 38),
                FONT, 1.0, alert_color, 2, cv2.LINE_AA)

    # ── Confidence bars (top 3) ──────────────────────────────────────────
    bar_x     = 14
    bar_y0    = h - panel_h + 60
    bar_w_max = w - 200
    bar_h_px  = 22
    bar_gap   = 30

    for rank, (cls_name, conf) in enumerate(predictions[:3]):
        y  = bar_y0 + rank * bar_gap
        is_top = rank == 0
        col    = alert_color if is_top else COLOR_GRAY

        # Background track
        cv2.rectangle(out, (bar_x, y), (bar_x + bar_w_max, y + bar_h_px),
                      (40, 40, 40), -1)
        # Filled portion
        fill_w = int(bar_w_max * conf)
        cv2.rectangle(out, (bar_x, y), (bar_x + fill_w, y + bar_h_px), col, -1)
        # Label
        cv2.putText(out, f"{cls_name:<14} {conf:.1%}",
                    (bar_x + bar_w_max + 6, y + bar_h_px - 6),
                    FONT_SMALL, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── Controls hint (bottom right) ─────────────────────────────────────
    hints = ["q=Quit  s=Save  p=Pause  +/-=Speed"]
    cv2.putText(out, hints[0], (12, h - 10),
                FONT_SMALL, 0.42, COLOR_GRAY, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Smoothing: running average over last N predictions
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """
    Maintains a rolling window of predictions and returns the smoothed
    class probabilities to reduce flickering.
    """
    def __init__(self, window: int = 5, num_classes: int = 10):
        self.window   = window
        self.num_cls  = num_classes
        self.history  : collections.deque = collections.deque(maxlen=window)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.history.append(probs)
        return np.mean(self.history, axis=0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_camera(
    model_path:    str,
    camera_index:  int,
    pred_interval: int,
    top_k:         int,
) -> None:
    """
    Open the camera and run live violence / action detection.

    Args:
        model_path:    Path to trained .keras model.
        camera_index:  Camera device index (0 = default webcam).
        pred_interval: Run inference every N frames.
        top_k:         Number of top predictions to display.
    """
    # Load model
    model       = load_model(model_path)
    class_names = cfg.data.classes
    image_size  = cfg.data.image_size
    num_classes = len(class_names)

    # Smoother
    smoother = PredictionSmoother(window=5, num_classes=num_classes)

    # Default prediction while model warms up
    default_probs = np.ones(num_classes) / num_classes
    current_preds = [(cls, 1.0 / num_classes) for cls in class_names[:top_k]]

    # Open camera
    logger.info("Opening camera (index %d)...", camera_index)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error("Cannot open camera index %d.", camera_index)
        logger.info("Falling back to DEMO MODE (synthetic frames).")
        _run_demo_mode(model, class_names, image_size, top_k)
        return

    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,         30)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera opened: %dx%d", actual_w, actual_h)

    # Warm up model (JIT compilation)
    logger.info("Warming up model...")
    dummy = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    logger.info("Model warm-up done. Starting live detection...")
    logger.info("Controls: q/ESC=quit | s=screenshot | p=pause | +/-=interval")

    os.makedirs(cfg.data.results_dir, exist_ok=True)

    frame_count  = 0
    paused       = False
    fps_history  : collections.deque = collections.deque(maxlen=30)
    t_last_frame = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame. Retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1

        # FPS
        now = time.perf_counter()
        fps_history.append(1.0 / max(now - t_last_frame, 1e-6))
        t_last_frame = now
        fps = float(np.mean(fps_history))

        # Run inference every pred_interval frames (unless paused)
        if not paused and frame_count % pred_interval == 0:
            batch      = preprocess_frame(frame, image_size)
            raw_probs  = model.predict(batch, verbose=0)[0]
            smooth     = smoother.update(raw_probs)
            top_idx    = np.argsort(smooth)[::-1][:top_k]
            current_preds = [(class_names[i], float(smooth[i])) for i in top_idx]

        # Draw overlay
        display = draw_overlay(frame, current_preds, fps, paused, frame_count)
        cv2.imshow("Violence / Action Detector — Press Q to quit", display)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):          # q or ESC
            break
        elif key == ord('s'):              # screenshot
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(cfg.data.results_dir, f"screenshot_{ts}.png")
            cv2.imwrite(path, display)
            logger.info("Screenshot saved: %s", path)
        elif key == ord('p'):              # pause / resume
            paused = not paused
            logger.info("%s", "Paused." if paused else "Resumed.")
        elif key == ord('+') and pred_interval > 1:
            pred_interval -= 1
            logger.info("Prediction interval: every %d frames", pred_interval)
        elif key == ord('-'):
            pred_interval += 1
            logger.info("Prediction interval: every %d frames", pred_interval)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera closed. Goodbye.")


# ---------------------------------------------------------------------------
# Demo mode (no camera)
# ---------------------------------------------------------------------------

def _run_demo_mode(
    model:       keras.Model,
    class_names: list,
    image_size:  int,
    top_k:       int,
) -> None:
    """
    Demo mode: cycle through synthetic colored frames to simulate camera input.
    Useful for testing the overlay UI without a physical camera.
    """
    logger.info("=== DEMO MODE (no camera) — showing synthetic frames ===")
    logger.info("Press Q or ESC to quit.")

    smoother    = PredictionSmoother(window=3, num_classes=len(class_names))
    frame_count = 0
    fps_history : collections.deque = collections.deque(maxlen=30)
    t_last      = time.perf_counter()

    while True:
        # Generate a cycling colored synthetic frame
        hue       = int((frame_count * 2) % 180)
        hsv_frame = np.full((720, 1280, 3), (hue, 180, 200), dtype=np.uint8)
        frame     = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
        frame_count += 1

        now = time.perf_counter()
        fps_history.append(1.0 / max(now - t_last, 1e-6))
        t_last = now
        fps    = float(np.mean(fps_history))

        if frame_count % 10 == 0:
            batch     = preprocess_frame(frame, image_size)
            raw_probs = model.predict(batch, verbose=0)[0]
            smooth    = smoother.update(raw_probs)
            top_idx   = np.argsort(smooth)[::-1][:top_k]
            preds     = [(class_names[i], float(smooth[i])) for i in top_idx]
        else:
            preds = [(cls, 0.1) for cls in class_names[:top_k]]

        display = draw_overlay(frame, preds, fps, False, frame_count)
        cv2.putText(display, "DEMO MODE — No Camera", (display.shape[1]//2 - 160, 80),
                    FONT_SMALL, 0.9, COLOR_ORANGE, 2, cv2.LINE_AA)
        cv2.imshow("Violence / Action Detector — DEMO", display)

        key = cv2.waitKey(33) & 0xFF
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live webcam violence / action detection."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a trained .keras checkpoint (default: first checkpoints/best_*.keras).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0 = built-in webcam).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Run inference every N frames (default: 5). Lower = more responsive.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    args = parser.parse_args()

    if args.model is None:
        import glob
        # live_camera does single-frame (2D) inference, so only consider 2D models.
        video_prefixes = ("best_cnn_lstm", "best_conv3d")
        candidates = sorted(
            c for c in glob.glob(os.path.join(cfg.data.model_dir, "best_*.keras"))
            if not os.path.basename(c).startswith(video_prefixes)
        )
        if not candidates:
            logger.error("No 2D checkpoint found in %s. Run train.py for a 2D model first.", cfg.data.model_dir)
            sys.exit(1)
        args.model = candidates[0]

    logger.info("=" * 60)
    logger.info("Live Violence / Action Detector")
    logger.info("=" * 60)
    logger.info("Model    : %s", args.model)
    logger.info("Camera   : index %d", args.camera)
    logger.info("Interval : every %d frames", args.interval)
    logger.info("Top-K    : %d predictions shown", args.top_k)

    run_camera(
        model_path   = args.model,
        camera_index = args.camera,
        pred_interval= args.interval,
        top_k        = args.top_k,
    )


if __name__ == "__main__":
    main()
