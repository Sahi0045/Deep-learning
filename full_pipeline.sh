#!/usr/bin/env bash
# full_pipeline.sh - Full real-data benchmark: download -> train -> evaluate -> (camera)
#
# Usage:
#   bash full_pipeline.sh                       # run everything (no camera in headless mode)
#   SKIP_DOWNLOAD=1 bash full_pipeline.sh       # reuse existing dataset
#   MODEL_TYPE=efficientnet_v2 bash full_pipeline.sh   # a single family
#   EPOCHS=30 VIDEO_EPOCHS=25 bash full_pipeline.sh
#   SKIP_CAMERA=1 bash full_pipeline.sh        # never launch the camera
set -e

EPOCHS=${EPOCHS:-25}
BATCH_SIZE=${BATCH_SIZE:-32}
VIDEO_EPOCHS=${VIDEO_EPOCHS:-25}
MODEL_TYPE=${MODEL_TYPE:-all}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
SKIP_CAMERA=${SKIP_CAMERA:-0}
# Auto-skip the camera when there is no display.
if [ -z "${DISPLAY:-}" ]; then SKIP_CAMERA=1; fi

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

echo ""
echo "=========================================================="
echo "  Violence-Detection Full Benchmark Pipeline"
echo "  Steps: Download -> Train -> Evaluate -> Camera"
echo "=========================================================="

# -- Step 1: Download / prepare the verified dataset -----------------------
MANIFEST="datasets/airtlab_violence/manifest.csv"
if [ "$SKIP_DOWNLOAD" = "1" ] || [ -f "$MANIFEST" ]; then
    echo ">> Step 1/4 - Dataset already present, skipping download."
else
    echo ">> Step 1/4 - Downloading + preparing the AIRTLab dataset..."
    python3 download_real_data.py
fi

# -- Step 2: Train ---------------------------------------------------------
echo ""
echo ">> Step 2/4 - Training (model_type=${MODEL_TYPE}, ${EPOCHS} 2D / ${VIDEO_EPOCHS} video epochs)..."
python3 train.py --model_type "$MODEL_TYPE" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --video_epochs "$VIDEO_EPOCHS"
echo ""

# -- Step 3: Evaluate ------------------------------------------------------
echo ">> Step 3/4 - Evaluating on the held-out test split..."
python3 evaluate.py --model_type "$MODEL_TYPE" --split test
echo ""

echo "  Saved outputs:"
echo "    results/model_comparison.{csv,md,png}  - unified comparison"
echo "    results/classification_report_<m>.txt  - per-model P/R/F1"
echo "    results/confusion_matrix_<m>.png       - per-model confusion matrix"
echo "    results/roc_curves_<m>.png             - per-model ROC curves"
echo "    results/training_history_<m>.png       - per-model curves"
echo ""

# -- Step 4: Live camera (optional) ---------------------------------------
if [ "$SKIP_CAMERA" = "1" ]; then
    echo ">> Step 4/4 - Skipping camera (SKIP_CAMERA=1 or headless)."
else
    echo ">> Step 4/4 - Launching live camera detection..."
    echo "   Controls: q/ESC=quit | s=screenshot | p=pause | +/-=speed"
    echo ""
    BEST=$(ls checkpoints/best_custom_cnn.keras 2>/dev/null || \
           ls checkpoints/best_*.keras 2>/dev/null | head -1)
    if [ -n "$BEST" ]; then
        python3 live_camera.py --model "$BEST" --camera 0 --interval 5 --top_k 3
    else
        echo "   No checkpoint found; skipping camera."
    fi
fi

echo ""
echo "=========================================================="
echo "  All steps complete."
echo "  Models   -> checkpoints/best_<model_type>.keras"
echo "  Results  -> results/"
echo "  Logs     -> logs/"
echo "=========================================================="
