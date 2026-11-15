#!/usr/bin/env bash
# run_pipeline.sh - End-to-end pipeline: (ensure data) -> train -> evaluate
#
# Usage:
#   bash run_pipeline.sh                       # train + evaluate ALL model families
#   MODEL_TYPE=custom_cnn bash run_pipeline.sh  # a single family
#   EPOCHS=25 BATCH_SIZE=32 bash run_pipeline.sh
set -e

EPOCHS=${EPOCHS:-25}
BATCH_SIZE=${BATCH_SIZE:-32}
VIDEO_EPOCHS=${VIDEO_EPOCHS:-25}
MODEL_TYPE=${MODEL_TYPE:-all}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

echo "=========================================================="
echo "  Violence-Detection Pipeline  (model_type=${MODEL_TYPE})"
echo "=========================================================="

# Step 1: ensure the verified dataset is present
MANIFEST="datasets/airtlab_violence/manifest.csv"
if [ "$SKIP_DOWNLOAD" = "1" ] || [ -f "$MANIFEST" ]; then
    echo ">> Step 1/3 - Dataset present, skipping download."
else
    echo ">> Step 1/3 - Downloading + preparing AIRTLab dataset..."
    python3 download_real_data.py
fi

# Step 2: train
echo ""
echo ">> Step 2/3 - Training (${EPOCHS} 2D epochs / ${VIDEO_EPOCHS} video epochs)..."
python3 train.py --model_type "$MODEL_TYPE" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --video_epochs "$VIDEO_EPOCHS"

# Step 3: evaluate
echo ""
echo ">> Step 3/3 - Evaluating on the held-out test split..."
python3 evaluate.py --model_type "$MODEL_TYPE" --split test

echo ""
echo "Done."
echo "  Model checkpoints -> checkpoints/best_<model_type>.keras"
echo "  Per-model reports  -> results/classification_report_<m>.txt, *.png"
echo "  Comparison report  -> results/model_comparison.{csv,md,png}"
