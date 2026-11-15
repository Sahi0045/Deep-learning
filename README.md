# Violence Detection in Videos — Multi-Model Benchmark

A TensorFlow 2.x / Keras 3 deep-learning pipeline that benchmarks four model
families for classifying short video clips into three action classes on the
peer-reviewed **AIRTLab Violence Detection Dataset**:

- **`custom_cnn`** — 4-block 2D Residual CNN baseline (single frames)
- **`efficientnet_v2`** — EfficientNetV2B0 transfer learning (single frames)
- **`cnn_lstm`** — Spatio-temporal CNN + Bidirectional LSTM (video clips)
- **`conv3d`** — Volumetric 3D CNN (video clips)

`--model_type all` trains every family and writes a unified comparison report
(`results/model_comparison.{csv,md,png}`). The data pipeline splits **by
scene** so that neither frames from one clip nor the two camera views of the
same event leak across train/val/test.

| Attribute | Value |
|-----------|-------|
| Framework | TensorFlow 2.x / Keras 3 |
| Dataset | AIRTLab Violence Detection (DOI [10.1016/j.dib.2020.106587](https://doi.org/10.1016/j.dib.2020.106587), CC-BY-4.0) |
| Classes | 3: `nonviolent`, `unarmed_assault`, `armed_assault` |
| Input | 128×128×3 frames; video models use 8-frame clips (8,128,128,3) |
| Split | 70 / 15 / 15 by `scene_id` (zero-leakage verified) |

> Dataset preparation, class mapping, and the manifest schema are documented in
> [`datasets/airtlab_violence/README.md`](datasets/airtlab_violence/README.md).

---

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset (downloads 350 MP4s + extracts frames + writes manifest)
```bash
python3 download_real_data.py
```
Re-running is safe: existing videos and frames are reused.

### 3. Train all four families and produce the comparison report
```bash
bash run_pipeline.sh                 # = download -> train --model_type all -> evaluate
# or directly:
python3 train.py --model_type all
python3 evaluate.py --model_type all
```

Train a single family, or add transfer-learning fine-tuning:
```bash
python3 train.py --model_type custom_cnn --epochs 25
python3 train.py --model_type efficientnet_v2 --fine_tune
```

### 4. Predict
```bash
python3 predict.py --model_type custom_cnn      --image path/to/img.jpg --top_k 3
python3 predict.py --model_type efficientnet_v2 --image path/to/folder/
python3 predict.py --model_type cnn_lstm       --video path/to/clip.mp4
```

### 5. Live camera (optional, needs a display + webcam)
```bash
python3 live_camera.py --camera 0 --interval 5
```

### 6. Tests
```bash
python3 tests/test_pipeline.py            # standalone runner
# SMOKE=1 python3 tests/test_pipeline.py   # also runs a 1-epoch training smoke test
```

---

## Fast, feature-cached transfer learning

EfficientNetV2B0's backbone is frozen, so its pooled features are computed
**once** for every frame and cached. The classification head is then trained on
the cache, which is far faster than re-running the backbone each epoch. The
trained head weights are reassembled into a **self-contained, end-to-end Keras
model** (`checkpoints/best_efficientnet_v2.keras`) whose inference is identical
to the cached head's (verified to <1e-4 in the tests). Optional end-to-end
fine-tuning (`--fine_tune`) unfreezes the top backbone layers afterward, with
the BatchNorm layers kept frozen.

Disable caching with `--no_cache_features` to train the frozen backbone
end-to-end on the image pipeline instead.

---

## Outputs

Per model (`<m>` = the model type, e.g. `custom_cnn`):
```
checkpoints/best_<m>.keras        # self-contained full model
checkpoints/final_<m>.keras
results/training_history_<m>.png
results/confusion_matrix_<m>.png      # train-time evaluation
results/roc_curves_<m>.png
results/classification_report_<m>.txt
logs/training_log_<m>.csv
logs/<m>/                              # TensorBoard
```

Unified benchmark report (from `train.py --model_type all` / `evaluate.py`):
```
results/model_comparison.csv   # machine-readable results table
results/model_comparison.md   # human-readable: dataset summary + results + winner
results/model_comparison.png   # grouped bar chart of accuracy / F1 / AUC
```

Monitor training:
```bash
tensorboard --logdir logs/
```

---

## Configuration

All hyperparameters are centralized in [`config.py`](config.py) and mirrored in
[`experiment_config.yaml`](experiment_config.yaml) (YAML overrides on load). No
magic numbers elsewhere.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 128 | Frame resolution |
| `sequence_length` | 8 | Frames per video clip |
| `train/val/test_split` | 0.70 / 0.15 / 0.15 | Split fractions (by `scene_id`) |
| `epochs` / `video_epochs` | 25 / 25 | 2D / video training epochs |
| `batch_size` / `video_batch_size` | 32 / 16 | Mini-batch sizes |
| `learning_rate` | 1e-3 | Initial Adam LR |
| `lr_decay_factor` / `lr_patience` | 0.5 / 4 | ReduceLROnPlateau |
| `early_stop_patience` | 8 | EarlyStopping on `val_loss` |
| `conv_dropout` / `fc_dropout` | 0.20 / 0.40 | Spatial dropout (conv) / dropout (FC) |
| `l2_lambda` | 1e-4 | Weight regularization |
| `augmentation` | horizontal flip + photometric | vertical flip disabled (upright scenes) |
| `fine_tune_at` | 180 | EfficientNetV2 layer index to start unfreezing |
| `lstm_units` | 128 | CNN-LSTM temporal units |
| `conv3d_filters` | [32, 64, 128] | 3D CNN filter schedule |

---

## Model families

**Custom 2D Residual CNN** — four conv blocks (`Conv→BN→ReLU→Conv→BN→ReLU +
skip → MaxPool → SpatialDropout`) with widths 32/64/128/256, Global Average
Pooling, then FC(256)→FC(128)→softmax. ~1.28M params.

**EfficientNetV2B0 transfer** — frozen ImageNet-pretrained backbone (loaded
from `checkpoints/pretrained/efficientnetv2-b0_notop.h5`), GAP, dropout, softmax
head. ~5.9M params (mostly the frozen backbone). Fast cached-head training by
default.

**CNN-BiLSTM (video)** — a frozen EfficientNetV2 spatial encoder wrapped in
`TimeDistributed`, an additive residual design: a static-appearance logit
(averaged per-frame logits) plus a temporal-delta logit from a Bidirectional
LSTM over the projected frame features + per-frame logits. ~6.0M params.

**3D CNN (video)** — volumetric residual blocks (`Conv3D→BN→ReLU→Conv3D→BN→ReLU
+ skip → MaxPool3D`) with filters [32, 64, 128], GAP3D, FC, softmax. ~0.88M
params.

---

## Illustrative result (1-epoch smoke run)

These numbers come from a quick 1-epoch run used to validate the pipeline, not
a tuned benchmark — run `train.py --model_type all` with the default 25 epochs
for the real comparison:

| Model | Params | Test Acc | Weighted F1 | Mean AUC |
|-------|--------|----------|-------------|----------|
| Custom CNN | 1.28M | 33.3% | 0.167 | 0.582 |
| EfficientNetV2 (cached head) | 5.92M | 58.3% | 0.566 | 0.761 |
| CNN-BiLSTM (video) | 5.98M | 59.3% | 0.560 | 0.765 |
| 3D CNN (video) | 0.88M | 33.3% | 0.167 | 0.504 |

After a single epoch the two transfer/pretrained-backbone models are already at
~58–59% with ~0.76 AUC, while the two from-scratch models are still at chance
(33%). That gap is exactly why transfer learning + the cached head are the
recommended starting point; the from-scratch CNNs need many more epochs.

---

## Project structure

```
config.py                  Centralized config (dataclasses + YAML)
data_pipeline.py           Scene-grouped loading, augmentation, tf.data pipelines
model.py                   4 architectures + feature-caching + factory/compile
report_utils.py            Shared metrics, plots, comparison-report writer
train.py                   Multi-model training + benchmark (--model_type all)
evaluate.py                Standalone evaluation + comparison report
predict.py                 Inference (image/dir for 2D, video for temporal models)
live_camera.py             Real-time webcam detection
download_real_data.py       Download + extract + split AIRTLab; write manifest
generate_synthetic_data.py  Isolated synthetic smoke-test data (never benchmark)
run_pipeline.sh            End-to-end: download -> train all -> evaluate
full_pipeline.sh           Above + optional live camera
tests/test_pipeline.py     Automated tests
experiment_config.yaml      Reproducible experiment config
datasets/airtlab_violence/  Dataset + manifest (+ README documenting provenance)
training_data/             Extracted frames, class sub-directories
checkpoints/               Saved models + pretrained EfficientNetV2 weights
results/                   Plots + reports (+ model_comparison.*)
logs/                      TensorBoard + per-model training CSVs
legacy/                    Original 2019 TF 1.x project (not used by this pipeline)
```

---

## References

- Bianculli et al. (2020). *A dataset for automatic violence detection in videos.* Data in Brief. [DOI](https://doi.org/10.1016/j.dib.2020.106587)
- Sernani et al. (2021). *Detecting violence in videos by means of deep learning and optical flow.* IEEE Access.
- Tan & Le (2021). *EfficientNetV2: Smaller Models and Faster Training.* [arXiv](https://arxiv.org/abs/2104.00298)
- He et al. (2015). *Deep Residual Learning for Image Recognition.* [arXiv](https://arxiv.org/abs/1512.03385)
- Ioffe & Szegedy (2015). *Batch Normalization.* [arXiv](https://arxiv.org/abs/1502.03167)

## Notes on use

This dataset depicts staged physical violence for research. Treat any
downstream deployment with care: evaluate bias, obtain consent where relevant,
and do not use it for surveillance of real people without oversight.

## Team

Original CPSC 8810 (Deep Learning, Clemson University) project by Mihir Phatak
and Netra Inamdar. The current pipeline is a TensorFlow 2.x / Keras 3 rewrite
that benchmarks multiple model families on the AIRTLab dataset.
