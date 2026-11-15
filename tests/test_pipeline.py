"""
tests/test_pipeline.py - Automated tests for the violence-detection pipeline.

Runnable two ways:
  python3 tests/test_pipeline.py            # standalone runner
  pytest tests/test_pipeline.py              # pytest, if installed

Fast tests cover pure logic and model construction. Integration tests that
touch the real AIRTLab data are tagged and skip gracefully when the dataset or
manifest is absent. Set SMOKE=1 to also run a 1-epoch training smoke test.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

# Make the workspace root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model import VIDEO_TYPES, build_model_by_name, compile_model
from data_pipeline import (
    is_synthetic_filename,
    extract_group_id_from_filename,
    verify_zero_group_leakage,
)

RESULTS_DIR = cfg.data.results_dir
MANIFEST = cfg.data.manifest_path
HAS_MANIFEST = os.path.exists(MANIFEST)

_passed = 0
_failed = 0


def _check(name, fn):
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"  PASS  {name}")
    except Exception as e:  # noqa
        _failed += 1
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def skip(name, fn, condition):
    if not condition:
        print(f"  SKIP  {name} (precondition not met)")
        return
    _check(name, fn)


# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------

def test_synthetic_filter():
    assert is_synthetic_filename("synthetic_img_0001.jpg")
    assert is_synthetic_filename("img_0003.jpg")
    assert not is_synthetic_filename("nonviolent_scene_001_cam1_frame_00.jpg")
    assert not is_synthetic_filename("real_00007.jpg")


def test_group_id_extraction():
    assert extract_group_id_from_filename("violent_scene_001_cam1_frame_03.jpg") == "violent_scene_001"
    assert extract_group_id_from_filename("nonviolent_scene_012_cam2_frame_0.jpg") == "nonviolent_scene_012"
    # Falls back to stripping the trailing token.
    assert extract_group_id_from_filename("clip02_frame01.jpg") == "clip02"


def test_leakage_detection_raises():
    try:
        verify_zero_group_leakage(np.array(["a", "b"]), np.array(["b"]), np.array(["c"]))
        raise AssertionError("expected AssertionError for train/val overlap")
    except AssertionError as e:
        assert "leakage" in str(e).lower()


def test_leakage_clean():
    info = verify_zero_group_leakage(np.array(["a", "b"]), np.array(["c"]), np.array(["d", "e"]))
    assert info["zero_leakage"] is True
    assert info["train_unique_groups"] == 2


def test_config_classes():
    assert cfg.data.classes == ["nonviolent", "unarmed_assault", "armed_assault"]
    assert cfg.data.num_channels == 3


# ---------------------------------------------------------------------------
# Model construction tests (no data needed)
# ---------------------------------------------------------------------------

def _build_and_check(mtype):
    model = build_model_by_name(mtype)
    model = compile_model(model, learning_rate=cfg.training.learning_rate)
    out_dim = model.output_shape[-1]
    assert out_dim == len(cfg.data.classes), (mtype, out_dim)
    return model


def test_custom_cnn():
    m = _build_and_check("custom_cnn")
    assert m.name == "CustomResCNN2D"
    assert m.count_params() > 0


def test_efficientnet_v2():
    m = _build_and_check("efficientnet_v2")
    assert m.name == "EfficientNetV2_Transfer"


def test_cnn_lstm():
    m = _build_and_check("cnn_lstm")
    assert m.name == "Video_CNN_BiLSTM"


def test_conv3d():
    m = _build_and_check("conv3d")
    assert m.name == "Video_Res3DCNN"


def test_video_types_constant():
    assert VIDEO_TYPES == {"cnn_lstm", "conv3d"}


def test_feature_extractor_and_reassembly():
    from model import build_feature_extractor, build_classification_head, reassemble_transfer_model
    ext, fdim = build_feature_extractor(cfg.data.image_size, cfg.data.num_channels)
    assert fdim == 1280
    head = build_classification_head(fdim, len(cfg.data.classes),
                                     fc_dropout=cfg.model.fc_dropout, l2_lambda=cfg.model.l2_lambda)
    head = compile_model(head, learning_rate=cfg.training.learning_rate)
    x = np.random.rand(4, fdim).astype(np.float32)
    y = np.eye(len(cfg.data.classes))[np.random.randint(0, len(cfg.data.classes), size=4)].astype(np.float32)
    head.fit(x, y, epochs=1, batch_size=4, verbose=0)
    w, b = head.get_layer("predictions").get_weights()
    full = reassemble_transfer_model(cfg.data.image_size, cfg.data.num_channels,
                                      len(cfg.data.classes), (w, b),
                                      fc_dropout=cfg.model.fc_dropout, l2_lambda=cfg.model.l2_lambda)
    # Cached-head inference must equal reassembled-model inference (dropout off).
    img = np.random.rand(2, cfg.data.image_size, cfg.data.image_size, 3).astype(np.float32)
    a = head.predict(ext.predict(img, verbose=0), verbose=0)
    bb = full.predict(img, verbose=0)
    assert np.abs(a - bb).max() < 1e-4, np.abs(a - bb).max()


# ---------------------------------------------------------------------------
# Data-pipeline integration tests (need the real manifest)
# ---------------------------------------------------------------------------

def test_get_datasets_shape_and_counts():
    from data_pipeline import get_datasets
    d = get_datasets(augment=False)
    assert d["num_classes"] == 3
    assert d["X_train"].shape[1:] == (cfg.data.image_size, cfg.data.image_size, 3)
    assert d["leakage_check"]["zero_leakage"] is True
    assert set(d["per_class_counts"]["total"].keys()) == set(cfg.data.classes)


def test_get_video_datasets_shape():
    from data_pipeline import get_video_datasets
    d = get_video_datasets(augment=False)
    assert d["X_train"].shape[1:] == (cfg.data.sequence_length, cfg.data.image_size, cfg.data.image_size, 3)
    assert d["leakage_check"]["zero_leakage"] is True


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------

def test_compute_metrics_and_comparison_report():
    from report_utils import compute_metrics, write_comparison_report
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=20)
    probs = rng.random((20, 3)); probs /= probs.sum(axis=1, keepdims=True)
    y_pred = np.argmax(probs, axis=1)
    metrics = compute_metrics(y_true, y_pred, probs, cfg.data.classes)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["per_class_accuracy"]) == 3
    rows = [{"model_type": "custom_cnn", "name": "CustomResCNN2D", "params": 1000,
             "train_time_s": 1.0, "best_val_accuracy": 0.5,
             "accuracy": metrics["accuracy"], "weighted_f1": metrics["weighted_f1"],
             "macro_f1": metrics["macro_f1"], "mean_auc": metrics["mean_auc"], "status": "ok"}]
    paths = write_comparison_report(rows, RESULTS_DIR, cfg.data.classes,
                                     per_class_counts={"train": {c: 10 for c in cfg.data.classes}},
                                     dataset_name="test")
    for p in paths.values():
        assert os.path.exists(p), p


# ---------------------------------------------------------------------------
# Optional training smoke test (set SMOKE=1)
# ---------------------------------------------------------------------------

def test_train_smoke():
    from data_pipeline import get_datasets
    from model import build_model_by_name, compile_model
    d = get_datasets(augment=False)
    n = 16
    m = build_model_by_name("custom_cnn")
    m = compile_model(m, learning_rate=1e-3)
    m.fit(d["X_train"][:n], d["y_train"][:n], epochs=1, batch_size=8, verbose=0)
    out = m.predict(d["X_test"][:4], verbose=0)
    assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("Running pure-logic tests:")
    for n, f in [("synthetic_filter", test_synthetic_filter),
                 ("group_id_extraction", test_group_id_extraction),
                 ("leakage_raises", test_leakage_detection_raises),
                 ("leakage_clean", test_leakage_clean),
                 ("config_classes", test_config_classes)]:
        _check(n, f)

    print("\nRunning model-construction tests:")
    for n, f in [("custom_cnn", test_custom_cnn),
                 ("efficientnet_v2", test_efficientnet_v2),
                 ("cnn_lstm", test_cnn_lstm),
                 ("conv3d", test_conv3d),
                 ("video_types", test_video_types_constant),
                 ("feature_cache_reassembly", test_feature_extractor_and_reassembly)]:
        _check(n, f)

    print("\nRunning data-pipeline integration tests:")
    skip("get_datasets", test_get_datasets_shape_and_counts, HAS_MANIFEST)
    skip("get_video_datasets", test_get_video_datasets_shape, HAS_MANIFEST)

    print("\nRunning reporting tests:")
    _check("compute_metrics+comparison_report", test_compute_metrics_and_comparison_report)

    print("\nRunning optional smoke test (SMOKE=1 to enable):")
    skip("train_smoke", test_train_smoke, os.environ.get("SMOKE") == "1")

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
