"""
data_pipeline.py - Scene-grouped data loading, preprocessing, and augmentation pipeline.

Key capabilities:
  - Loads verified real-world video sequences and extracted frames from
    datasets/airtlab_violence/manifest.csv or directory structures.
  - Enforces synthetic image exclusion (exclude_synthetic=True by default).
  - Performs stratified splitting grouped by scene/video/source ID (scene_id),
    preventing both frame-level and multi-camera (cam1/cam2) data leakage.
  - Formal zero-leakage verification across train, validation, and test splits.
  - Builds tf.data pipelines for both 2D image models (H, W, 3) and
    spatio-temporal video models (T, H, W, 3).
"""

import csv
import glob
import logging
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYNTHETIC_PATTERNS = (
    re.compile(r"^synthetic_", re.IGNORECASE),
    re.compile(r"^img_\d{4}\.(jpg|jpeg|png)$", re.IGNORECASE),
)


def is_synthetic_filename(filename: str) -> bool:
    """Return True if a filename matches known synthetic generator naming patterns."""
    base = os.path.basename(filename)
    return any(pat.match(base) for pat in SYNTHETIC_PATTERNS)


def extract_group_id_from_filename(filename: str) -> str:
    """
    Extract scene/video/source group ID from a filename.

    Examples:
      'violent_scene_001_cam1_frame_00.jpg' -> 'violent_scene_001'
      'nonviolent_scene_012_cam2_frame_03.jpg' -> 'nonviolent_scene_012'
      'subject05_clip02_frame01.jpg' -> 'subject05_clip02'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^((?:non)?violent_scene_\d+)", base)
    if m:
        return m.group(1)
    m2 = re.match(r"^(.+?)(?:_cam\d+)?(?:_frame_?\d+)$", base)
    if m2:
        return m2.group(1)
    parts = base.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:-1])
    return base


def verify_zero_group_leakage(
    train_groups: np.ndarray,
    val_groups: np.ndarray,
    test_groups: np.ndarray,
) -> Dict[str, Any]:
    """
    Verify that no scene/video/source group ID appears in more than one split.

    Raises:
        AssertionError: If any group ID overlaps across train, val, or test.
    """
    s_train = set(train_groups.tolist())
    s_val = set(val_groups.tolist())
    s_test = set(test_groups.tolist())

    overlap_tv = s_train.intersection(s_val)
    overlap_tt = s_train.intersection(s_test)
    overlap_vt = s_val.intersection(s_test)

    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Data leakage detected across splits! Overlaps: "
            f"train∩val={overlap_tv}, train∩test={overlap_tt}, val∩test={overlap_vt}"
        )

    return {
        "zero_leakage": True,
        "train_unique_groups": len(s_train),
        "val_unique_groups": len(s_val),
        "test_unique_groups": len(s_test),
    }


def load_image(image_path: str, image_size: int) -> Optional[np.ndarray]:
    """Load, convert BGR->RGB, resize, and normalize a single image to [0, 1]."""
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("Could not read image: %s — skipping.", image_path)
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[0] != image_size or image.shape[1] != image_size:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def load_dataset(
    train_path: str,
    image_size: int,
    classes: List[str],
    exclude_synthetic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load images from class subdirectories, filtering synthetic images if requested
    and extracting scene/video group IDs.

    Returns:
        Tuple of (images, one_hot_labels, filenames, class_names, group_ids).
    """
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    filenames: List[str] = []
    class_names: List[str] = []
    group_ids: List[str] = []

    num_classes = len(classes)
    total_loaded = 0
    total_skipped = 0
    synthetic_filtered = 0

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(train_path, class_name)
        if not os.path.isdir(class_dir):
            logger.warning("Class directory not found: %s — skipping.", class_dir)
            continue

        pattern = os.path.join(class_dir, "**", "*")
        all_files = sorted(glob.glob(pattern, recursive=True))
        image_files = [
            f for f in all_files
            if os.path.isfile(f) and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        for img_path in image_files:
            if exclude_synthetic and is_synthetic_filename(img_path):
                synthetic_filtered += 1
                continue
            img = load_image(img_path, image_size)
            if img is None:
                total_skipped += 1
                continue

            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[class_idx] = 1.0

            fname = os.path.basename(img_path)
            images.append(img)
            labels.append(one_hot)
            filenames.append(fname)
            class_names.append(class_name)
            group_ids.append(extract_group_id_from_filename(fname))
            total_loaded += 1

    if synthetic_filtered > 0:
        logger.info("Filtered out %d synthetic images (exclude_synthetic=True).", synthetic_filtered)

    if total_loaded == 0:
        raise ValueError(
            f"No valid verified images found in '{train_path}' "
            f"(filtered {synthetic_filtered} synthetic files)."
        )

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(filenames, dtype=str),
        np.array(class_names, dtype=str),
        np.array(group_ids, dtype=str),
    )


def split_dataset_by_group(
    images: np.ndarray,
    labels: np.ndarray,
    filenames: np.ndarray,
    class_names: np.ndarray,
    group_ids: np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
) -> Tuple:
    """
    Stratified group split ensuring all samples sharing a group_id (scene/video/person)
    are assigned to the exact same split.
    """
    unique_groups: List[str] = []
    group_label_map: Dict[str, int] = {}
    int_labels = np.argmax(labels, axis=1)

    for g, lbl in zip(group_ids.tolist(), int_labels.tolist()):
        if g not in group_label_map:
            unique_groups.append(g)
            group_label_map[g] = lbl

    unique_groups_arr = np.array(unique_groups)
    unique_group_labels = np.array([group_label_map[g] for g in unique_groups])

    # Stratified split on unique groups
    holdout_frac = val_split + test_split
    train_g, temp_g, _, temp_y = train_test_split(
        unique_groups_arr,
        unique_group_labels,
        test_size=holdout_frac,
        stratify=unique_group_labels,
        random_state=random_seed,
    )
    test_rel = test_split / holdout_frac
    val_g, test_g = train_test_split(
        temp_g,
        test_size=test_rel,
        stratify=temp_y,
        random_state=random_seed,
    )

    train_set = set(train_g.tolist())
    val_set = set(val_g.tolist())
    test_set = set(test_g.tolist())

    train_mask = np.array([g in train_set for g in group_ids])
    val_mask = np.array([g in val_set for g in group_ids])
    test_mask = np.array([g in test_set for g in group_ids])

    verify_zero_group_leakage(
        group_ids[train_mask], group_ids[val_mask], group_ids[test_mask]
    )

    return (
        images[train_mask], images[val_mask], images[test_mask],
        labels[train_mask], labels[val_mask], labels[test_mask],
        filenames[train_mask], filenames[val_mask], filenames[test_mask],
        class_names[train_mask], class_names[val_mask], class_names[test_mask],
        group_ids[train_mask], group_ids[val_mask], group_ids[test_mask],
    )


def _compute_split_class_counts(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    classes: List[str],
) -> Dict[str, Dict[str, int]]:
    """Return per-class sample counts across train, val, test, and total."""
    counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}, "total": {}}
    for split_name, y_arr in [("train", y_train), ("val", y_val), ("test", y_test)]:
        c = Counter(np.argmax(y_arr, axis=1).tolist()) if len(y_arr) > 0 else Counter()
        for idx, cls_name in enumerate(classes):
            counts[split_name][cls_name] = int(c.get(idx, 0))
    for cls_name in classes:
        counts["total"][cls_name] = (
            counts["train"][cls_name] + counts["val"][cls_name] + counts["test"][cls_name]
        )
    return counts


# ---------------------------------------------------------------------------
# Augmentations (upright human interaction safe — no vertical flip by default)
# ---------------------------------------------------------------------------

def _augment_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply realistic photometric and horizontal-flip augmentations to a single frame."""
    if cfg.training.horizontal_flip:
        image = tf.image.random_flip_left_right(image)
    if cfg.training.vertical_flip:
        image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.88, upper=1.12)
    image = tf.image.random_saturation(image, lower=0.88, upper=1.12)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def _augment_video(clip: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply temporally consistent augmentation across all T frames of a video clip."""
    if cfg.training.horizontal_flip:
        do_flip = tf.random.uniform([]) > 0.5
        clip = tf.cond(do_flip, lambda: tf.reverse(clip, axis=[2]), lambda: clip)
    delta = tf.random.uniform([], -0.10, 0.10)
    clip = tf.clip_by_value(clip + delta, 0.0, 1.0)
    return clip, label


def build_tf_dataset(
    inputs: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    is_video: bool = False,
) -> tf.data.Dataset:
    """Build a batched, prefetched tf.data.Dataset for 2D images or 3D video clips."""
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(inputs), seed=seed, reshuffle_each_iteration=True
        )
    if augment:
        aug_fn = _augment_video if is_video else _augment_image
        dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Manifest-backed loaders (guaranteed identical scene splits for frames & videos)
# ---------------------------------------------------------------------------

def get_datasets(
    train_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    image_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    val_split: Optional[float] = None,
    test_split: Optional[float] = None,
    augment: Optional[bool] = None,
    random_seed: Optional[int] = None,
    classes: Optional[List[str]] = None,
    exclude_synthetic: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Load 2D frame datasets with strict scene/video group splitting.
    Uses manifest.csv when available so frame and video splits match identically.
    """
    dc = cfg.data
    tc = cfg.training

    train_path = train_path or dc.train_path
    manifest_path = manifest_path or dc.manifest_path
    image_size = image_size or dc.image_size
    batch_size = batch_size or tc.batch_size
    val_split = val_split if val_split is not None else dc.validation_split
    test_split = test_split if test_split is not None else dc.test_split
    augment = augment if augment is not None else tc.augmentation
    random_seed = random_seed or dc.random_seed
    classes = classes or dc.classes
    exclude_synthetic = (
        exclude_synthetic if exclude_synthetic is not None else dc.exclude_synthetic
    )
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if manifest_path and os.path.exists(manifest_path) and train_path == dc.train_path:
        logger.info("Loading scene-grouped frame dataset from manifest: %s", manifest_path)
        splits: Dict[str, Dict[str, list]] = {
            sp: {"X": [], "y": [], "fn": [], "cn": [], "grp": [], "vid": []}
            for sp in ["train", "val", "test"]
        }
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for r in rows:
            cls_name = r["class_name"]
            if cls_name not in class_to_idx:
                continue
            sp = r["split"]
            cls_idx = class_to_idx[cls_name]
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[cls_idx] = 1.0

            for fpath in r["frame_paths"].split(";"):
                if exclude_synthetic and is_synthetic_filename(fpath):
                    continue
                img = load_image(fpath, image_size)
                if img is None:
                    continue
                splits[sp]["X"].append(img)
                splits[sp]["y"].append(one_hot)
                splits[sp]["fn"].append(f"{r['video_id']}_{os.path.basename(fpath)}")
                splits[sp]["cn"].append(cls_name)
                splits[sp]["grp"].append(r["scene_id"])
                splits[sp]["vid"].append(r["video_id"])

        X_train = np.array(splits["train"]["X"], dtype=np.float32)
        X_val = np.array(splits["val"]["X"], dtype=np.float32)
        X_test = np.array(splits["test"]["X"], dtype=np.float32)
        y_train = np.array(splits["train"]["y"], dtype=np.float32)
        y_val = np.array(splits["val"]["y"], dtype=np.float32)
        y_test = np.array(splits["test"]["y"], dtype=np.float32)
        fn_train = np.array(splits["train"]["fn"], dtype=str)
        fn_val = np.array(splits["val"]["fn"], dtype=str)
        fn_test = np.array(splits["test"]["fn"], dtype=str)
        cn_train = np.array(splits["train"]["cn"], dtype=str)
        cn_val = np.array(splits["val"]["cn"], dtype=str)
        cn_test = np.array(splits["test"]["cn"], dtype=str)
        grp_train = np.array(splits["train"]["grp"], dtype=str)
        grp_val = np.array(splits["val"]["grp"], dtype=str)
        grp_test = np.array(splits["test"]["grp"], dtype=str)
        vid_test = np.array(splits["test"]["vid"], dtype=str)
    else:
        images, labels, filenames, class_names_arr, group_ids = load_dataset(
            train_path=train_path,
            image_size=image_size,
            classes=classes,
            exclude_synthetic=exclude_synthetic,
        )
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            fn_train, fn_val, fn_test,
            cn_train, cn_val, cn_test,
            grp_train, grp_val, grp_test,
        ) = split_dataset_by_group(
            images, labels, filenames, class_names_arr, group_ids,
            val_split=val_split,
            test_split=test_split,
            random_seed=random_seed,
        )
        vid_test = grp_test

    leakage_info = verify_zero_group_leakage(grp_train, grp_val, grp_test)
    per_class_counts = _compute_split_class_counts(y_train, y_val, y_test, classes)

    logger.info(
        "Grouped Frame Split — Train: %d (%d scenes) | Val: %d (%d scenes) | Test: %d (%d scenes)",
        len(X_train), leakage_info["train_unique_groups"],
        len(X_val), leakage_info["val_unique_groups"],
        len(X_test), leakage_info["test_unique_groups"],
    )

    train_ds = build_tf_dataset(
        X_train, y_train, batch_size, augment=augment, shuffle=True, seed=random_seed
    )
    val_ds = build_tf_dataset(X_val, y_val, batch_size, augment=False, shuffle=False)
    test_ds = build_tf_dataset(X_test, y_test, batch_size, augment=False, shuffle=False)

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "fn_train": fn_train,
        "fn_val": fn_val,
        "fn_test": fn_test,
        "cn_train": cn_train,
        "cn_val": cn_val,
        "cn_test": cn_test,
        "train_groups": grp_train,
        "val_groups": grp_val,
        "test_groups": grp_test,
        "test_video_ids": vid_test,
        "leakage_check": leakage_info,
        "per_class_counts": per_class_counts,
        "class_names": classes,
        "num_classes": num_classes,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }


def get_video_datasets(
    manifest_path: Optional[str] = None,
    image_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    augment: Optional[bool] = None,
    random_seed: Optional[int] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load spatio-temporal video clip sequences of shape (N, T, H, W, 3) grouped by scene_id.
    """
    dc = cfg.data
    tc = cfg.training

    manifest_path = manifest_path or dc.manifest_path
    image_size = image_size or dc.image_size
    seq_len = seq_len or dc.sequence_length
    batch_size = batch_size or tc.video_batch_size
    augment = augment if augment is not None else tc.augmentation
    random_seed = random_seed or dc.random_seed
    classes = classes or dc.classes
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Video manifest not found at '{manifest_path}'. Run download_real_data.py first."
        )

    splits: Dict[str, Dict[str, list]] = {
        sp: {"X": [], "y": [], "vid": [], "grp": [], "cn": []}
        for sp in ["train", "val", "test"]
    }

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        cls_name = r["class_name"]
        if cls_name not in class_to_idx:
            continue
        sp = r["split"]
        cls_idx = class_to_idx[cls_name]
        fpaths = r["frame_paths"].split(";")[:seq_len]
        frames = [load_image(fp, image_size) for fp in fpaths]
        frames = [fr for fr in frames if fr is not None]
        if len(frames) == 0:
            continue
        while len(frames) < seq_len:
            frames.append(frames[-1])
        clip = np.stack(frames[:seq_len], axis=0)  # (T, H, W, 3)

        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[cls_idx] = 1.0

        splits[sp]["X"].append(clip)
        splits[sp]["y"].append(one_hot)
        splits[sp]["vid"].append(r["video_id"])
        splits[sp]["grp"].append(r["scene_id"])
        splits[sp]["cn"].append(cls_name)

    X_train = np.array(splits["train"]["X"], dtype=np.float32)
    X_val = np.array(splits["val"]["X"], dtype=np.float32)
    X_test = np.array(splits["test"]["X"], dtype=np.float32)
    y_train = np.array(splits["train"]["y"], dtype=np.float32)
    y_val = np.array(splits["val"]["y"], dtype=np.float32)
    y_test = np.array(splits["test"]["y"], dtype=np.float32)
    grp_train = np.array(splits["train"]["grp"], dtype=str)
    grp_val = np.array(splits["val"]["grp"], dtype=str)
    grp_test = np.array(splits["test"]["grp"], dtype=str)
    vid_test = np.array(splits["test"]["vid"], dtype=str)

    leakage_info = verify_zero_group_leakage(grp_train, grp_val, grp_test)
    per_class_counts = _compute_split_class_counts(y_train, y_val, y_test, classes)

    logger.info(
        "Grouped Video Split — Train: %d clips (%d scenes) | Val: %d clips (%d scenes) | Test: %d clips (%d scenes)",
        len(X_train), leakage_info["train_unique_groups"],
        len(X_val), leakage_info["val_unique_groups"],
        len(X_test), leakage_info["test_unique_groups"],
    )

    train_ds = build_tf_dataset(
        X_train, y_train, batch_size, augment=augment, shuffle=True, seed=random_seed, is_video=True
    )
    val_ds = build_tf_dataset(
        X_val, y_val, batch_size, augment=False, shuffle=False, is_video=True
    )
    test_ds = build_tf_dataset(
        X_test, y_test, batch_size, augment=False, shuffle=False, is_video=True
    )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_groups": grp_train,
        "val_groups": grp_val,
        "test_groups": grp_test,
        "test_video_ids": vid_test,
        "leakage_check": leakage_info,
        "per_class_counts": per_class_counts,
        "class_names": classes,
        "num_classes": num_classes,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }
