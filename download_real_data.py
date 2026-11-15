"""
download_real_data.py - Downloads and prepares the verified AIRTLab Violence Detection Dataset.

Dataset Provenance:
  - Title:   A dataset for automatic violence detection in videos
  - Authors: Bianculli et al. (Data in Brief, 2020) & Sernani et al. (IEEE Access, 2021)
  - DOI:     https://doi.org/10.1016/j.dib.2020.106587
  - License: Creative Commons Attribution 4.0 International (CC-BY 4.0)
  - Source:  https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos

What this script does:
  1. Downloads all 350 manually annotated real-world MP4 video clips (175 unique scenes
     recorded simultaneously across cam1 and cam2) and the 3 official CSV annotation files.
  2. Maps each scene's human-labeled action tags into 3 verified classes:
       - nonviolent      (60 scenes / 120 videos: hugs, greetings, high-fives, hand gestures)
       - unarmed_assault (55 scenes / 110 videos: slapping, punching, kicking, pushing, choking)
       - armed_assault   (60 scenes / 120 videos: stabbing, clubbing, gunshot threats)
  3. Performs stratified scene-grouped splitting (70% train, 15% val, 15% test by scene_id)
     so that neither frames from the same clip nor alternate camera angles (cam1/cam2) of
     the same physical event ever cross split boundaries.
  4. Extracts evenly spaced frames per video clip for both 2D frame models and 3D/recurrent
     video models, writes datasets/airtlab_violence/manifest.csv, and syncs training_data/.
"""

import argparse
import csv
import logging
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import cv2
import numpy as np
import requests
from sklearn.model_selection import train_test_split

from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = (
    "https://raw.githubusercontent.com/airtlab/"
    "A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/violence-detection-dataset"
)
CSV_FILES = [
    "action-class-occurrences.csv",
    "nonviolent-action-classes.csv",
    "violent-action-classes.csv",
]
WEAPON_ACTIONS = {"club", "stab", "gunshot"}


def parse_action_csv(path: str) -> Dict[str, List[str]]:
    """Parse official semicolon-delimited action annotation CSV."""
    mapping: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(";")]
        if len(parts) >= 2:
            fname = parts[0]
            acts = [a.strip() for a in parts[1].split(",") if a.strip()]
            mapping[fname] = acts
    return mapping


def _download_file(task: Tuple[str, str]) -> Tuple[str, int, bool]:
    url, dst = task
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
        return dst, os.path.getsize(dst), True
    for _ in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 10_000:
                with open(dst, "wb") as f:
                    f.write(r.content)
                return dst, len(r.content), True
        except Exception:
            time.sleep(1.0)
    return dst, 0, False


def download_airtlab_dataset(dataset_dir: str, threads: int = 16) -> None:
    """Download CSV annotations and all 350 MP4 video clips if not already present."""
    os.makedirs(dataset_dir, exist_ok=True)

    for csv_name in CSV_FILES:
        csv_path = os.path.join(dataset_dir, csv_name)
        if not os.path.exists(csv_path):
            r = requests.get(f"{BASE_URL}/{csv_name}", timeout=30)
            r.raise_for_status()
            with open(csv_path, "wb") as f:
                f.write(r.content)

    tasks: List[Tuple[str, str]] = []
    for split_dir, count in [("non-violent", 60), ("violent", 115)]:
        for cam in ["cam1", "cam2"]:
            os.makedirs(os.path.join(dataset_dir, split_dir, cam), exist_ok=True)
            for idx in range(1, count + 1):
                rel = f"{split_dir}/{cam}/{idx}.mp4"
                tasks.append((f"{BASE_URL}/{rel}", os.path.join(dataset_dir, rel)))

    missing = [t for t in tasks if not (os.path.exists(t[1]) and os.path.getsize(t[1]) > 10_000)]
    if not missing:
        logger.info("All 350 verified MP4 videos are already present in %s.", dataset_dir)
        return

    logger.info("Downloading %d missing MP4 videos with %d threads...", len(missing), threads)
    failed = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(_download_file, t) for t in missing]
        for fut in as_completed(futures):
            dst, _, ok = fut.result()
            if not ok:
                failed.append(dst)
    if failed:
        raise RuntimeError(f"Failed to download {len(failed)} video files: {failed[:5]}")
    logger.info("Successfully downloaded all 350 verified MP4 video clips.")


def prepare_manifest_and_frames(
    dataset_dir: str,
    train_path: str,
    seq_len: int = 8,
    image_size: int = 128,
    seed: int = 42,
) -> str:
    """
    Extract frame sequences, assign scene-grouped train/val/test splits,
    write manifest.csv, and populate training_data/ with verified real frames.
    """
    frames_dir = os.path.join(dataset_dir, "frames")
    nv_map = parse_action_csv(os.path.join(dataset_dir, "nonviolent-action-classes.csv"))
    v_map = parse_action_csv(os.path.join(dataset_dir, "violent-action-classes.csv"))

    scenes = []
    for idx in range(1, 61):
        fname = f"{idx}.mp4"
        acts = nv_map.get(fname, ["nonviolent"])
        scenes.append({
            "scene_id": f"nonviolent_scene_{idx:03d}",
            "folder": "non-violent",
            "fname": fname,
            "class_name": "nonviolent",
            "class_id": 0,
            "binary_label": "nonviolent",
            "raw_actions": ",".join(acts),
        })

    for idx in range(1, 116):
        fname = f"{idx}.mp4"
        acts = v_map.get(fname, ["fight"])
        if any(a in WEAPON_ACTIONS for a in acts):
            cls_name, cls_id = "armed_assault", 2
        else:
            cls_name, cls_id = "unarmed_assault", 1
        scenes.append({
            "scene_id": f"violent_scene_{idx:03d}",
            "folder": "violent",
            "fname": fname,
            "class_name": cls_name,
            "class_id": cls_id,
            "binary_label": "violent",
            "raw_actions": ",".join(acts),
        })

    # Scene-grouped stratified split (70% train, 15% val, 15% test)
    scene_labels = [s["class_id"] for s in scenes]
    train_scenes, temp_scenes = train_test_split(
        scenes, test_size=0.30, stratify=scene_labels, random_state=seed
    )
    temp_labels = [s["class_id"] for s in temp_scenes]
    val_scenes, test_scenes = train_test_split(
        temp_scenes, test_size=0.50, stratify=temp_labels, random_state=seed
    )

    split_map = {}
    for s in train_scenes:
        split_map[s["scene_id"]] = "train"
    for s in val_scenes:
        split_map[s["scene_id"]] = "val"
    for s in test_scenes:
        split_map[s["scene_id"]] = "test"

    manifest_rows = []
    for s in scenes:
        split = split_map[s["scene_id"]]
        cls_name = s["class_name"]
        for cam in ["cam1", "cam2"]:
            vid_rel = os.path.join(dataset_dir, s["folder"], cam, s["fname"])
            video_id = f"{s['scene_id']}_{cam}"
            out_dir = os.path.join(frames_dir, cls_name, video_id)
            os.makedirs(out_dir, exist_ok=True)

            cap = cv2.VideoCapture(vid_rel)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            start_f = max(0, int(total_frames * 0.10))
            end_f = max(start_f + seq_len, int(total_frames * 0.90))
            indices = np.linspace(start_f, min(end_f, max(total_frames - 1, 0)), seq_len, dtype=int)

            frame_paths = []
            for f_i, frame_idx in enumerate(indices):
                f_path = os.path.join(out_dir, f"frame_{f_i:02d}.jpg")
                if not os.path.exists(f_path):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    frame_resized = cv2.resize(
                        frame, (image_size, image_size), interpolation=cv2.INTER_AREA
                    )
                    cv2.imwrite(f_path, frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                frame_paths.append(f_path)
            cap.release()

            manifest_rows.append({
                "video_id": video_id,
                "scene_id": s["scene_id"],
                "camera_id": cam,
                "video_path": vid_rel,
                "frames_dir": out_dir,
                "frame_paths": ";".join(frame_paths),
                "num_frames": len(frame_paths),
                "total_video_frames": total_frames,
                "fps": round(fps, 2),
                "class_name": cls_name,
                "class_id": s["class_id"],
                "binary_label": s["binary_label"],
                "raw_actions": s["raw_actions"],
                "split": split,
            })

    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Sync training_data/ directory with verified frames only (no synthetic images)
    shutil.rmtree(train_path, ignore_errors=True)
    os.makedirs(train_path, exist_ok=True)
    for r in manifest_rows:
        cls_dir = os.path.join(train_path, r["class_name"])
        os.makedirs(cls_dir, exist_ok=True)
        for fpath in r["frame_paths"].split(";"):
            fname = f"{r['video_id']}_{os.path.basename(fpath)}"
            shutil.copy2(fpath, os.path.join(cls_dir, fname))

    for sp in ["train", "val", "test"]:
        c = Counter(r["class_name"] for r in manifest_rows if r["split"] == sp)
        logger.info("Split %-5s — Videos: %s | Frames: %d", sp, dict(c), sum(c.values()) * seq_len)

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the verified AIRTLab Violence Detection Dataset."
    )
    parser.add_argument("--dataset_dir", default=cfg.data.dataset_dir)
    parser.add_argument("--train_path", default=cfg.data.train_path)
    parser.add_argument("--seq_len", type=int, default=cfg.data.sequence_length)
    parser.add_argument("--image_size", type=int, default=cfg.data.image_size)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    download_airtlab_dataset(args.dataset_dir, threads=args.threads)
    prepare_manifest_and_frames(
        dataset_dir=args.dataset_dir,
        train_path=args.train_path,
        seq_len=args.seq_len,
        image_size=args.image_size,
        seed=cfg.data.random_seed,
    )


if __name__ == "__main__":
    main()
