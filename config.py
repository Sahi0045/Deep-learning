"""
config.py - Centralized configuration management backed by experiment_config.yaml.

All hyperparameters, dataset paths, split strategies, and training settings are
defined here and synchronized with experiment_config.yaml for reproducibility.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import yaml


@dataclass
class DataConfig:
    """Dataset and grouped-split configuration."""

    dataset_name: str = "AIRTLab Violence Detection Dataset"
    doi: str = "10.1016/j.dib.2020.106587"
    license: str = "CC-BY-4.0"
    source_url: str = "https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos"

    dataset_dir: str = "datasets/airtlab_violence"
    manifest_path: str = "datasets/airtlab_violence/manifest.csv"
    train_path: str = "training_data"
    model_dir: str = "checkpoints"
    results_dir: str = "results"
    logs_dir: str = "logs"

    # Image & video sequence properties
    image_size: int = 128
    num_channels: int = 3
    sequence_length: int = 8

    # Grouped splitting (by scene_id / video / source to prevent frame & multi-cam leakage)
    split_strategy: str = "grouped_by_scene"
    group_column: str = "scene_id"
    train_split: float = 0.70
    validation_split: float = 0.15
    test_split: float = 0.15

    # Reproducibility & data integrity
    random_seed: int = 42
    exclude_synthetic: bool = True

    # Verified action classes from manual annotations
    classes: List[str] = field(default_factory=lambda: [
        "nonviolent",
        "unarmed_assault",
        "armed_assault",
    ])


@dataclass
class ModelConfig:
    """Model architecture configuration (2D CNN, EfficientNetV2, CNN-LSTM, 3D CNN)."""

    conv_layers: List[Tuple[int, int]] = field(default_factory=lambda: [
        (32, 3),
        (64, 3),
        (128, 3),
        (256, 3),
    ])
    pool_size: Tuple[int, int] = (2, 2)
    conv_dropout: float = 0.20
    fc_dropout: float = 0.40
    fc_units: List[int] = field(default_factory=lambda: [256, 128])
    use_batch_norm: bool = True
    l2_lambda: float = 1e-4

    # Transfer learning & video architecture settings
    pretrained_weights_path: str = "checkpoints/pretrained/efficientnetv2-b0_notop.h5"
    fine_tune_at: int = 180
    lstm_units: int = 128
    conv3d_filters: List[int] = field(default_factory=lambda: [32, 64, 128])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 25
    video_epochs: int = 25
    batch_size: int = 32
    video_batch_size: int = 16
    learning_rate: float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience: int = 4
    min_lr: float = 1e-6
    early_stop_patience: int = 8
    optimizer: str = "adam"

    # Data augmentation (vertical_flip disabled for upright human action scenes)
    augmentation: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    zoom_range: float = 0.10

    # Checkpointing
    save_best_only: bool = True
    checkpoint_metric: str = "val_accuracy"


@dataclass
class Config:
    """Master configuration — aggregates sub-configs and loads from YAML if available."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    config_path: str = "experiment_config.yaml"

    def __post_init__(self) -> None:
        if os.path.exists(self.config_path):
            self.load_from_yaml(self.config_path)
        for directory in [
            self.data.model_dir,
            self.data.results_dir,
            self.data.logs_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

    def load_from_yaml(self, yaml_path: str) -> None:
        """Override default configuration values from a YAML experiment config file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        exp = raw.get("experiment", {})
        ds = raw.get("dataset", {})
        md = raw.get("model", {})
        tr = raw.get("training", {})

        if "random_seed" in exp:
            self.data.random_seed = int(exp["random_seed"])

        for key in [
            "dataset_dir", "manifest_path", "train_path", "model_dir",
            "results_dir", "logs_dir", "image_size", "num_channels",
            "sequence_length", "split_strategy", "group_column",
            "train_split", "validation_split", "test_split",
            "exclude_synthetic", "classes", "doi", "license", "source_url",
        ]:
            if key in ds:
                setattr(self.data, key, ds[key])
        if "name" in ds:
            self.data.dataset_name = ds["name"]

        if "conv_layers" in md:
            self.model.conv_layers = [tuple(x) for x in md["conv_layers"]]
        if "pool_size" in md:
            self.model.pool_size = tuple(md["pool_size"])
        for key in [
            "conv_dropout", "fc_dropout", "fc_units", "use_batch_norm",
            "l2_lambda", "pretrained_weights_path", "fine_tune_at",
            "lstm_units", "conv3d_filters",
        ]:
            if key in md:
                setattr(self.model, key, md[key])

        for key in [
            "epochs", "video_epochs", "batch_size", "video_batch_size",
            "learning_rate", "lr_decay_factor", "lr_patience", "min_lr",
            "early_stop_patience", "optimizer", "augmentation",
            "horizontal_flip", "vertical_flip", "zoom_range",
            "save_best_only", "checkpoint_metric",
        ]:
            if key in tr:
                setattr(self.training, key, tr[key])


def load_config(yaml_path: Optional[str] = None) -> Config:
    """Load configuration from an explicit YAML path or default experiment_config.yaml."""
    if yaml_path:
        return Config(config_path=yaml_path)
    return Config()


# Singleton instance
cfg = Config()
