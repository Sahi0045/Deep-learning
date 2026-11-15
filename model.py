"""
model.py - Image and Spatio-Temporal Video Architectures for Violence/Action Detection.

Provides four architectures for rigorous experimental comparison:
  1. custom_cnn      : 4-block 2D Residual CNN baseline (single frame input)
  2. efficientnet_v2 : EfficientNetV2B0 transfer learning model (single frame input)
  3. cnn_lstm        : Spatio-temporal TimeDistributed CNN + Bidirectional LSTM (video clip input)
  4. conv3d          : Volumetric 3D Convolutional Neural Network (video clip input)
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import cfg

logger = logging.getLogger(__name__)

# Model families that consume spatio-temporal video clips (T, H, W, C).
VIDEO_TYPES = {"cnn_lstm", "conv3d"}


def _resolve_imagenet_weights() -> Optional[str]:
    """Use locally cached EfficientNetV2B0 ImageNet weights if present, else 'imagenet'."""
    local_path = cfg.model.pretrained_weights_path
    if local_path and os.path.exists(local_path):
        return local_path
    return "imagenet"


# ---------------------------------------------------------------------------
# 1. Custom 2D Residual CNN Baseline
# ---------------------------------------------------------------------------

def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    use_batch_norm: bool,
    dropout_rate: float,
    l2_lambda: float,
    block_idx: int,
    first_stride: int = 1,
) -> tf.Tensor:
    """2D Residual Convolutional Block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU + Skip -> Pool -> Dropout."""
    name = f"block{block_idx}"
    reg = regularizers.l2(l2_lambda)

    x = layers.Conv2D(
        filters, kernel_size, strides=first_stride, padding="same",
        kernel_regularizer=reg, name=f"{name}_conv1",
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    residual = x
    x = layers.Conv2D(
        filters, kernel_size, padding="same",
        kernel_regularizer=reg, name=f"{name}_conv2",
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)

    x = layers.Add(name=f"{name}_skip")([x, residual])
    x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(x)
    x = layers.SpatialDropout2D(dropout_rate, name=f"{name}_dropout")(x)
    return x


def build_custom_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    conv_layers: Optional[List[Tuple[int, int]]] = None,
    fc_units: Optional[List[int]] = None,
    use_batch_norm: bool = True,
    conv_dropout: float = 0.20,
    fc_dropout: float = 0.40,
    l2_lambda: float = 1e-4,
) -> keras.Model:
    """Build the custom 4-block 2D Residual CNN baseline."""
    mc = cfg.model
    conv_layers = conv_layers or mc.conv_layers
    fc_units = fc_units or mc.fc_units
    reg = regularizers.l2(l2_lambda)

    inputs = keras.Input(shape=input_shape, name="image_input")
    x = inputs

    for idx, (filters, kernel_size) in enumerate(conv_layers):
        x = conv_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=conv_dropout,
            l2_lambda=l2_lambda,
            block_idx=idx + 1,
            first_stride=2 if idx == 0 else 1,
        )

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    for fc_idx, units in enumerate(fc_units):
        x = layers.Dense(units, kernel_regularizer=reg, name=f"fc{fc_idx + 1}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"fc{fc_idx + 1}_bn")(x)
        x = layers.Activation("relu", name=f"fc{fc_idx + 1}_relu")(x)
        x = layers.Dropout(fc_dropout, name=f"fc{fc_idx + 1}_dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="CustomResCNN2D")


# ---------------------------------------------------------------------------
# 2. EfficientNetV2B0 Transfer Learning Model
# ---------------------------------------------------------------------------

def build_transfer_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    fc_dropout: float = 0.40,
    l2_lambda: float = 1e-4,
    fine_tune_at: Optional[int] = None,
    weights: Optional[str] = None,
    head_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> keras.Model:
    """Build EfficientNetV2B0 transfer-learning model with classification head."""
    reg = regularizers.l2(l2_lambda)
    fine_tune_at = fine_tune_at if fine_tune_at is not None else cfg.model.fine_tune_at
    resolved_weights = weights if weights is not None else _resolve_imagenet_weights()

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=resolved_weights,
        input_shape=input_shape,
    )
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape, name="image_input")
    x = layers.Rescaling(255.0, name="rescale_255")(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(fc_dropout, name="head_dropout")(x)

    if head_weights is not None:
        w_init, b_init = head_weights
        outputs = layers.Dense(
            num_classes,
            activation="softmax",
            kernel_initializer=keras.initializers.Constant(w_init),
            bias_initializer=keras.initializers.Constant(b_init),
            kernel_regularizer=regularizers.l2(l2_lambda),
            name="predictions",
        )(x)
    else:
        x = layers.Dense(256, kernel_regularizer=reg, name="fc1")(x)
        x = layers.BatchNormalization(name="fc1_bn")(x)
        x = layers.Activation("relu", name="fc1_relu")(x)
        x = layers.Dropout(fc_dropout, name="fc1_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetV2_Transfer")
    model._backbone = backbone  # type: ignore[attr-defined]
    model._fine_tune_at = fine_tune_at  # type: ignore[attr-defined]
    return model


def unfreeze_backbone(model: keras.Model, fine_tune_at: Optional[int] = None) -> None:
    """Unfreeze top EfficientNetV2B0 layers (keeping BatchNorm frozen) for fine-tuning."""
    backbone = getattr(model, "_backbone", None)
    if backbone is None:
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                backbone = layer
                break
    if backbone is None:
        logger.warning("No backbone found on model to unfreeze.")
        return

    fine_tune_at = fine_tune_at or getattr(model, "_fine_tune_at", cfg.model.fine_tune_at)
    backbone.trainable = True
    for idx, layer in enumerate(backbone.layers):
        if idx < fine_tune_at or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    logger.info(
        "Backbone unfrozen from layer %d / %d (BatchNorm kept frozen).",
        fine_tune_at, len(backbone.layers),
    )


# ---------------------------------------------------------------------------
# 2b. Feature-caching helpers (fast frozen-backbone transfer learning)
# ---------------------------------------------------------------------------

def build_feature_extractor(
    image_size: int,
    num_channels: int = 3,
    weights: Optional[str] = None,
) -> Tuple[keras.Model, int]:
    """
    Build a frozen EfficientNetV2B0 feature extractor that maps a [0, 1] image
    to a pooled feature vector. Used to cache backbone features once so that
    head training avoids recomputing the backbone every epoch.

    Returns:
        (extractor_model, feature_dim) where feature_dim == 1280 for V2B0.
    """
    resolved_weights = weights if weights is not None else _resolve_imagenet_weights()
    inputs = keras.Input(shape=(image_size, image_size, num_channels), name="feat_input")
    scaled = layers.Rescaling(255.0, name="feat_rescale")(inputs)
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=resolved_weights,
        input_shape=(image_size, image_size, num_channels),
        pooling="avg",
    )
    backbone.trainable = False
    feats = backbone(scaled, training=False)
    extractor = keras.Model(inputs, feats, name="effnetv2_feature_extractor")
    return extractor, int(feats.shape[-1])


def build_classification_head(
    feature_dim: int,
    num_classes: int,
    fc_dropout: float = 0.40,
    l2_lambda: float = 1e-4,
) -> keras.Model:
    """
    Build a lightweight classification head on cached pooled features:
    Input(feature_dim) -> Dropout -> Dense(num_classes, softmax).
    After training, pass its final-layer weights to build_transfer_model(...,
    head_weights=...) to reassemble a self-contained end-to-end Keras model.
    """
    reg = regularizers.l2(l2_lambda)
    inputs = keras.Input(shape=(feature_dim,), name="head_input")
    x = layers.Dropout(fc_dropout, name="head_dropout")(inputs)
    outputs = layers.Dense(num_classes, activation="softmax",
                            kernel_regularizer=reg, name="predictions")(x)
    return keras.Model(inputs, outputs, name="effnetv2_head")


def reassemble_transfer_model(
    image_size: int,
    num_channels: int,
    num_classes: int,
    head_weights: Tuple[np.ndarray, np.ndarray],
    fc_dropout: float = 0.40,
    l2_lambda: float = 1e-4,
) -> keras.Model:
    """
    Reassemble a full, self-contained end-to-end transfer model whose final
    Dense layer is initialised from a trained head's weights.

    ``head_weights`` is the (kernel, bias) tuple from a build_classification_head
    output layer. The resulting model reproduces the head's inference exactly
    (Dropout is inactive at inference) while remaining a single loadable .keras.
    """
    return build_transfer_model(
        input_shape=(image_size, image_size, num_channels),
        num_classes=num_classes,
        fc_dropout=fc_dropout,
        l2_lambda=l2_lambda,
        head_weights=head_weights,
    )


# ---------------------------------------------------------------------------
# 3. Spatio-Temporal CNN-LSTM Video Model
# ---------------------------------------------------------------------------

def build_cnn_lstm_model(
    sequence_shape: Tuple[int, int, int, int],
    num_classes: int,
    lstm_units: Optional[int] = None,
    fc_dropout: float = 0.40,
    l2_lambda: float = 1e-4,
    use_pretrained_backbone: bool = True,
    weights: Optional[str] = None,
) -> keras.Model:
    """
    Build a Spatio-Temporal CNN-LSTM model for video action classification.

    Architecture:
      Input (T, H, W, 3)
        -> TimeDistributed Spatial Encoder (Pretrained EfficientNetV2B0 or Conv2D stack)
        -> Residual Static Appearance Branch + Bidirectional LSTM Temporal Dynamics Branch
        -> Softmax(num_classes)
    """
    seq_len, h, w, c = sequence_shape
    lstm_units = lstm_units or cfg.model.lstm_units
    reg = regularizers.l2(l2_lambda)

    inputs = keras.Input(shape=sequence_shape, name="video_input")

    if use_pretrained_backbone:
        resolved_weights = weights if weights is not None else _resolve_imagenet_weights()
        frame_inp = keras.Input(shape=(h, w, c), name="frame_in")
        scaled = layers.Rescaling(255.0, name="rescale_255")(frame_inp)
        base_cnn = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=resolved_weights,
            input_shape=(h, w, c),
            pooling="avg",
        )
        base_cnn.trainable = False
        feat_out = base_cnn(scaled, training=False)
        spatial_encoder = keras.Model(frame_inp, feat_out, name="spatial_encoder")
        seq_feats = layers.TimeDistributed(spatial_encoder, name="td_encoder")(inputs)
    else:
        x = layers.TimeDistributed(
            layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"), name="td_conv1"
        )(inputs)
        x = layers.TimeDistributed(layers.MaxPooling2D(2), name="td_pool1")(x)
        x = layers.TimeDistributed(
            layers.Conv2D(64, 3, padding="same", activation="relu"), name="td_conv2"
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2), name="td_pool2")(x)
        x = layers.TimeDistributed(
            layers.Conv2D(128, 3, padding="same", activation="relu"), name="td_conv3"
        )(x)
        seq_feats = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="td_gap")(x)

    frame_logits = layers.TimeDistributed(
        layers.Dense(num_classes, kernel_regularizer=regularizers.l2(1e-2)),
        name="td_appearance_logits",
    )(seq_feats)
    static_logits = layers.GlobalAveragePooling1D(name="avg_appearance_logits")(frame_logits)

    proj = layers.TimeDistributed(
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        name="td_proj32",
    )(layers.Dropout(fc_dropout, name="seq_dropout")(seq_feats))
    lstm_in = layers.Concatenate(name="lstm_concat")([proj, frame_logits])
    bilstm = layers.Bidirectional(
        layers.LSTM(max(16, lstm_units // 4), dropout=0.30, kernel_regularizer=reg),
        name="bilstm_temporal",
    )(lstm_in)
    delta_logits = layers.Dense(
        num_classes,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(1e-2),
        name="temporal_delta_logits",
    )(bilstm)

    combined_logits = layers.Add(name="spatiotemporal_logits")([static_logits, delta_logits])
    outputs = layers.Activation("softmax", name="predictions")(combined_logits)

    return keras.Model(inputs=inputs, outputs=outputs, name="Video_CNN_BiLSTM")


# ---------------------------------------------------------------------------
# 4. Volumetric 3D CNN Video Model
# ---------------------------------------------------------------------------

def build_3d_cnn_model(
    sequence_shape: Tuple[int, int, int, int],
    num_classes: int,
    filters_list: Optional[List[int]] = None,
    fc_dropout: float = 0.40,
    l2_lambda: float = 1e-4,
) -> keras.Model:
    """
    Build a 3D Convolutional Neural Network (3D CNN) for spatio-temporal video volumes.

    Input shape: (T, H, W, 3)
    """
    filters_list = filters_list or cfg.model.conv3d_filters
    reg = regularizers.l2(l2_lambda)

    inputs = keras.Input(shape=sequence_shape, name="video_input")
    x = inputs

    for idx, filters in enumerate(filters_list):
        bname = f"c3d_block{idx + 1}"
        spatial_stride = 2 if idx == 0 else 1
        x = layers.Conv3D(
            filters,
            kernel_size=(3, 3, 3),
            strides=(1, spatial_stride, spatial_stride),
            padding="same",
            kernel_regularizer=reg,
            name=f"{bname}_conv1",
        )(x)
        x = layers.BatchNormalization(name=f"{bname}_bn1")(x)
        x = layers.Activation("relu", name=f"{bname}_relu1")(x)

        residual = x
        x = layers.Conv3D(
            filters, kernel_size=(3, 3, 3), padding="same",
            kernel_regularizer=reg, name=f"{bname}_conv2",
        )(x)
        x = layers.BatchNormalization(name=f"{bname}_bn2")(x)
        x = layers.Activation("relu", name=f"{bname}_relu2")(x)
        x = layers.Add(name=f"{bname}_skip")([x, residual])

        t_pool = 2 if idx > 0 else 1
        x = layers.MaxPooling3D(pool_size=(t_pool, 2, 2), name=f"{bname}_pool")(x)
        x = layers.SpatialDropout3D(0.20, name=f"{bname}_dropout")(x)

    x = layers.GlobalAveragePooling3D(name="gap3d")(x)
    x = layers.Dense(128, kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(fc_dropout, name="fc1_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="Video_Res3DCNN")


# ---------------------------------------------------------------------------
# Factory & Compilation
# ---------------------------------------------------------------------------

def build_model_by_name(
    model_type: str,
    num_classes: Optional[int] = None,
    image_size: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> keras.Model:
    """Build any of the supported architectures by name."""
    num_classes = num_classes or len(cfg.data.classes)
    image_size = image_size or cfg.data.image_size
    seq_len = seq_len or cfg.data.sequence_length
    c = cfg.data.num_channels

    if model_type == "custom_cnn":
        return build_custom_cnn(
            input_shape=(image_size, image_size, c),
            num_classes=num_classes,
            conv_dropout=cfg.model.conv_dropout,
            fc_dropout=cfg.model.fc_dropout,
            l2_lambda=cfg.model.l2_lambda,
        )
    if model_type in ("efficientnet_v2", "transfer"):
        return build_transfer_model(
            input_shape=(image_size, image_size, c),
            num_classes=num_classes,
            fc_dropout=cfg.model.fc_dropout,
            l2_lambda=cfg.model.l2_lambda,
        )
    if model_type == "cnn_lstm":
        return build_cnn_lstm_model(
            sequence_shape=(seq_len, image_size, image_size, c),
            num_classes=num_classes,
            lstm_units=cfg.model.lstm_units,
            fc_dropout=cfg.model.fc_dropout,
            l2_lambda=cfg.model.l2_lambda,
        )
    if model_type in ("conv3d", "3d_cnn"):
        return build_3d_cnn_model(
            sequence_shape=(seq_len, image_size, image_size, c),
            num_classes=num_classes,
            filters_list=cfg.model.conv3d_filters,
            fc_dropout=cfg.model.fc_dropout,
            l2_lambda=cfg.model.l2_lambda,
        )
    raise ValueError(
        f"Unknown model_type '{model_type}'. Choose from: "
        "'custom_cnn', 'efficientnet_v2', 'cnn_lstm', 'conv3d'."
    )


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
) -> keras.Model:
    """Compile a Keras model with standard classification metrics."""
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: '{optimizer}'. Choose 'adam', 'sgd', or 'rmsprop'.")

    num_classes = model.output_shape[-1]
    top_k = min(2 if num_classes <= 3 else 3, num_classes)

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=top_k, name="top_k_accuracy"),
            keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    return model


if __name__ == "__main__":
    for mtype in ["custom_cnn", "efficientnet_v2", "cnn_lstm", "conv3d"]:
        m = build_model_by_name(mtype)
        m = compile_model(m, learning_rate=cfg.training.learning_rate)
        print(f"{mtype}: {m.name} — {m.count_params():,} params")
