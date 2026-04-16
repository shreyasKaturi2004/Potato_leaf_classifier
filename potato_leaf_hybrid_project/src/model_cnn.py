from __future__ import annotations

import warnings

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from src.augmentation import build_augmentation_pipeline
from src.config import ProjectConfig


def _conv_bn_act(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    activation: str,
    l2_reg: float,
    name: str,
    strides: int = 1,
) -> tf.Tensor:
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation(activation, name=f"{name}_act")(x)
    return x


def _multi_scale_block(
    x: tf.Tensor,
    filters: int,
    activation: str,
    dropout_rate: float,
    l2_reg: float,
    block_id: int,
) -> tf.Tensor:
    """
    A slightly more original CNN block that mixes 3x3 and 5x5 receptive fields.
    This helps the network see both fine disease spots and larger damaged regions.
    """

    shortcut = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"block_{block_id}_shortcut_conv",
    )(x)
    shortcut = layers.BatchNormalization(name=f"block_{block_id}_shortcut_bn")(shortcut)

    branch_3x3 = _conv_bn_act(
        x,
        filters=filters // 2,
        kernel_size=3,
        activation=activation,
        l2_reg=l2_reg,
        name=f"block_{block_id}_branch3x3_a",
    )
    branch_3x3 = _conv_bn_act(
        branch_3x3,
        filters=filters // 2,
        kernel_size=3,
        activation=activation,
        l2_reg=l2_reg,
        name=f"block_{block_id}_branch3x3_b",
    )

    branch_5x5 = _conv_bn_act(
        x,
        filters=filters // 2,
        kernel_size=5,
        activation=activation,
        l2_reg=l2_reg,
        name=f"block_{block_id}_branch5x5_a",
    )
    branch_5x5 = _conv_bn_act(
        branch_5x5,
        filters=filters // 2,
        kernel_size=3,
        activation=activation,
        l2_reg=l2_reg,
        name=f"block_{block_id}_branch5x5_b",
    )

    merged = layers.Concatenate(name=f"block_{block_id}_concat")([branch_3x3, branch_5x5])
    merged = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"block_{block_id}_fusion_conv",
    )(merged)
    merged = layers.BatchNormalization(name=f"block_{block_id}_fusion_bn")(merged)
    merged = layers.Add(name=f"block_{block_id}_add")([merged, shortcut])
    merged = layers.Activation(activation, name=f"block_{block_id}_out_act")(merged)
    merged = layers.MaxPooling2D(pool_size=(2, 2), name=f"block_{block_id}_pool")(merged)
    merged = layers.Dropout(dropout_rate, name=f"block_{block_id}_dropout")(merged)
    return merged


def build_feature_extractor(config: ProjectConfig) -> models.Model:
    """Builds the shared classical feature extractor."""

    inputs = layers.Input(
        shape=(*config.dataset.image_size, config.dataset.num_channels),
        name="image_input",
    )
    x = build_augmentation_pipeline(config)(inputs)
    x = _conv_bn_act(
        x,
        filters=config.model.base_filters[0],
        kernel_size=3,
        activation=config.model.activation,
        l2_reg=config.model.l2_reg,
        name="stem",
    )

    for block_id, filters in enumerate(config.model.base_filters, start=1):
        x = _multi_scale_block(
            x,
            filters=filters,
            activation=config.model.activation,
            dropout_rate=config.model.dropout_rate,
            l2_reg=config.model.l2_reg,
            block_id=block_id,
        )

    x = _conv_bn_act(
        x,
        filters=config.model.base_filters[-1] * 2,
        kernel_size=3,
        activation=config.model.activation,
        l2_reg=config.model.l2_reg,
        name="refinement",
    )
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dense(
        config.model.dense_units,
        activation=config.model.activation,
        kernel_regularizer=regularizers.l2(config.model.l2_reg),
        name="feature_dense",
    )(x)
    x = layers.Dropout(config.model.dropout_rate, name="feature_dropout")(x)

    return models.Model(inputs=inputs, outputs=x, name="custom_feature_extractor")


def build_custom_cnn_model(config: ProjectConfig) -> models.Model:
    """Baseline model: modified custom CNN for 3-class potato leaf classification."""

    feature_extractor = build_feature_extractor(config)
    outputs = layers.Dense(
        config.dataset.num_classes,
        activation="softmax",
        name="class_probabilities",
    )(feature_extractor.output)

    return models.Model(
        inputs=feature_extractor.input,
        outputs=outputs,
        name="custom_potato_cnn",
    )


def build_transfer_learning_model(config: ProjectConfig) -> models.Model:
    """
    Optional baseline using ResNet50.
    This model is heavier and may try to download ImageNet weights the first time.
    """

    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input

    inputs = layers.Input(
        shape=(*config.dataset.image_size, config.dataset.num_channels),
        name="image_input",
    )
    x = build_augmentation_pipeline(config)(inputs)
    x = layers.Lambda(lambda tensor: preprocess_input(tensor * 255.0), name="resnet_preprocess")(x)

    weights_choice = "imagenet" if config.model.use_imagenet_weights else None
    try:
        backbone = ResNet50(
            include_top=False,
            weights=weights_choice,
            input_shape=(*config.dataset.image_size, config.dataset.num_channels),
        )
    except Exception as exc:
        warnings.warn(
            "ImageNet weights could not be loaded. Falling back to randomly initialized ResNet50. "
            f"Reason: {exc}"
        )
        backbone = ResNet50(
            include_top=False,
            weights=None,
            input_shape=(*config.dataset.image_size, config.dataset.num_channels),
        )

    backbone.trainable = not config.model.freeze_backbone
    x = backbone(x, training=False if config.model.freeze_backbone else None)
    x = layers.GlobalAveragePooling2D(name="transfer_gap")(x)
    x = layers.Dense(
        config.model.transfer_dense_units,
        activation=config.model.activation,
        name="transfer_dense",
    )(x)
    x = layers.Dropout(config.model.dropout_rate, name="transfer_dropout")(x)
    outputs = layers.Dense(
        config.dataset.num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    return models.Model(inputs=inputs, outputs=outputs, name="transfer_resnet50_baseline")
