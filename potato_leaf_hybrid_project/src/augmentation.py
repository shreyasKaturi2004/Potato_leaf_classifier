from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential, layers

from src.config import ProjectConfig


@tf.keras.utils.register_keras_serializable(package="PotatoLeaf")
class RandomBrightness(layers.Layer):
    """Simple custom brightness augmentation that only runs during training."""

    def __init__(self, factor: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"factor": self.factor})
        return config

    def _augment(self, images: tf.Tensor) -> tf.Tensor:
        images = tf.image.random_brightness(images, max_delta=self.factor)
        return tf.clip_by_value(images, 0.0, 1.0)

    def call(self, images: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        images = tf.cast(images, tf.float32)

        if training is None:
            return images

        if isinstance(training, bool):
            return self._augment(images) if training else images

        return tf.cond(
            tf.cast(training, tf.bool),
            lambda: self._augment(images),
            lambda: tf.identity(images),
        )


def build_augmentation_pipeline(config: ProjectConfig) -> Sequential:
    """Creates the augmentation pipeline used inside the models."""

    zoom_factor = config.model.random_zoom

    return Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", name="random_flip"),
            layers.RandomRotation(
                factor=config.model.random_rotation,
                fill_mode="nearest",
                name="random_rotation",
            ),
            layers.RandomZoom(
                height_factor=(-zoom_factor, zoom_factor),
                width_factor=(-zoom_factor, zoom_factor),
                fill_mode="nearest",
                name="random_zoom",
            ),
            RandomBrightness(config.model.random_brightness, name="random_brightness"),
            layers.RandomContrast(config.model.random_contrast, name="random_contrast"),
        ],
        name="augmentation_pipeline",
    )
