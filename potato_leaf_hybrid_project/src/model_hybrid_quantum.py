from __future__ import annotations

import numpy as np
import pennylane as qml
import tensorflow as tf
import warnings
from tensorflow.keras import layers, models

from src.config import ProjectConfig
from src.model_cnn import build_feature_extractor

warnings.filterwarnings(
    "ignore",
    message="Support for the TensorFlow interface is deprecated.*",
    category=getattr(qml, "PennyLaneDeprecationWarning", Warning),
)


@tf.keras.utils.register_keras_serializable(package="PotatoLeaf")
class QuantumCircuitLayer(layers.Layer):
    """Lightweight TensorFlow-compatible PennyLane layer for batched 4-qubit inference."""

    def __init__(self, num_qubits: int, circuit_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.num_qubits = num_qubits
        self.circuit_layers = circuit_layers
        self.device = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.device, interface="tf", diff_method="backprop")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
            qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wire)) for wire in range(num_qubits)]

        self.circuit = circuit

    def build(self, input_shape):
        self.quantum_weights = self.add_weight(
            name="quantum_weights",
            shape=(self.circuit_layers, self.num_qubits, 3),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        quantum_weights = tf.convert_to_tensor(self.quantum_weights, dtype=tf.float32)
        outputs = [
            tf.cast(
                tf.stack(self.circuit(sample, quantum_weights)),
                tf.float32,
            )
            for sample in tf.unstack(inputs, axis=0)
        ]
        return tf.stack(outputs, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_qubits)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_qubits": self.num_qubits,
                "circuit_layers": self.circuit_layers,
            }
        )
        return config


def build_hybrid_quantum_model(
    config: ProjectConfig,
    feature_extractor: models.Model | None = None,
) -> models.Model:
    feature_extractor = feature_extractor or build_feature_extractor(config)
    feature_extractor.trainable = not config.quantum.freeze_feature_extractor

    inputs = layers.Input(
        shape=(*config.dataset.image_size, config.dataset.num_channels),
        name="image_input",
    )
    x = feature_extractor(inputs)
    x = layers.Dense(
        config.quantum.pre_quantum_units,
        activation=config.model.activation,
        name="pre_quantum_dense",
    )(x)
    x = layers.Dropout(
        config.model.dropout_rate * 0.5,
        name="pre_quantum_dropout",
    )(x)
    x = layers.Dense(
        config.quantum.num_qubits,
        activation="linear",
        name="quantum_projection",
    )(x)
    x = layers.Lambda(lambda tensor: tf.math.tanh(tensor) * np.pi, name="quantum_angle_scale")(x)
    x = QuantumCircuitLayer(
        num_qubits=config.quantum.num_qubits,
        circuit_layers=config.quantum.circuit_layers,
        name="quantum_layer",
    )(x)
    x = layers.Dense(
        config.quantum.post_quantum_units,
        activation=config.model.activation,
        name="post_quantum_dense",
    )(x)
    x = layers.Dropout(
        config.model.dropout_rate,
        name="post_quantum_dropout",
    )(x)
    outputs = layers.Dense(
        config.dataset.num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    return models.Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_quantum_model")
