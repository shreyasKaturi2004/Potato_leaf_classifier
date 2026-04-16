from __future__ import annotations

import sys


def main() -> int:
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"TensorFlow import failed: {exc}")
        return 1

    print(f"TensorFlow version: {tf.__version__}")
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"Physical GPUs: {physical_gpus}")
    print(f"Logical GPUs: {logical_gpus}")

    if not physical_gpus:
        return 2

    with tf.device("/GPU:0"):
        a = tf.random.normal((2048, 2048))
        b = tf.random.normal((2048, 2048))
        c = tf.matmul(a, b)

    print(f"Matmul device: {c.device}")
    print(f"Matmul mean: {float(tf.reduce_mean(c).numpy()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
