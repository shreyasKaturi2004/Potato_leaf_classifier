from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import ProjectConfig, config_to_dict


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories(config: ProjectConfig) -> None:
    for directory in (
        config.paths.data_dir,
        config.paths.outputs_dir,
        config.paths.models_dir,
        config.paths.plots_dir,
        config.paths.reports_dir,
        config.paths.project_root / "notebooks",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable.")


def save_json(data: Any, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=_json_default)


def save_text(text: str, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")


def humanize_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip()


def ensure_model_built(model: tf.keras.Model, config: ProjectConfig) -> tf.keras.Model:
    dummy_batch = tf.zeros(
        (1, *config.dataset.image_size, config.dataset.num_channels),
        dtype=tf.float32,
    )
    model(dummy_batch, training=False)
    return model


def compile_classification_model(model: tf.keras.Model, config: ProjectConfig) -> tf.keras.Model:
    if model.name == "hybrid_cnn_quantum_model":
        tf.get_logger().setLevel("ERROR")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
        run_eagerly=(model.name == "hybrid_cnn_quantum_model"),
    )
    return model


def build_model_by_name(model_name: str, config: ProjectConfig) -> tf.keras.Model:
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        from src.model_cnn import build_custom_cnn_model

        return build_custom_cnn_model(config)

    if model_name == "hybrid":
        from src.model_hybrid_quantum import build_hybrid_quantum_model

        return build_hybrid_quantum_model(config)

    if model_name == "transfer":
        from src.model_cnn import build_transfer_learning_model

        return build_transfer_learning_model(config)

    raise ValueError("model_name must be one of: cnn, hybrid, transfer")


def build_hybrid_model_from_cnn_checkpoint(
    config: ProjectConfig,
    cnn_checkpoint_path: Path | None = None,
) -> tuple[tf.keras.Model, bool]:
    from src.model_cnn import build_custom_cnn_model, build_feature_extractor
    from src.model_hybrid_quantum import build_hybrid_quantum_model

    checkpoint_path = cnn_checkpoint_path or get_checkpoint_path(config, "cnn")
    cnn_model = build_custom_cnn_model(config)
    feature_extractor = build_feature_extractor(config)
    checkpoint_loaded = False

    if checkpoint_path.exists():
        cnn_model.load_weights(checkpoint_path)
        baseline_layers = {layer.name: layer for layer in cnn_model.layers}
        for layer in feature_extractor.layers:
            baseline_layer = baseline_layers.get(layer.name)
            if baseline_layer is None:
                continue

            baseline_weights = baseline_layer.get_weights()
            if not baseline_weights:
                continue

            try:
                layer.set_weights(baseline_weights)
            except ValueError:
                continue
        checkpoint_loaded = True

    hybrid_model = build_hybrid_quantum_model(config, feature_extractor=feature_extractor)
    return hybrid_model, checkpoint_loaded


def get_checkpoint_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.models_dir / f"{model_name.lower()}_best.weights.h5"


def get_saved_model_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.models_dir / f"{model_name.lower()}_best.keras"


def get_model_architecture_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.models_dir / f"{model_name.lower()}_architecture.json"


def get_history_csv_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.reports_dir / f"{model_name.lower()}_history.csv"


def get_training_plot_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.plots_dir / f"{model_name.lower()}_training_curves.png"


def get_confusion_matrix_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.plots_dir / f"{model_name.lower()}_confusion_matrix.png"


def get_confusion_matrix_csv_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.reports_dir / f"{model_name.lower()}_confusion_matrix.csv"


def get_classification_report_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.reports_dir / f"{model_name.lower()}_classification_report.txt"


def get_metrics_json_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.reports_dir / f"{model_name.lower()}_metrics.json"


def get_predictions_csv_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.reports_dir / f"{model_name.lower()}_predictions.csv"


def get_model_summary_path(config: ProjectConfig, model_name: str) -> Path:
    return config.paths.reports_dir / f"{model_name.lower()}_model_summary.txt"


def get_model_comparison_json_path(config: ProjectConfig) -> Path:
    return config.paths.reports_dir / "model_comparison.json"


def get_model_comparison_markdown_path(config: ProjectConfig) -> Path:
    return config.paths.reports_dir / "model_comparison.md"


def get_class_names_path(config: ProjectConfig) -> Path:
    return config.paths.reports_dir / "class_names.json"


def get_dataset_summary_path(config: ProjectConfig) -> Path:
    return config.paths.reports_dir / "dataset_summary.json"


def get_config_snapshot_path(config: ProjectConfig) -> Path:
    return config.paths.reports_dir / "config_snapshot.json"


def save_model_summary(model: tf.keras.Model, file_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    save_text("\n".join(lines), file_path)


def save_training_history(history: tf.keras.callbacks.History, file_path: Path) -> None:
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(file_path, index=False)


def save_training_curves(
    history: tf.keras.callbacks.History,
    file_path: Path,
    model_name: str,
) -> None:
    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, metrics["loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, metrics["val_loss"], marker="o", label="Validation Loss")
    axes[0].set_title(f"{model_name.upper()} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, metrics["accuracy"], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, metrics["val_accuracy"], marker="o", label="Validation Accuracy")
    axes[1].set_title(f"{model_name.upper()} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()
    figure.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    file_path: Path,
    title: str,
) -> None:
    display_names = [humanize_label(name) for name in class_names]
    figure, axis = plt.subplots(figsize=(8, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_names)
    display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=45)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_project_metadata(
    config: ProjectConfig,
    class_names: list[str],
    dataset_summary: dict[str, Any],
) -> None:
    save_json(class_names, get_class_names_path(config))
    save_json(dataset_summary, get_dataset_summary_path(config))
    save_json(config_to_dict(config), get_config_snapshot_path(config))


def load_saved_class_names(config: ProjectConfig) -> list[str]:
    class_names_path = get_class_names_path(config)
    if class_names_path.exists():
        with class_names_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return list(config.class_names_hint)


def save_model_comparison_report(
    config: ProjectConfig,
    baseline_metrics: dict[str, Any],
    hybrid_metrics: dict[str, Any],
) -> dict[str, Any]:
    tracked_metrics = [
        "test_loss",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    deltas = {
        metric_name: float(hybrid_metrics[metric_name] - baseline_metrics[metric_name])
        for metric_name in tracked_metrics
    }
    comparison = {
        "baseline_model": baseline_metrics["model_name"],
        "hybrid_model": hybrid_metrics["model_name"],
        "tracked_metrics": tracked_metrics,
        "baseline_metrics": {metric_name: baseline_metrics[metric_name] for metric_name in tracked_metrics},
        "hybrid_metrics": {metric_name: hybrid_metrics[metric_name] for metric_name in tracked_metrics},
        "delta_hybrid_minus_baseline": deltas,
    }
    save_json(comparison, get_model_comparison_json_path(config))

    lines = [
        "# Model Comparison",
        "",
        f"Baseline: {baseline_metrics['model_name']}",
        f"Hybrid: {hybrid_metrics['model_name']}",
        "",
        "| Metric | Baseline CNN | Hybrid CNN + Quantum | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric_name in tracked_metrics:
        lines.append(
            f"| {metric_name} | "
            f"{baseline_metrics[metric_name]:.4f} | "
            f"{hybrid_metrics[metric_name]:.4f} | "
            f"{deltas[metric_name]:+.4f} |"
        )
    save_text("\n".join(lines), get_model_comparison_markdown_path(config))
    return comparison
