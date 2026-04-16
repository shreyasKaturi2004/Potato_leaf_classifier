from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import CONFIG
from src.data_loader import load_datasets
from src.utils import (
    build_model_by_name,
    compile_classification_model,
    ensure_model_built,
    ensure_directories,
    get_checkpoint_path,
    get_classification_report_path,
    get_confusion_matrix_path,
    get_confusion_matrix_csv_path,
    get_metrics_json_path,
    get_predictions_csv_path,
    humanize_label,
    save_model_comparison_report,
    save_confusion_matrix_plot,
    save_json,
    save_text,
    set_global_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained potato leaf classifier.")
    parser.add_argument(
        "--model",
        choices=["cnn", "hybrid", "transfer"],
        default=CONFIG.training.default_model_name,
        help="Model type to evaluate.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional path to the dataset root directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to a specific weights file.",
    )
    return parser


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.dataset_dir:
        CONFIG.paths.dataset_dir = Path(args.dataset_dir)
    if args.batch_size:
        CONFIG.dataset.batch_size = args.batch_size


def _collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = model.predict(dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    true_labels: list[int] = []
    for _, batch_labels in dataset:
        true_labels.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())

    return np.asarray(true_labels), predicted_labels, probabilities


def evaluate_model(
    model,
    test_dataset,
    class_names: list[str],
    model_name: str,
    test_loss: float,
) -> dict[str, Any]:
    label_indices = list(range(len(class_names)))
    y_true, y_pred, probabilities = _collect_predictions(model, test_dataset)
    matrix = confusion_matrix(y_true, y_pred, labels=label_indices)
    friendly_class_names = [humanize_label(name) for name in class_names]

    report_text = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=friendly_class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=friendly_class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_name": model_name,
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_names": class_names,
        "classification_report": report_dict,
        "confusion_matrix": matrix,
    }

    save_text(report_text, get_classification_report_path(CONFIG, model_name))
    save_json(metrics, get_metrics_json_path(CONFIG, model_name))
    pd.DataFrame(
        matrix,
        index=friendly_class_names,
        columns=friendly_class_names,
    ).to_csv(get_confusion_matrix_csv_path(CONFIG, model_name), index=True)
    prediction_frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "true_label_index": y_true,
            "true_label": [class_names[index] for index in y_true],
            "predicted_label_index": y_pred,
            "predicted_label": [class_names[index] for index in y_pred],
            "predicted_confidence": probabilities[np.arange(len(y_pred)), y_pred],
        }
    )
    prediction_frame.to_csv(get_predictions_csv_path(CONFIG, model_name), index=False)
    save_confusion_matrix_plot(
        matrix,
        class_names,
        get_confusion_matrix_path(CONFIG, model_name),
        title=f"{model_name.upper()} Confusion Matrix",
    )

    return metrics


def cli_main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()

    _apply_cli_overrides(args)
    ensure_directories(CONFIG)
    set_global_seed(CONFIG.training.seed)

    dataset_bundle = load_datasets(CONFIG)

    model = build_model_by_name(args.model, CONFIG)
    ensure_model_built(model, CONFIG)
    compile_classification_model(model, CONFIG)

    weights_path = Path(args.weights) if args.weights else get_checkpoint_path(CONFIG, args.model)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights were not found at: {weights_path}\n"
            "Train the model first or pass a different --weights path."
        )

    model.load_weights(weights_path)
    test_loss, test_accuracy = model.evaluate(dataset_bundle.test_ds, verbose=1)

    metrics = evaluate_model(
        model=model,
        test_dataset=dataset_bundle.test_ds,
        class_names=dataset_bundle.class_names,
        model_name=args.model,
        test_loss=test_loss,
    )

    print("\nEvaluation complete.")
    print(f"Model: {args.model}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
    print(f"Weighted F1-score: {metrics['f1_weighted']:.4f}")
    print(f"Metrics JSON: {get_metrics_json_path(CONFIG, args.model)}")
    print(f"Classification Report: {get_classification_report_path(CONFIG, args.model)}")
    print(f"Confusion Matrix Plot: {get_confusion_matrix_path(CONFIG, args.model)}")
    print(f"Confusion Matrix CSV: {get_confusion_matrix_csv_path(CONFIG, args.model)}")
    print(f"Predictions CSV: {get_predictions_csv_path(CONFIG, args.model)}")

    if args.model == "hybrid":
        baseline_metrics_path = CONFIG.paths.reports_dir / "cnn_metrics.json"
        if baseline_metrics_path.exists():
            with baseline_metrics_path.open("r", encoding="utf-8") as file:
                baseline_metrics = json.load(file)
            save_model_comparison_report(CONFIG, baseline_metrics, metrics)
            print(f"Model Comparison: {CONFIG.paths.reports_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    cli_main()
