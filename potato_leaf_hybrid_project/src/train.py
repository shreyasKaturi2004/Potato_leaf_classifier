from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import CONFIG
from src.data_loader import load_datasets
from src.evaluate import evaluate_model
from src.utils import (
    build_model_by_name,
    build_hybrid_model_from_cnn_checkpoint,
    compile_classification_model,
    ensure_model_built,
    ensure_directories,
    get_model_architecture_path,
    get_checkpoint_path,
    get_history_csv_path,
    get_model_summary_path,
    get_saved_model_path,
    get_training_plot_path,
    save_model_summary,
    save_project_metadata,
    save_training_curves,
    save_training_history,
    save_model_comparison_report,
    set_global_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a potato leaf disease classifier.")
    parser.add_argument(
        "--model",
        choices=["cnn", "hybrid", "transfer"],
        default=CONFIG.training.default_model_name,
        help="Model type to train.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional path to the dataset root directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional learning-rate override.",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip test-set evaluation after training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the model's existing checkpoint if one is available.",
    )
    return parser


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.dataset_dir:
        CONFIG.paths.dataset_dir = Path(args.dataset_dir)
    if args.epochs:
        CONFIG.training.epochs = args.epochs
    if args.batch_size:
        CONFIG.dataset.batch_size = args.batch_size
    if args.learning_rate:
        CONFIG.training.learning_rate = args.learning_rate


def cli_main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()

    _apply_cli_overrides(args)
    ensure_directories(CONFIG)
    set_global_seed(CONFIG.training.seed)

    dataset_bundle = load_datasets(CONFIG)
    dataset_summary = {
        "dataset_dir": str(CONFIG.paths.dataset_dir),
        "class_names": dataset_bundle.class_names,
        "split_sizes": dataset_bundle.split_sizes,
        "class_distribution": dataset_bundle.class_distribution,
    }
    save_project_metadata(CONFIG, dataset_bundle.class_names, dataset_summary)

    hybrid_initialized_from_cnn = False
    if args.model == "hybrid":
        model, hybrid_initialized_from_cnn = build_hybrid_model_from_cnn_checkpoint(CONFIG)
    else:
        model = build_model_by_name(args.model, CONFIG)
    ensure_model_built(model, CONFIG)
    compile_classification_model(model, CONFIG)
    save_model_summary(model, get_model_summary_path(CONFIG, args.model))

    checkpoint_path = get_checkpoint_path(CONFIG, args.model)
    if args.resume and checkpoint_path.exists():
        model.load_weights(checkpoint_path)
        print(f"Resumed model weights from: {checkpoint_path}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=CONFIG.training.monitor_metric,
            patience=CONFIG.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=CONFIG.training.monitor_metric,
            factor=0.5,
            patience=CONFIG.training.reduce_lr_patience,
            min_lr=CONFIG.training.min_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=CONFIG.training.monitor_metric,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        dataset_bundle.train_ds,
        validation_data=dataset_bundle.val_ds,
        epochs=CONFIG.training.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_history(history, get_history_csv_path(CONFIG, args.model))
    save_training_curves(history, get_training_plot_path(CONFIG, args.model), args.model)

    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)

    saved_model_path = None
    architecture_path = None
    if args.model == "hybrid":
        architecture_path = get_model_architecture_path(CONFIG, args.model)
        architecture_path.write_text(model.to_json(), encoding="utf-8")
    else:
        saved_model_path = get_saved_model_path(CONFIG, args.model)
        model.save(saved_model_path)

    print("\nTraining complete.")
    print(f"Model summary saved to: {get_model_summary_path(CONFIG, args.model)}")
    print(f"Best weights saved to: {checkpoint_path}")
    if saved_model_path is not None:
        print(f"Best full model saved to: {saved_model_path}")
    if architecture_path is not None:
        print(f"Model architecture saved to: {architecture_path}")
    print(f"Training curves saved to: {get_training_plot_path(CONFIG, args.model)}")
    if args.model == "hybrid":
        print(f"Initialized from trained CNN feature extractor: {hybrid_initialized_from_cnn}")

    if not args.skip_test_eval:
        test_loss, _ = model.evaluate(dataset_bundle.test_ds, verbose=1)
        metrics = evaluate_model(
            model=model,
            test_dataset=dataset_bundle.test_ds,
            class_names=dataset_bundle.class_names,
            model_name=args.model,
            test_loss=test_loss,
        )
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1-score: {metrics['f1_weighted']:.4f}")

        if args.model == "hybrid":
            baseline_metrics_path = CONFIG.paths.reports_dir / "cnn_metrics.json"
            if baseline_metrics_path.exists():
                with baseline_metrics_path.open("r", encoding="utf-8") as file:
                    baseline_metrics = json.load(file)
                comparison = save_model_comparison_report(CONFIG, baseline_metrics, metrics)
                print(
                    "Model comparison saved to: "
                    f"{CONFIG.paths.reports_dir / 'model_comparison.json'}"
                )


if __name__ == "__main__":
    cli_main()
