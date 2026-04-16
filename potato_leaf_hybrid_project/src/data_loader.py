from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import ProjectConfig


AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class DatasetBundle:
    """Groups all dataset objects and metadata in one return value."""

    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    class_names: list[str]
    split_sizes: dict[str, int]
    class_distribution: dict[str, dict[str, int]]


def _resolve_class_directories(
    dataset_dir: Path,
    config: ProjectConfig,
) -> list[tuple[str, Path]]:
    available_dirs = {
        item.name: item
        for item in dataset_dir.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    }

    resolved: list[tuple[str, Path]] = []
    consumed_dir_names: set[str] = set()
    missing_classes: list[str] = []

    for canonical_name in config.class_names_hint:
        aliases = config.class_name_aliases[canonical_name]
        matches = [available_dirs[alias] for alias in aliases if alias in available_dirs]

        if not matches:
            alias_text = ", ".join(aliases)
            missing_classes.append(f"{canonical_name} (accepted folder names: {alias_text})")
            continue

        if len(matches) > 1:
            matching_names = [match.name for match in matches]
            raise ValueError(
                f"Multiple folders matched the same class '{canonical_name}': {matching_names}. "
                "Keep only one folder per class to avoid ambiguous label mapping."
            )

        resolved_dir = matches[0]
        resolved.append((canonical_name, resolved_dir))
        consumed_dir_names.add(resolved_dir.name)

    if missing_classes:
        raise ValueError(
            "The dataset directory is missing one or more required class folders:\n- "
            + "\n- ".join(missing_classes)
        )

    unexpected_dirs = sorted(set(available_dirs) - consumed_dir_names)
    if unexpected_dirs:
        raise ValueError(
            "Unexpected class folders were found in the dataset directory: "
            f"{unexpected_dirs}. This project expects only the configured 3 potato classes."
        )

    return resolved


def _discover_image_files(dataset_dir: Path, config: ProjectConfig) -> tuple[list[str], list[int], list[str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: {dataset_dir}\n"
            "Set the correct path in src/config.py or pass --dataset-dir while training."
        )

    class_dir_pairs = _resolve_class_directories(dataset_dir, config)
    if not class_dir_pairs:
        raise ValueError(
            "No class folders were found inside the dataset directory. "
            "Expected one folder per class."
        )

    class_names = list(config.class_names_hint)

    image_paths: list[str] = []
    labels: list[int] = []

    for label_index, (canonical_name, class_dir) in enumerate(class_dir_pairs):
        class_images = [
            file_path
            for file_path in class_dir.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in config.dataset.allowed_extensions
        ]

        if not class_images:
            raise ValueError(
                f"No supported image files were found inside: {class_dir} "
                f"for class '{canonical_name}'."
            )

        for file_path in sorted(class_images):
            image_paths.append(str(file_path))
            labels.append(label_index)

    return image_paths, labels, class_names


def _summarize_distribution(labels: list[int], class_names: list[str]) -> dict[str, int]:
    label_array = np.asarray(labels)
    return {
        class_names[index]: int(np.sum(label_array == index))
        for index in range(len(class_names))
    }


def _split_filepaths(
    image_paths: list[str],
    labels: list[int],
    config: ProjectConfig,
) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    test_and_val_fraction = config.dataset.val_split + config.dataset.test_split

    try:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths,
            labels,
            test_size=test_and_val_fraction,
            stratify=labels,
            random_state=config.training.seed,
        )
    except ValueError as exc:
        raise ValueError(
            "The dataset is too small for a stratified train/validation/test split. "
            "Please add more images per class."
        ) from exc

    relative_test_fraction = config.dataset.test_split / test_and_val_fraction

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_fraction,
        stratify=temp_labels,
        random_state=config.training.seed,
    )

    return (
        (list(train_paths), list(train_labels)),
        (list(val_paths), list(val_labels)),
        (list(test_paths), list(test_labels)),
    )


def _load_image(path: tf.Tensor, label: tf.Tensor, config: ProjectConfig) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=config.dataset.num_channels,
        expand_animations=False,
    )
    image = tf.image.resize(image, config.dataset.image_size)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((*config.dataset.image_size, config.dataset.num_channels))

    one_hot_label = tf.one_hot(label, depth=config.dataset.num_classes)
    return image, one_hot_label


def _build_tf_dataset(
    image_paths: list[str],
    labels: list[int],
    config: ProjectConfig,
    training: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(image_paths), config.dataset.shuffle_buffer),
            seed=config.training.seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: _load_image(path, label, config),
        num_parallel_calls=AUTOTUNE,
    )

    if config.dataset.cache_dataset:
        dataset = dataset.cache()

    dataset = dataset.batch(config.dataset.batch_size).prefetch(AUTOTUNE)
    return dataset


def load_datasets(config: ProjectConfig) -> DatasetBundle:
    """
    Loads potato leaf images from a folder-per-class dataset and returns
    train/validation/test datasets along with useful metadata.
    """

    image_paths, labels, class_names = _discover_image_files(config.paths.dataset_dir, config)
    train_split, val_split, test_split = _split_filepaths(image_paths, labels, config)

    train_paths, train_labels = train_split
    val_paths, val_labels = val_split
    test_paths, test_labels = test_split

    train_ds = _build_tf_dataset(train_paths, train_labels, config, training=True)
    val_ds = _build_tf_dataset(val_paths, val_labels, config, training=False)
    test_ds = _build_tf_dataset(test_paths, test_labels, config, training=False)

    split_sizes = {
        "train": len(train_paths),
        "validation": len(val_paths),
        "test": len(test_paths),
        "total": len(image_paths),
    }

    class_distribution = {
        "train": _summarize_distribution(train_labels, class_names),
        "validation": _summarize_distribution(val_labels, class_names),
        "test": _summarize_distribution(test_labels, class_names),
    }

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        class_names=class_names,
        split_sizes=split_sizes,
        class_distribution=class_distribution,
    )
