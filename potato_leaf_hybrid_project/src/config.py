from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathConfig:
    """Stores all important project paths in one place."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    dataset_dir: Path = PROJECT_ROOT / "data" / "potato_leaf_dataset"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    models_dir: Path = PROJECT_ROOT / "outputs" / "models"
    plots_dir: Path = PROJECT_ROOT / "outputs" / "plots"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "reports"


@dataclass
class DatasetConfig:
    """Dataset-related settings."""

    image_size: tuple[int, int] = (224, 224)
    num_channels: int = 3
    num_classes: int = 3
    batch_size: int = 16
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    shuffle_buffer: int = 1000
    cache_dataset: bool = False


@dataclass
class ModelConfig:
    """Classical CNN settings and optional transfer learning settings."""

    base_filters: tuple[int, ...] = (32, 64, 128)
    dense_units: int = 128
    transfer_dense_units: int = 128
    activation: str = "relu"
    dropout_rate: float = 0.30
    l2_reg: float = 1e-4
    random_rotation: float = 0.08
    random_zoom: float = 0.10
    random_brightness: float = 0.10
    random_contrast: float = 0.10
    freeze_backbone: bool = True
    use_imagenet_weights: bool = True


@dataclass
class QuantumConfig:
    """Quantum layer settings."""

    num_qubits: int = 4
    circuit_layers: int = 2
    pre_quantum_units: int = 32
    post_quantum_units: int = 32
    freeze_feature_extractor: bool = True


@dataclass
class TrainingConfig:
    """Training settings."""

    default_model_name: str = "cnn"
    epochs: int = 25
    learning_rate: float = 1e-3
    early_stopping_patience: int = 6
    reduce_lr_patience: int = 3
    min_learning_rate: float = 1e-6
    monitor_metric: str = "val_loss"
    seed: int = 42


@dataclass
class ProjectConfig:
    """Top-level project configuration."""

    class_names_hint: tuple[str, ...] = (
        "Potato_Early_Blight",
        "Potato_Late_Blight",
        "Potato_Healthy",
    )
    class_name_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "Potato_Early_Blight": ("Potato_Early_Blight", "Early_Blight"),
            "Potato_Late_Blight": ("Potato_Late_Blight", "Late_Blight"),
            "Potato_Healthy": ("Potato_Healthy", "Healthy"),
        }
    )
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> "ProjectConfig":
        split_total = self.dataset.train_split + self.dataset.val_split + self.dataset.test_split
        if abs(split_total - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must equal 1.0")

        if self.dataset.num_classes != len(self.class_names_hint):
            raise ValueError(
                "Dataset num_classes must match the number of expected potato leaf classes."
            )

        if set(self.class_names_hint) != set(self.class_name_aliases):
            raise ValueError("class_name_aliases keys must exactly match class_names_hint.")

        if self.quantum.num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer.")

        return self


CONFIG = ProjectConfig().validate()


def config_to_dict(config: ProjectConfig = CONFIG) -> dict[str, Any]:
    """Converts config dataclasses into a JSON-friendly dictionary."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(asdict(config))
