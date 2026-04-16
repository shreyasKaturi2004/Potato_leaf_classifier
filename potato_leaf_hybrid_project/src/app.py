from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import CONFIG
from src.utils import (
    build_model_by_name,
    ensure_model_built,
    get_checkpoint_path,
    humanize_label,
    load_saved_class_names,
)


def load_json_file(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_text_file(file_path: Path) -> str | None:
    if not file_path.exists():
        return None
    return file_path.read_text(encoding="utf-8")


def get_metrics_path(model_name: str) -> Path:
    return CONFIG.paths.reports_dir / f"{model_name.lower()}_metrics.json"


def get_classification_report_path(model_name: str) -> Path:
    return CONFIG.paths.reports_dir / f"{model_name.lower()}_classification_report.txt"


def get_confusion_matrix_csv_path(model_name: str) -> Path:
    return CONFIG.paths.reports_dir / f"{model_name.lower()}_confusion_matrix.csv"


def get_confusion_matrix_plot_path(model_name: str) -> Path:
    return CONFIG.paths.plots_dir / f"{model_name.lower()}_confusion_matrix.png"


def get_model_comparison_path() -> Path:
    return CONFIG.paths.reports_dir / "model_comparison.json"


def preprocess_uploaded_image(image_bytes: bytes, image_size: tuple[int, int]) -> tuple[np.ndarray, Image.Image]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    resized = image.resize(image_size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array, image


@st.cache_resource(show_spinner=False)
def load_trained_model(model_name: str, weights_path: str):
    model = build_model_by_name(model_name, CONFIG)
    ensure_model_built(model, CONFIG)
    model.load_weights(weights_path)
    return model


def get_available_models() -> list[str]:
    candidates = ["cnn", "hybrid", "transfer"]
    available = [
        model_name
        for model_name in candidates
        if get_checkpoint_path(CONFIG, model_name).exists()
    ]
    return available or ["cnn"]


def build_metric_comparison_frame(
    cnn_metrics: dict[str, Any],
    hybrid_metrics: dict[str, Any],
) -> pd.DataFrame:
    tracked_metrics = [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Weighted Precision"),
        ("recall_weighted", "Weighted Recall"),
        ("f1_weighted", "Weighted F1-score"),
        ("test_loss", "Test Loss"),
    ]
    rows = []
    for metric_key, label in tracked_metrics:
        cnn_value = float(cnn_metrics[metric_key])
        hybrid_value = float(hybrid_metrics[metric_key])
        rows.append(
            {
                "Metric": label,
                "CNN": cnn_value,
                "Hybrid CNN + Quantum": hybrid_value,
                "Hybrid - CNN": hybrid_value - cnn_value,
            }
        )
    return pd.DataFrame(rows).set_index("Metric")


def build_classification_report_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    report = metrics.get("classification_report", {})
    rows: list[dict[str, Any]] = []

    for class_name in metrics.get("class_names", []):
        label = humanize_label(class_name)
        row = report.get(label)
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "Label": label,
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1-score": row["f1-score"],
                "Support": int(row["support"]),
            }
        )

    for aggregate_label in ("macro avg", "weighted avg"):
        row = report.get(aggregate_label)
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "Label": aggregate_label.title(),
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1-score": row["f1-score"],
                "Support": int(row["support"]),
            }
        )

    rows.append(
        {
            "Label": "Accuracy",
            "Precision": None,
            "Recall": None,
            "F1-score": float(metrics["accuracy"]),
            "Support": int(report.get("weighted avg", {}).get("support", 0)),
        }
    )

    return pd.DataFrame(rows).set_index("Label")


def render_benchmark_section() -> None:
    cnn_metrics = load_json_file(get_metrics_path("cnn"))
    hybrid_metrics = load_json_file(get_metrics_path("hybrid"))
    comparison_metrics = load_json_file(get_model_comparison_path())

    if not cnn_metrics and not hybrid_metrics:
        st.info("No saved evaluation reports were found yet.")
        return

    st.subheader("Saved Test-Set Comparison")
    st.caption("These numbers come from the latest saved checkpoints evaluated on the held-out test split.")

    if cnn_metrics and hybrid_metrics:
        comparison_frame = build_metric_comparison_frame(cnn_metrics, hybrid_metrics)
        st.dataframe(
            comparison_frame.style.format(
                {
                    "CNN": "{:.4f}",
                    "Hybrid CNN + Quantum": "{:.4f}",
                    "Hybrid - CNN": "{:+.4f}",
                }
            ),
            use_container_width=True,
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Hybrid Accuracy",
            f"{hybrid_metrics['accuracy']:.4f}",
            f"{hybrid_metrics['accuracy'] - cnn_metrics['accuracy']:+.4f} vs CNN",
        )
        metric_cols[1].metric(
            "Hybrid Precision",
            f"{hybrid_metrics['precision_weighted']:.4f}",
            f"{hybrid_metrics['precision_weighted'] - cnn_metrics['precision_weighted']:+.4f}",
        )
        metric_cols[2].metric(
            "Hybrid Recall",
            f"{hybrid_metrics['recall_weighted']:.4f}",
            f"{hybrid_metrics['recall_weighted'] - cnn_metrics['recall_weighted']:+.4f}",
        )
        metric_cols[3].metric(
            "Hybrid F1-score",
            f"{hybrid_metrics['f1_weighted']:.4f}",
            f"{hybrid_metrics['f1_weighted'] - cnn_metrics['f1_weighted']:+.4f}",
        )

        if comparison_metrics:
            st.caption(
                "Current winner on test accuracy: "
                + (
                    "Hybrid CNN + Quantum"
                    if hybrid_metrics["accuracy"] >= cnn_metrics["accuracy"]
                    else "CNN Baseline"
                )
            )
    else:
        available_metrics = cnn_metrics or hybrid_metrics
        st.json(available_metrics)

    confusion_cols = st.columns(2)
    for column, model_name, heading in (
        (confusion_cols[0], "cnn", "CNN Confusion Matrix"),
        (confusion_cols[1], "hybrid", "Hybrid Confusion Matrix"),
    ):
        with column:
            st.markdown(f"**{heading}**")
            plot_path = get_confusion_matrix_plot_path(model_name)
            csv_path = get_confusion_matrix_csv_path(model_name)
            if plot_path.exists():
                st.image(str(plot_path), use_container_width=True)
            if csv_path.exists():
                st.dataframe(pd.read_csv(csv_path, index_col=0), use_container_width=True)

    report_cols = st.columns(2)
    for column, model_name, heading in (
        (report_cols[0], "cnn", "CNN Classification Report"),
        (report_cols[1], "hybrid", "Hybrid Classification Report"),
    ):
        with column:
            metrics = load_json_file(get_metrics_path(model_name))
            report_text = load_text_file(get_classification_report_path(model_name))
            st.markdown(f"**{heading}**")
            if metrics:
                st.dataframe(
                    build_classification_report_frame(metrics).style.format(
                        {
                            "Precision": "{:.4f}",
                            "Recall": "{:.4f}",
                            "F1-score": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                )
            if report_text:
                with st.expander(f"Raw {model_name.upper()} classification report"):
                    st.code(report_text)


def main() -> None:
    st.set_page_config(page_title="Potato Leaf Classifier", layout="wide")
    st.title("Potato Leaf Disease Classification")
    st.write(
        "Upload a potato leaf image to predict one of the three configured classes: "
        "Potato Early Blight, Potato Late Blight, or Potato Healthy."
    )
    st.caption(
        "Images are resized to 224x224 and normalized in the same way used during training."
    )

    available_models = get_available_models()
    default_choice = "hybrid" if "hybrid" in available_models else "cnn"
    default_index = available_models.index(default_choice) if default_choice in available_models else 0
    selected_model = st.sidebar.selectbox("Choose model", available_models, index=default_index)

    weights_path = get_checkpoint_path(CONFIG, selected_model)
    if not weights_path.exists():
        st.warning(
            f"No trained weights were found for `{selected_model}` at:\n\n{weights_path}\n\n"
            "Train the model first before using the demo."
        )
        return

    prediction_tab, benchmark_tab = st.tabs(["Single Image Prediction", "CNN vs Hybrid Benchmark"])

    with benchmark_tab:
        render_benchmark_section()

    with prediction_tab:
        uploaded_file = st.file_uploader(
            "Upload a potato leaf image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        selected_metrics = load_json_file(get_metrics_path(selected_model))
        if selected_metrics:
            st.caption(
                f"Selected model `{selected_model}` latest test accuracy: "
                f"{selected_metrics['accuracy']:.4f}"
            )

        if uploaded_file is None:
            st.info("Awaiting an image file.")
            return

        try:
            image_array, original_image = preprocess_uploaded_image(
                uploaded_file.getvalue(),
                CONFIG.dataset.image_size,
            )
            class_names = load_saved_class_names(CONFIG)

            with st.spinner("Loading model and predicting..."):
                model = load_trained_model(selected_model, str(weights_path))
                probabilities = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_label = humanize_label(class_names[predicted_index])
            confidence = float(probabilities[predicted_index]) * 100.0

            st.image(original_image, caption="Uploaded Image", use_container_width=True)
            st.subheader(f"Prediction: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}%")

            result_frame = pd.DataFrame(
                {
                    "Class": [humanize_label(name) for name in class_names],
                    "Confidence": probabilities,
                }
            ).set_index("Class")
            st.bar_chart(result_frame)

            with st.expander("Label mapping"):
                st.json({str(index): label for index, label in enumerate(class_names)})
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
