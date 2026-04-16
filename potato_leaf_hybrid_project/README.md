## Potato Leaf 3-Class Classifier

This project is configured for a 3-class potato leaf image classification task with the canonical label order:

1. `Potato_Early_Blight`
2. `Potato_Late_Blight`
3. `Potato_Healthy`

The default dataset path is:

```text
data/potato_leaf_dataset
```

Images are resized to `224x224`, normalized to `[0, 1]`, and augmented during training with random flip, rotation, zoom, brightness, and contrast changes.

## Dataset Layout

Expected canonical layout:

```text
data/potato_leaf_dataset/
├── Potato_Early_Blight/
├── Potato_Late_Blight/
└── Potato_Healthy/
```

For compatibility with this workspace, the loader also accepts these legacy folder names on disk:

- `Early_Blight` -> `Potato_Early_Blight`
- `Late_Blight` -> `Potato_Late_Blight`
- `Healthy` -> `Potato_Healthy`

Saved metadata and reports always use the canonical class names above.

## What Is Implemented

- automatic train / validation / test splitting
- custom CNN baseline
- early stopping
- model checkpointing
- saved full Keras model
- saved training curves and history
- test evaluation with:
  - accuracy
  - precision
  - recall
  - F1-score
  - confusion matrix
  - classification report
- Streamlit app for single-image prediction

## Project Structure

```text
potato_leaf_hybrid_project/
├── data/
├── outputs/
│   ├── models/
│   ├── plots/
│   └── reports/
├── src/
│   ├── app.py
│   ├── augmentation.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model_cnn.py
│   ├── model_hybrid_quantum.py
│   ├── train.py
│   └── utils.py
├── main.py
└── requirements.txt
```

## Environment Notes

This workspace is currently using Python `3.13.4`.

- `tensorflow` is available here and the CNN baseline was run successfully.
- `pennylane` is not installed in this interpreter yet.
- `streamlit` is not installed in this interpreter yet.

If you want the full CNN + quantum workflow and the Streamlit app in one environment, Python `3.11` is the safest choice.

## Install

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install streamlit pennylane
```

## Train The CNN Baseline

Recommended command:

```powershell
python src/train.py --model cnn --dataset-dir data\potato_leaf_dataset
```

Quick smoke test:

```powershell
python src/train.py --model cnn --dataset-dir data\potato_leaf_dataset --epochs 1
```

If you want to skip test evaluation during training and evaluate separately:

```powershell
python src/train.py --model cnn --dataset-dir data\potato_leaf_dataset --skip-test-eval
```

## Evaluate The CNN Baseline

```powershell
python src/evaluate.py --model cnn --dataset-dir data\potato_leaf_dataset
```

You can also use the unified entry point:

```powershell
python main.py train --model cnn --dataset-dir data\potato_leaf_dataset
python main.py evaluate --model cnn --dataset-dir data\potato_leaf_dataset
```

## Run The Streamlit App

```powershell
streamlit run src/app.py
```

Or:

```powershell
python main.py app
```

## Saved Outputs

After training and evaluation, the project writes artifacts to `outputs/`:

- `outputs/models/cnn_best.weights.h5`
- `outputs/models/cnn_best.keras`
- `outputs/plots/cnn_training_curves.png`
- `outputs/plots/cnn_confusion_matrix.png`
- `outputs/reports/cnn_history.csv`
- `outputs/reports/cnn_metrics.json`
- `outputs/reports/cnn_classification_report.txt`
- `outputs/reports/cnn_confusion_matrix.csv`
- `outputs/reports/cnn_predictions.csv`
- `outputs/reports/cnn_model_summary.txt`
- `outputs/reports/class_names.json`
- `outputs/reports/dataset_summary.json`
- `outputs/reports/config_snapshot.json`

## Verified Baseline Run In This Workspace

The verified run performed here used:

```powershell
python src/train.py --model cnn --dataset-dir data\potato_leaf_dataset --epochs 1
python src/evaluate.py --model cnn --dataset-dir data\potato_leaf_dataset
```

Observed baseline results from that 1-epoch run:

- test accuracy: `0.3904`
- weighted precision: `0.2255`
- weighted recall: `0.3904`
- weighted F1-score: `0.2859`

These are early baseline numbers from a single epoch only. They are not presented as final performance.

## Next Step

Once you want to continue, the next extension is to install `pennylane`, validate the hybrid model in the same 3-class pipeline, and then expose both trained models through the Streamlit app.

## GPU Setup

For future GPU-enabled runs on your RTX 3050, use the dedicated guide:

- `GPU_SETUP.md`

In short:

- PyTorch GPU is supported on native Windows
- TensorFlow GPU should be run through WSL2 for this machine and TensorFlow version family
