## GPU Setup For RTX 3050

This machine has an NVIDIA RTX 3050, but the supported GPU path depends on the framework:

- PyTorch GPU: supported on native Windows
- TensorFlow GPU: supported through WSL2, not native Windows for TensorFlow 2.11+

Why:

- TensorFlow officially supports GPU on Linux and WSL2 with `tensorflow[and-cuda]`
- Native Windows TensorFlow GPU support ended after TensorFlow 2.10
- Microsoft DirectML for TensorFlow is discontinued and only supports Python up to 3.10

Current local facts already verified on this machine:

- GPU present in `nvidia-smi`
- Current Python versions:
  - `3.13`
  - `3.12`
- The accessible native interpreter for automation right now is `3.13`
- Current TensorFlow environment is native Windows CPU-only
- WSL is not installed yet

## Recommended Future Layout

Use separate environments:

1. Windows native PyTorch GPU environment
2. WSL2 Ubuntu TensorFlow GPU environment

That avoids breaking the current CPU-based project environment.

## 1. PyTorch GPU On Windows

Create and verify the Windows GPU environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_pytorch_gpu_windows.ps1
```

Manual equivalent:

```powershell
python -m venv .venvs\torch-cu128
.\.venvs\torch-cu128\Scripts\python.exe -m pip install --upgrade pip
.\.venvs\torch-cu128\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venvs\torch-cu128\Scripts\python.exe .\scripts\verify_torch_gpu.py
```

Expected success signal:

- `CUDA available: True`
- device name includes `RTX 3050`

## 2. TensorFlow GPU Through WSL2

TensorFlow GPU should be run in WSL2 Ubuntu, not native Windows.

First-time one-time Windows setup:

```powershell
wsl --install -d Ubuntu
```

After the reboot and Ubuntu first-launch setup, from Windows PowerShell run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_tensorflow_gpu_wsl.ps1
```

That script:

- uses the project from `/mnt/c/...`
- creates `.venvs/tf-gpu-wsl`
- installs `tensorflow[and-cuda]`
- installs project dependencies needed for training and the app
- runs GPU verification

Expected success signal:

- `Physical GPUs:` is non-empty
- matmul runs on `/device:GPU:0`

## Verification Commands

Windows stack summary:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_gpu_stack.ps1
```

PyTorch GPU check:

```powershell
.\.venvs\torch-cu128\Scripts\python.exe .\scripts\verify_torch_gpu.py
```

TensorFlow GPU check in WSL:

```powershell
wsl bash -lc "cd /mnt/c/Users/skatu/Desktop/Potato_leaf_pr/potato_leaf_hybrid_project && source .venvs/tf-gpu-wsl/bin/activate && python scripts/verify_tensorflow_gpu.py"
```

## Training Guidance

For this current project:

- keep native Windows for quick CPU-only TensorFlow work if needed
- use WSL2 for future TensorFlow GPU training
- use the Windows PyTorch GPU environment only for future PyTorch experiments or migrations

## Official References

- TensorFlow install: https://www.tensorflow.org/install
- PyTorch local install: https://pytorch.org/get-started/locally/
- Microsoft DirectML TensorFlow note: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin
- NVIDIA CUDA on WSL overview: https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl
