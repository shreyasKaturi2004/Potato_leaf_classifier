$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venvs\torch-cu128"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "Creating PyTorch GPU environment at $VenvPath"
python -m venv $VenvPath

Write-Host "Upgrading pip"
& $PythonExe -m pip install --upgrade pip

Write-Host "Installing PyTorch CUDA wheels"
& $PythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "Verifying CUDA visibility from PyTorch"
& $PythonExe (Join-Path $ProjectRoot "scripts\verify_torch_gpu.py")
