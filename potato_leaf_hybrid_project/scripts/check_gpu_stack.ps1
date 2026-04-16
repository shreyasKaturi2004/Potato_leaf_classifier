$ErrorActionPreference = "Continue"

Write-Host "=== NVIDIA-SMI ==="
nvidia-smi

Write-Host ""
Write-Host "=== Python installations ==="
py -0p

Write-Host ""
Write-Host "=== Current TensorFlow GPU visibility ==="
python -c "import tensorflow as tf; print('physical_gpus=', tf.config.list_physical_devices('GPU')); print('logical_gpus=', tf.config.list_logical_devices('GPU'))"

Write-Host ""
Write-Host "=== WSL status ==="
wsl --status
