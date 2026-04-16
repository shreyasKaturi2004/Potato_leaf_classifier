$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WindowsProjectPath = (Resolve-Path $ProjectRoot).Path
$LinuxProjectPath = "/mnt/" + $WindowsProjectPath.Substring(0, 1).ToLower() + $WindowsProjectPath.Substring(2).Replace("\", "/")

wsl bash -lc "cd '$LinuxProjectPath' && bash scripts/setup_tensorflow_gpu_wsl.sh"
