# Resonova System Diagnostics Check Script
# Run this in PowerShell to identify hardware, software, and dependency issues.

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "            RESONOVA SYSTEM CONFIGURATION CHECK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check OS and PowerShell
Write-Host "[1/6] System Environment:" -ForegroundColor Yellow
Write-Host "  OS: Windows ($([System.Environment]::OSVersion.VersionString))"
Write-Host "  Processor count: $env:NUMBER_OF_PROCESSORS"

# 2. Check Python Version
Write-Host ""
Write-Host "[2/6] Python Environment:" -ForegroundColor Yellow
$pyPath = Get-Command python -ErrorAction SilentlyContinue
if ($pyPath) {
    $pyVer = python -V 2>&1
    Write-Host "  Python Executable: $($pyPath.Source)"
    Write-Host "  Python Version: $pyVer"
    if ($pyVer -like "*3.9*") {
        Write-Host "  [OK] Python 3.9 detected (Correct version)." -ForegroundColor Green
    } else {
        Write-Host "  [WARN] WARNING: Pinned Python version is 3.9, but you are running $pyVer." -ForegroundColor Red
        Write-Host "    Wav2Lip and older PyTorch dependencies are highly likely to fail on Python 3.10+." -ForegroundColor Red
    }
} else {
    Write-Host "  [ERROR] ERROR: Python is not found in your PATH." -ForegroundColor Red
    Write-Host "    Please install Python 3.9 and add it to PATH or use Miniconda." -ForegroundColor Red
}

# 3. Check ffmpeg and ffprobe
Write-Host ""
Write-Host "[3/6] Media Utilities (ffmpeg and ffprobe):" -ForegroundColor Yellow

# Look for ffmpeg in local folder first
$localFfmpeg = Join-Path $PSScriptRoot "ffmpeg.exe"
$localFfprobe = Join-Path $PSScriptRoot "ffprobe.exe"

if (Test-Path $localFfmpeg) {
    Write-Host "  [OK] Found local ffmpeg.exe in project root: $localFfmpeg" -ForegroundColor Green
} else {
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        Write-Host "  [OK] Found system ffmpeg in PATH: $($ffmpegCmd.Source)" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] ERROR: ffmpeg.exe not found in project root or system PATH." -ForegroundColor Red
        Write-Host "    Whisper and pydub require ffmpeg to process audio." -ForegroundColor Red
    }
}

if (Test-Path $localFfprobe) {
    Write-Host "  [OK] Found local ffprobe.exe in project root: $localFfprobe" -ForegroundColor Green
} else {
    $ffprobeCmd = Get-Command ffprobe -ErrorAction SilentlyContinue
    if ($ffprobeCmd) {
        Write-Host "  [OK] Found system ffprobe in PATH: $($ffprobeCmd.Source)" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] WARNING: ffprobe.exe not found in project root or system PATH." -ForegroundColor Yellow
    }
}

# 4. Check PyTorch and CUDA VRAM
Write-Host ""
Write-Host "[4/6] PyTorch and GPU Hardware Acceleration (CUDA):" -ForegroundColor Yellow

$pyCheck = @'
import sys
import os

try:
    import torch
    print(f"torch_version:{torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"cuda_available:{cuda_avail}")
    if cuda_avail:
        print(f"gpu_name:{torch.cuda.get_device_name(0)}")
        print(f"gpu_vram:{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    print("torch_installed:False")
'@

$tempFile = [System.IO.Path]::GetTempFileName() + ".py"
$pyCheck | Out-File -FilePath $tempFile -Encoding utf8
$pyOut = python $tempFile 2>&1
Remove-Item $tempFile -ErrorAction SilentlyContinue

if ($pyOut -match "torch_installed:False") {
    Write-Host "  [ERROR] ERROR: PyTorch is not installed in the active environment." -ForegroundColor Red
} else {
    $torchVer = ""
    $cudaAvail = "False"
    $gpuName = ""
    $gpuVram = ""

    foreach ($line in ($pyOut -split "`n")) {
        if ($line -match "torch_version:(.*)") { $torchVer = $Matches[1].Trim() }
        if ($line -match "cuda_available:(.*)") { $cudaAvail = $Matches[1].Trim() }
        if ($line -match "gpu_name:(.*)") { $gpuName = $Matches[1].Trim() }
        if ($line -match "gpu_vram:(.*)") { $gpuVram = $Matches[1].Trim() }
    }

    Write-Host "  PyTorch Version: $torchVer"
    if ($cudaAvail -eq "True") {
        Write-Host "  [OK] CUDA is AVAILABLE." -ForegroundColor Green
        Write-Host "  GPU Device: $gpuName"
        Write-Host "  Total VRAM: $gpuVram"
    } else {
        Write-Host "  [WARN] CUDA is NOT AVAILABLE (Running in CPU mode)." -ForegroundColor Yellow
        Write-Host "    Model inference will run extremely slowly. Ensure you have installed CUDA-enabled PyTorch." -ForegroundColor Red
    }
}

# 5. Check Package Imports and NumPy version compatibility
Write-Host ""
Write-Host "[5/6] Python Package Integrity Checks:" -ForegroundColor Yellow

$pkgCheckCode = @'
packages = [
    ("numpy", "NumPy (Scientific data stack)"),
    ("whisper", "OpenAI Whisper (ASR / Transcription)"),
    ("transformers", "HuggingFace Transformers (Translation tokenizer)"),
    ("TTS", "Coqui TTS (Voice Cloning / XTTS)"),
    ("face_alignment", "Face Alignment (Lip-Sync detection)"),
    ("cv2", "OpenCV (Frame and video handling)"),
    ("gradio", "Gradio (Web App UI)")
]

for name, desc in packages:
    try:
        mod = __import__(name)
        if name == "numpy":
            import numpy as np
            print(f"OK:{name}:{desc}:{np.__version__}")
        else:
            print(f"OK:{name}:{desc}:")
    except Exception as e:
        print(f"FAIL:{name}:{desc}:{str(e)}")
'@

$tempFile2 = [System.IO.Path]::GetTempFileName() + ".py"
$pkgCheckCode | Out-File -FilePath $tempFile2 -Encoding utf8
$pkgOut = python $tempFile2 2>&1
Remove-Item $tempFile2 -ErrorAction SilentlyContinue

foreach ($line in ($pkgOut -split "`r?`n")) {
    if ($line -match "(OK|FAIL):([^:]+):([^:]+):(.*)") {
        $status = $Matches[1]
        $pkgName = $Matches[2]
        $pkgDesc = $Matches[3]
        $pkgInfo = $Matches[4].Trim()

        if ($status -eq "OK") {
            if ($pkgName -eq "numpy") {
                if ($pkgInfo -like "1.24*" -or $pkgInfo -like "1.25*" -or $pkgInfo -like "1.26*" -or $pkgInfo -like "2.*") {
                    Write-Host "  [WARN] $($pkgDesc): Import OK, but version is $($pkgInfo)." -ForegroundColor Yellow
                    Write-Host "     NumPy >=1.24 deprecated/removed 'np.bool' and 'np.int', causing Wav2Lip to crash." -ForegroundColor Red
                    Write-Host "     Fix: pip install numpy==1.23.5" -ForegroundColor Yellow
                } else {
                    Write-Host "  [OK] $($pkgDesc): Version $($pkgInfo) OK" -ForegroundColor Green
                }
            } else {
                Write-Host "  [OK] $($pkgDesc): Import OK" -ForegroundColor Green
            }
        } else {
            Write-Host "  [ERROR] $($pkgDesc): FAILED to import." -ForegroundColor Red
            Write-Host "     Error detail: $($pkgInfo)" -ForegroundColor DarkRed
        }
    }
}

# 6. Check Environment Variables for Wav2Lip
Write-Host ""
Write-Host "[6/6] Environment Paths (Wav2Lip configurations):" -ForegroundColor Yellow

$wav2lipRepo = $env:WAV2LIP_REPO_PATH
$wav2lipCkpt = $env:WAV2LIP_CHECKPOINT_PATH

if ($wav2lipRepo) {
    Write-Host "  WAV2LIP_REPO_PATH: $wav2lipRepo"
    if (Test-Path $wav2lipRepo) {
        Write-Host "  [OK] Wav2Lip Repo directory exists." -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] ERROR: Directory does not exist: $wav2lipRepo" -ForegroundColor Red
    }
} else {
    Write-Host "  [WARN] WAV2LIP_REPO_PATH environment variable is not set." -ForegroundColor Yellow
}

if ($wav2lipCkpt) {
    Write-Host "  WAV2LIP_CHECKPOINT_PATH: $wav2lipCkpt"
    if (Test-Path $wav2lipCkpt) {
        Write-Host "  [OK] Wav2Lip checkpoint file exists." -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] ERROR: Checkpoint file does not exist at: $wav2lipCkpt" -ForegroundColor Red
    }
} else {
    Write-Host "  [WARN] WAV2LIP_CHECKPOINT_PATH environment variable is not set." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Check complete. Review warnings/errors above to fix your env." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
