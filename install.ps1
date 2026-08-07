# One-time setup on a fresh Windows laptop. Safe to re-run: every step checks
# what already exists and skips it, so this also works as a repair tool.
#
#   .\install.ps1              # full setup (needs internet; models are ~1GB)
#   .\install.ps1 -SkipTools   # don't try to install Python/Ollama via winget
#
# When it finishes there are two icons on the desktop: "Start Reachy" and
# "Reachy Dashboard". See OPERATOR.md for the day-to-day routine.

param([switch]$SkipTools)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# --- Python -----------------------------------------------------------------
# Two layouts are supported: a .venv inside the repo (what this script
# creates) and the original development layout where the repo sits inside the
# virtualenv folder. Whichever exists is used; start_reachy.ps1 looks in the
# same order.
Step "Python environment"
$venvPython  = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$parentPython = Join-Path (Split-Path -Parent $PSScriptRoot) "Scripts\python.exe"

if (Test-Path $venvPython) {
    $python = $venvPython
    Write-Host "   using existing $venvPython"
} elseif (Test-Path $parentPython) {
    $python = $parentPython
    Write-Host "   using existing $parentPython"
} else {
    $base = Get-Command python -ErrorAction SilentlyContinue
    if (-not $base) {
        if ($SkipTools) { Write-Host "   Python not found and -SkipTools set." -ForegroundColor Red; exit 1 }
        Write-Host "   installing Python via winget (this can take a few minutes)..."
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        # winget updates PATH for new shells, not this one:
        $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
    }
    Write-Host "   creating .venv..."
    python -m venv .venv
    $python = $venvPython
}

Step "Python packages"
# anthropic and openai are optional at runtime -- the robot falls back to the
# local Ollama model when neither is installed or no API key is set -- but both
# are installed here so that setting a key is the only step needed to move the
# brain to the cloud, whichever provider the Hub's credits are with.
& $python -m pip install -q --disable-pip-version-check `
    sherpa-onnx piper-tts sounddevice sentencepiece ollama `
    mediapipe opencv-python onnxruntime fastapi uvicorn requests `
    reachy-mini numpy scipy anthropic openai
if ($LASTEXITCODE -ne 0) { Write-Host "   pip failed" -ForegroundColor Red; exit 1 }

# --- Ollama -----------------------------------------------------------------
Step "Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if ($SkipTools) {
        Write-Host "   ollama not found; install it from https://ollama.com/download" -ForegroundColor Yellow
    } else {
        winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
    }
}
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "   pulling the language model (~1GB on first run)..."
    ollama pull qwen2.5:1.5b-instruct-q4_K_M
    if ($LASTEXITCODE -ne 0) {
        # A fresh install's background server may not be up yet.
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 6
        ollama pull qwen2.5:1.5b-instruct-q4_K_M
    }
}

# --- Model files ------------------------------------------------------------
Step "Speech and vision models (~1GB total, skipped where already present)"
New-Item -ItemType Directory -Force models\asr, models\kws, models\tts, models\face | Out-Null

function Fetch($marker, $url, $out) {
    if (Test-Path $marker) { Write-Host "   ok  $marker"; return $false }
    Write-Host "   downloading $(Split-Path -Leaf $out)..."
    Invoke-WebRequest -Uri $url -OutFile $out
    return $true
}

if (Fetch "models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26" `
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2" `
    "models\asr\zipformer-en.tar.bz2") { tar xjf models\asr\zipformer-en.tar.bz2 -C models\asr }

if (Fetch "models\asr\sherpa-onnx-whisper-base.en" `
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2" `
    "models\asr\whisper-base.en.tar.bz2") { tar xjf models\asr\whisper-base.en.tar.bz2 -C models\asr }

if (Fetch "models\kws\sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01" `
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2" `
    "models\kws\kws-gigaspeech.tar.bz2") { tar xjf models\kws\kws-gigaspeech.tar.bz2 -C models\kws }

# Speaking voices. The first is the default; the rest are offered in the
# dashboard's voice selector, which lists whatever is present in models\tts
# rather than a hardcoded set -- so adding a fifth is dropping the pair of
# files in beside these, with no code change anywhere.
foreach ($voice in @(
    @("en_US-amy-medium",    "en/en_US/amy/medium"),
    @("en_US-ryan-medium",   "en/en_US/ryan/medium"),
    @("en_US-lessac-medium", "en/en_US/lessac/medium"),
    @("en_GB-alba-medium",   "en/en_GB/alba/medium")
)) {
    $name, $path = $voice
    Fetch "models\tts\$name.onnx" `
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/$path/$name.onnx" `
        "models\tts\$name.onnx" | Out-Null
    Fetch "models\tts\$name.onnx.json" `
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/$path/$name.onnx.json" `
        "models\tts\$name.onnx.json" | Out-Null
}
Fetch "models\face\blaze_face_short_range.tflite" `
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" `
    "models\face\blaze_face_short_range.tflite" | Out-Null
Fetch "models\face\w600k_mbf.onnx" `
    "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx" `
    "models\face\w600k_mbf.onnx" | Out-Null

# --- Wake phrases -----------------------------------------------------------
# models/ is gitignored, so the tokenized wake-word files do not survive a
# clone; they are regenerated from the tracked assets/wake_phrases.txt.
Step "Wake phrases"
$env:PYTHONIOENCODING = "utf-8"
& $python tools\generate_wake_tokens.py
if ($LASTEXITCODE -ne 0) { Write-Host "   wake phrase generation failed" -ForegroundColor Red; exit 1 }

# --- Desktop icons ----------------------------------------------------------
Step "Desktop icons"
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $desktop "Start Reachy.lnk"))
$lnk.TargetPath = "powershell.exe"
$lnk.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\start_reachy.ps1`""
$lnk.WorkingDirectory = $PSScriptRoot
$lnk.Save()
Set-Content -Encoding ascii (Join-Path $desktop "Reachy Dashboard.url") @"
[InternetShortcut]
URL=http://localhost:8080
"@
Write-Host "   'Start Reachy' and 'Reachy Dashboard' created"

Write-Host "`nDone. Next: read OPERATOR.md, then double-click 'Start Reachy'." -ForegroundColor Green
