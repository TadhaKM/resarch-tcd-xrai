# Start the Reachy companion with the laptop as the brain and the robot as
# ears/mouth/body (REACHY_TARGET=robot_remote).
#
#   .\start_reachy.ps1            # run it, streaming the log to this window
#   .\start_reachy.ps1 -OnRobot   # run against the robot's own CPU instead
#
# Speech/LLM run here, so replies come back in ~5s instead of the ~12s the
# robot's own CM4 manages. Ctrl+C stops it.

param(
    [switch]$OnRobot,
    [string]$RobotIp = "10.41.102.231"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$target = if ($OnRobot) { "robot" } else { "robot_remote" }
$logPath = Join-Path $PSScriptRoot "reachy_live.log"
$errPath = Join-Path $PSScriptRoot "reachy_live.err.log"

# The project's dependencies (ollama, sherpa_onnx, piper, reachy_mini) live in
# the reachy_mini_env virtualenv that contains this folder. Resolve its
# interpreter explicitly -- a bare `python` picks up whatever is first on PATH,
# which on this machine is a different interpreter without those packages and
# fails at import with a bare ModuleNotFoundError.
$python = Join-Path (Split-Path -Parent $PSScriptRoot) "Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Could not find the project's Python at:" -ForegroundColor Red
    Write-Host "  $python" -ForegroundColor Red
    Write-Host "Expected reachy_companion to sit inside the reachy_mini_env virtualenv." -ForegroundColor Yellow
    exit 1
}

Write-Host "Reachy companion" -ForegroundColor Cyan
Write-Host "  target : $target"
Write-Host "  log    : $logPath"
Write-Host ""

if (-not $OnRobot) {
    # Fail loudly here rather than 20s later inside model loading: nothing in
    # remote mode works if the robot isn't reachable.
    Write-Host "Checking robot at $RobotIp ... " -NoNewline
    if (Test-Connection -ComputerName $RobotIp -Count 2 -Quiet) {
        Write-Host "reachable" -ForegroundColor Green
    } else {
        Write-Host "UNREACHABLE" -ForegroundColor Red
        Write-Host ""
        Write-Host "The robot isn't answering. Check that it's powered on and joined" -ForegroundColor Yellow
        Write-Host "to the same network/hotspot. If its IP changed, pass the new one:" -ForegroundColor Yellow
        Write-Host "  .\start_reachy.ps1 -RobotIp <new-ip>" -ForegroundColor Yellow
        exit 1
    }

    # The daemon hands its camera/mic to one client at a time and hangs onto
    # them after a client goes away, so a previous run (or the Reachy Mini
    # control app) can leave this one with no audio. Asking for them back is
    # harmless when they're already free.
    try {
        Invoke-RestMethod -Method Post -Uri "http://${RobotIp}:8000/api/media/acquire" -TimeoutSec 5 | Out-Null
        Write-Host "Media acquired from daemon." -ForegroundColor Green
    } catch {
        Write-Host "Could not reacquire media (continuing anyway): $_" -ForegroundColor Yellow
    }
}

# Only one client can hold the robot's mic. A second instance doesn't fail
# loudly -- both connect and split the audio, so the robot simply stops
# responding, which looks like a microphone problem rather than a duplicate
# process. Stopping a launcher shell does not stop the Python child, so these
# accumulate easily across restarts.
$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*main.py*' })
if ($running.Count -gt 0) {
    Write-Host "Already running (PID $($running.ProcessId -join ', ')) -- stopping it first." -ForegroundColor Yellow
    $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "Loading speech models (~20s), then say 'Hey Reachy'." -ForegroundColor Cyan
Write-Host "Close the Reachy Mini control app first -- it takes the mic." -ForegroundColor DarkGray
Write-Host ""

$env:REACHY_TARGET = $target
$env:PYTHONIOENCODING = "utf-8"   # replies contain curly quotes; cp1252 mangles them

# Deliberately NOT `python ... 2>&1 | Tee-Object`: redirecting a native
# command's stderr inside PowerShell wraps each line in a NativeCommandError,
# so a Python traceback arrives as an unreadable .NET error blob with the
# actual message truncated. Let Python write both streams to the log itself
# and tail that instead -- the traceback then survives verbatim.
$py = Start-Process -FilePath $python -ArgumentList "-u", "main.py" `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $logPath -RedirectStandardError $errPath

try {
    # Follow the log until Python exits, so the window shows progress live.
    Get-Content -Path $logPath -Wait -Tail 0 |
        Where-Object { $_ -notmatch 'device_discovery|GetGpuDevices|XNNPACK|InitGoogle|inference_feedback' }
} finally {
    if (-not $py.HasExited) { $py.Kill() }
    if ((Test-Path $errPath) -and (Get-Item $errPath).Length -gt 0) {
        Write-Host ""
        Write-Host "--- stderr ---" -ForegroundColor Yellow
        Get-Content $errPath -Tail 40
    }
}
