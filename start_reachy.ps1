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
    [switch]$NoBrowser,
    # mDNS name, not a fixed IP: the address is assigned by whatever network
    # the robot joins and changed three times in one session, each time
    # leaving this script's reachability check pointing at a dead address.
    [string]$RobotIp = "reachy-mini.local"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$target = if ($OnRobot) { "robot" } else { "robot_remote" }
# Matches config.py's DAEMON_PORT / REACHY_DAEMON_PORT, so the media-acquire
# call below reaches the same daemon the app itself connects to.
$daemonPort = if ($env:REACHY_DAEMON_PORT) { $env:REACHY_DAEMON_PORT } else { 8888 }
$logPath = Join-Path $PSScriptRoot "reachy_live.log"
$errPath = Join-Path $PSScriptRoot "reachy_live.err.log"

# Resolve the project's interpreter explicitly -- a bare `python` picks up
# whatever is first on PATH, which on at least one machine was a different
# interpreter without the packages, failing with a bare ModuleNotFoundError.
# Two layouts: .venv inside the repo (what install.ps1 creates) or the
# original development layout with the repo inside the virtualenv folder.
$candidates = @(
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $PSScriptRoot) "Scripts\python.exe")
)
$python = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    Write-Host "No Python environment found. Run .\install.ps1 first." -ForegroundColor Red
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
    # Test-Connection's WMI backend throws "Generic failure" instead of
    # returning $false when a .local name resolves to a link-local IPv6
    # address (seen on this hotspot's mDNS replies) -- fall back to a raw
    # ping, which handles that case fine.
    $reachable = $false
    try {
        $reachable = Test-Connection -ComputerName $RobotIp -Count 2 -Quiet
    } catch {
        # ping.exe's exit code, not its text, since IPv6 replies (this mDNS
        # name currently resolves to a link-local IPv6-only address) format
        # as "Reply from ...: time=" with no "TTL=" field like IPv4 has.
        ping.exe $RobotIp -n 2 | Out-Null
        $reachable = ($LASTEXITCODE -eq 0)
    }
    if ($reachable) {
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
        Invoke-RestMethod -Method Post -Uri "http://${RobotIp}:${daemonPort}/api/media/acquire" -TimeoutSec 5 | Out-Null
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

# Every address a phone on the same network could use -- printed because
# "what do I type on my phone" is the first question every new operator asks.
$lanIps = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -ExpandProperty IPAddress)

Write-Host ""
Write-Host "Dashboard (this laptop):  http://localhost:8080" -ForegroundColor Green
foreach ($ip in $lanIps) {
    Write-Host "Dashboard (phone, same WiFi): http://${ip}:8080" -ForegroundColor Green
}
Write-Host ""
Write-Host "Loading speech models (~20s). The robot wiggles when it is ready," -ForegroundColor Cyan
Write-Host "then say 'Hey Reachy'." -ForegroundColor Cyan
Write-Host "Close the Reachy Mini control app first -- it takes the mic." -ForegroundColor DarkGray
Write-Host ""

if (-not $NoBrowser) {
    # Detached so it survives this script's supervision loop; the page shows
    # "starting..." until the app is up, which is the right feedback anyway.
    Start-Process cmd -ArgumentList "/c timeout /t 8 /nobreak >nul & start http://localhost:8080" -WindowStyle Hidden
}

$env:REACHY_TARGET = $target
$env:PYTHONIOENCODING = "utf-8"   # replies contain curly quotes; cp1252 mangles them

# Deliberately NOT `python ... 2>&1 | Tee-Object`: redirecting a native
# command's stderr inside PowerShell wraps each line in a NativeCommandError,
# so a Python traceback arrives as an unreadable .NET error blob with the
# actual message truncated. Let Python write both streams to the log itself
# and tail that instead -- the traceback then survives verbatim.
# Exit code 3 means the robot connection died in a way the app cannot repair
# in-process (see body/voice_loop.py). Relaunching is the fix, and doing it
# here beats leaving a frozen robot that still listens and answers.
$LINK_LOST = 3

while ($true) {
    $py = Start-Process -FilePath $python -ArgumentList "-u", "main.py" `
        -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $logPath -RedirectStandardError $errPath

    if ($py.ExitCode -ne $LINK_LOST) { break }

    Write-Host ""
    Write-Host "Lost the connection to the robot. Restarting..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    try {
        Invoke-RestMethod -Method Post -Uri "http://${RobotIp}:${daemonPort}/api/media/acquire" -TimeoutSec 5 | Out-Null
    } catch { }
}

if ((Test-Path $errPath) -and (Get-Item $errPath).Length -gt 0) {
    Write-Host ""
    Write-Host "--- stderr ---" -ForegroundColor Yellow
    Get-Content $errPath -Tail 40
}
