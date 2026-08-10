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
    # Empty means "go and find it" (find_robot.ps1). Pass an address to skip
    # the search. Neither a fixed IP nor the mDNS name works as a default: the
    # address is assigned by whatever network the robot joined and has changed
    # three times in one session, and mDNS is filtered on the Hub's own IoT
    # network, so the name resolves at home and not at work.
    [string]$RobotIp = ""
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
    # Find the robot before anything else. An operator should not have to know
    # the robot's IP address, and on the Hub's network they cannot look it up
    # by name -- mDNS is filtered there. find_robot.ps1 tries the last known
    # address, mDNS, the ARP table and finally a sweep of this machine's own
    # subnet, and confirms a Reachy daemon is answering rather than merely that
    # something is at that address.
    if (-not $RobotIp) {
        Write-Host "Looking for the robot ... " -NoNewline
        # Returns "<ip>:<port>". The port matters as much as the address: the
        # daemon's default is 8000 and this project's config asks for 8888,
        # and guessing wrong presents as an unexplained media timeout rather
        # than as a wrong port.
        $found = & (Join-Path $PSScriptRoot "find_robot.ps1")
        if ($found) {
            $parts = $found -split ":"
            $RobotIp = $parts[0]
            if ($parts.Count -gt 1) { $daemonPort = $parts[1] }
        }
        if ($LASTEXITCODE -ne 0 -or -not $RobotIp) {
            Write-Host "NOT FOUND" -ForegroundColor Red
            Write-Host ""
            Write-Host "No Reachy daemon answered anywhere on this network." -ForegroundColor Yellow
            Write-Host "Check that:" -ForegroundColor Yellow
            Write-Host "  - the robot is powered on (the antennas move when it boots)" -ForegroundColor Yellow
            Write-Host "  - it has joined the same WiFi as this laptop" -ForegroundColor Yellow
            Write-Host "  - this laptop is on that WiFi too, not a different one" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "If you know the address, pass it: .\start_reachy.ps1 -RobotIp <ip>" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "found at ${RobotIp}:$daemonPort" -ForegroundColor Green
    }

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
        Write-Host "The robot answered a moment ago but is not responding now." -ForegroundColor Yellow
        Write-Host "Check that it's powered on and joined to the same network." -ForegroundColor Yellow
        Write-Host "If its IP changed, pass the new one:" -ForegroundColor Yellow
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

# Only one client can hold the robot's mic and speaker. A second instance does
# not fail loudly -- both connect, and the robot talks over itself while
# hearing half of what is said.
#
# The SUPERVISING SHELL has to go first, and this is not tidiness. Each
# launcher relaunches its child on any non-zero exit, so killing the Python
# while its launcher is alive is indistinguishable from a crash: the old
# launcher immediately starts a replacement. Killing children first therefore
# multiplies instances rather than reducing them -- observed live, four robots
# talking over each other after a few restarts, each with its own supervisor
# faithfully keeping it alive. Stop the supervisors, then the children.
$launchers = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like '*start_reachy*' -and $_.ProcessId -ne $PID })
if ($launchers.Count -gt 0) {
    Write-Host "Stopping $($launchers.Count) other launcher(s) first." -ForegroundColor Yellow
    $launchers | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*main.py*' })
if ($running.Count -gt 0) {
    Write-Host "Already running (PID $($running.ProcessId -join ', ')) -- stopping it." -ForegroundColor Yellow
    $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

# Whatever is left is unsupervised and should be gone. If it is not, something
# is restarting it that this script does not know about, and starting anyway
# would put a second voice in the room -- so say so and stop.
$stillRunning = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*main.py*' })
if ($stillRunning.Count -gt 0) {
    Write-Host ""
    Write-Host "Could not stop $($stillRunning.Count) running instance(s): PID $($stillRunning.ProcessId -join ', ')." -ForegroundColor Red
    Write-Host "Starting now would make the robot talk over itself. Close them and retry." -ForegroundColor Red
    exit 1
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
# The address this script just found, handed to the app so it connects to the
# same robot the checks above passed against. Without this the app falls back
# to the address committed in config.py, which is a record of a robot on a
# different network on a different day -- exactly the failure that presents as
# a bare "TimeoutError" from inside the media stack with nothing to explain it.
if (-not $OnRobot -and $RobotIp) { $env:REACHY_HOST = $RobotIp }
# Likewise the port that was actually found answering, so the app and this
# script agree about which daemon they are talking to.
if (-not $OnRobot -and $daemonPort) { $env:REACHY_DAEMON_PORT = "$daemonPort" }
$env:PYTHONIOENCODING = "utf-8"   # replies contain curly quotes; cp1252 mangles them

# Deliberately NOT `python ... 2>&1 | Tee-Object`: redirecting a native
# command's stderr inside PowerShell wraps each line in a NativeCommandError,
# so a Python traceback arrives as an unreadable .NET error blob with the
# actual message truncated. Let Python write both streams to the log itself
# and tail that instead -- the traceback then survives verbatim.
# Exit code 3 means the robot connection died in a way the app cannot repair
# in-process (see body/voice_loop.py). Relaunching is the fix, and doing it
# here beats leaving a frozen robot that still listens and answers.
#
# Any other non-zero code is relaunched too. Only an allow-list of two was
# restarted before, so anything unexpected -- an unhandled exception exiting
# with code 1, a crash in a native library -- left the robot dead in front of
# whoever was standing there, which is the one outcome this loop exists to
# prevent. Exit code 0 is a deliberate stop (Ctrl+C, "go to sleep") and is the
# only thing that ends the loop.
$LINK_LOST = 3
$restarts = 0

while ($true) {
    $py = Start-Process -FilePath $python -ArgumentList "-u", "main.py" `
        -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $logPath -RedirectStandardError $errPath

    if ($py.ExitCode -eq 0) { break }

    if ($py.ExitCode -eq $LINK_LOST) {
        Write-Host ""
        Write-Host "Lost the connection to the robot. Restarting..." -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "Reachy stopped unexpectedly (exit $($py.ExitCode)). Restarting..." -ForegroundColor Yellow
        if ((Test-Path $errPath) -and (Get-Item $errPath).Length -gt 0) {
            Get-Content $errPath -Tail 12
        }
    }

    # Back off as failures repeat, so a fault that recurs immediately doesn't
    # become a relaunch loop hammering the daemon. Capped so a robot that
    # recovers later still comes back without someone re-running the script.
    $restarts += 1
    $delay = [Math]::Min(5 * $restarts, 30)
    Start-Sleep -Seconds $delay
    try {
        Invoke-RestMethod -Method Post -Uri "http://${RobotIp}:${daemonPort}/api/media/acquire" -TimeoutSec 5 | Out-Null
    } catch { }
}

if ((Test-Path $errPath) -and (Get-Item $errPath).Length -gt 0) {
    Write-Host ""
    Write-Host "--- stderr ---" -ForegroundColor Yellow
    Get-Content $errPath -Tail 40
}
