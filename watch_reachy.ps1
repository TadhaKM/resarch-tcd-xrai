# Live view of what the robot is hearing and saying.
#
#   .\watch_reachy.ps1          # conversation only
#   .\watch_reachy.ps1 -All     # also mic levels, tracking stats, link warnings
#
# Read-only: safe to start, stop, and restart at any time, and safe to run
# alongside start_reachy.ps1 -- it only follows the log file.

param(
    [switch]$All,
    [string]$LogPath = (Join-Path $PSScriptRoot "reachy_live.log")
)

if (-not (Test-Path $LogPath)) {
    Write-Host "No log yet at $LogPath" -ForegroundColor Yellow
    Write-Host "Start the robot first:  .\start_reachy.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "  Watching $LogPath" -ForegroundColor DarkGray
Write-Host "  Ctrl+C to stop (this does not stop the robot)" -ForegroundColor DarkGray
Write-Host ""

# -Tail 0 so it starts from now rather than replaying the whole session.
Get-Content -Path $LogPath -Wait -Tail 0 | ForEach-Object {
    $line = $_
    # Timestamps are the first token; everything after the module name is the
    # message. Showing the raw line would bury the transcript in module paths.
    $time = if ($line -match '^(\d\d:\d\d:\d\d)') { $matches[1] } else { "        " }
    $msg  = $line -replace '^\d\d:\d\d:\d\d\s+\w+\s+[\w\.]+:\s*', ''

    switch -Regex ($line) {
        'Wake word matched' {
            Write-Host ""
            Write-Host "$time  [ WAKE ] $msg" -ForegroundColor Cyan; break
        }
        'hearing:' {
            # Partial results, updated as you speak.
            Write-Host "$time     ... $($msg -replace '^\s*\[.*?\]\s*hearing:\s*','')" -ForegroundColor DarkGray; break
        }
        'Heard:' {
            Write-Host "$time  [ HEARD] $($msg -replace '^Heard:\s*','')" -ForegroundColor White; break
        }
        'Saying \[' {
            if ($msg -match "Saying \[(\w+)\]:\s*(.*)") {
                Write-Host "$time  [ SAYS ] ($($matches[1])) $($matches[2])" -ForegroundColor Green
            }
            break
        }
        'Nothing transcribed|Still nothing' {
            Write-Host "$time  [ ---- ] $msg" -ForegroundColor DarkYellow; break
        }
        'Dance requested|Dancing:|Dance finished' {
            Write-Host "$time  [ DANCE] $msg" -ForegroundColor Magenta; break
        }
        'Shutdown phrase|Shutting down' {
            Write-Host "$time  [ STOP ] $msg" -ForegroundColor Red; break
        }
        'Turn complete|First sentence' {
            if ($All) { Write-Host "$time  [ time ] $msg" -ForegroundColor DarkGray }
            break
        }
        'unrecoverable|Traceback|ERROR' {
            Write-Host "$time  [ ERROR] $msg" -ForegroundColor Red; break
        }
        default {
            if ($All) { Write-Host "$time         $msg" -ForegroundColor DarkGray }
        }
    }
}
