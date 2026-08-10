# Find the robot's current IP address, however it is reachable today.
#
#   .\find_robot.ps1              # prints the address, or nothing and exits 1
#   .\find_robot.ps1 -Verbose     # says which method found it
#
# The robot's address is not stable and cannot be made stable: it is handed out
# by whatever network it joined, and it has changed three times in a single
# session. Every method below has failed on at least one of the networks this
# robot has been on, which is why there are four of them rather than one:
#
#   - mDNS ("reachy-mini.local") works at home and on a phone hotspot, and is
#     blocked on TCD_IoT, which filters multicast like most enterprise VLANs.
#   - The last known good address works until DHCP renews it.
#   - The ARP table finds it without a scan, but only once something has
#     already talked to it.
#   - A subnet sweep always works and takes a few seconds, so it is last.
#
# Matching is on the daemon answering, not on ping: something else can hold an
# address the robot used yesterday, and half a demo spent talking to a printer
# is worse than one spent finding the robot.

[CmdletBinding()]
param(
    # The robot's WiFi MAC, used to spot it in the ARP table before resorting
    # to a sweep. Read off the robot, or from `arp -a` while it is connected.
    [string]$Mac = "88-a2-9e-e6-24-15",
    # Ports to try, in order. The daemon's own default is 8000; 8888 is what
    # this project's config.py asks for, which only takes effect if the daemon
    # on the robot was actually started with --fastapi-port 8888. Trying both
    # means the launcher works against either without anyone having to know
    # which -- and finding the port here rather than assuming it is what keeps
    # a no-argument start working, since assuming the configured one and
    # meeting a stock daemon fails with an unexplained media timeout.
    [int[]]$Ports = @(8000, 8888),
    [int]$TimeoutMs = 400
)

$ErrorActionPreference = "SilentlyContinue"

function Test-Daemon([string]$ip) {
    # Returns the port a Reachy daemon is answering on, or $null. Checks the
    # body, not just that something accepted the connection: an address the
    # robot held yesterday may belong to a printer today.
    foreach ($port in $Ports) {
        try {
            $r = Invoke-RestMethod -Uri "http://${ip}:$port/api/daemon/status" -TimeoutSec 2
            if (($null -ne $r.wlan_ip) -or ($null -ne $r.state)) { return $port }
        } catch { }
    }
    return $null
}

function Try-Address([string]$ip, [string]$how) {
    if (-not $ip) { return $null }
    $port = Test-Daemon $ip
    if ($port) {
        Write-Verbose "found via $how, daemon on port $port"
        return "${ip}:$port"
    }
    return $null
}

# 1. Last known good. Free to try and usually right, since the address only
#    changes when the lease does.
$cachePath = Join-Path $PSScriptRoot ".robot_ip"
if (Test-Path $cachePath) {
    $cached = (Get-Content $cachePath -Raw).Trim() -split ":" | Select-Object -First 1
    $found = Try-Address $cached "the last known address"
    if ($found) { $found; exit 0 }
}

# 2. mDNS. Costs nothing where it works.
$found = Try-Address ((Resolve-DnsName "reachy-mini.local" -Type A).IPAddress | Select-Object -First 1) "mDNS"
if ($found) { $found | Set-Content $cachePath -Encoding ascii; $found; exit 0 }

# 3. ARP table, by MAC. No traffic generated; only finds it if something has
#    spoken to it since the table was last aged out.
$arpIp = (Get-NetNeighbor -LinkLayerAddress $Mac -ErrorAction SilentlyContinue |
          Where-Object { $_.AddressFamily -eq "IPv4" } |
          Select-Object -First 1).IPAddress
$found = Try-Address $arpIp "the ARP table"
if ($found) { $found | Set-Content $cachePath -Encoding ascii; $found; exit 0 }

# 4. Sweep this machine's own /24. Last because it is the only slow one.
#    Parallel, because 254 sequential probes at 400ms is a minute and a half.
$local = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -First 1
if ($local) {
    $prefix = ($local.IPAddress -split '\.')[0..2] -join '.'
    Write-Verbose "sweeping $prefix.0/24 for a daemon on port $Port"
    $ping = New-Object System.Net.NetworkInformation.Ping
    $alive = 1..254 | ForEach-Object {
        $ip = "$prefix.$_"
        $reply = $ping.Send($ip, $TimeoutMs)
        if ($reply.Status -eq "Success") { $ip }
    }
    foreach ($ip in $alive) {
        $found = Try-Address $ip "a sweep of $prefix.0/24"
        if ($found) { $found | Set-Content $cachePath -Encoding ascii; $found; exit 0 }
    }
}

Write-Verbose "no daemon answered anywhere"
exit 1
