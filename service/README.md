# Process supervision: systemd service + heartbeat watchdog

Two independent recovery mechanisms, layered because they catch different
failure modes:

- **`reachy-supervised.service`** -- `Restart=on-failure`, `RestartSec=5`,
  `StartLimitIntervalSec=60` / `StartLimitBurst=5`. Catches any failure where
  the process actually *exits* (crash, `SIGKILL`, uncaught exception). Gives
  up (stops restarting) if it fails more than 5 times in 60 seconds, so a
  persistently broken build doesn't spin forever.
- **`reachy-watchdog.timer`** (fires every 15s) → **`reachy-watchdog.service`**
  (oneshot) → `watchdog_check.py`. Catches the failure mode `Restart=` *can't*:
  a process that's still running from systemd's point of view but is actually
  hung (deadlock, stuck syscall, blocked read) and doing nothing. The main
  process touches a heartbeat file every 30s; if the watchdog finds it older
  than 90s (3x the interval, to tolerate jitter), it sends the process a real
  `SIGKILL` via `systemctl --user kill`. That's an abnormal exit, so
  `Restart=on-failure` on the main unit takes it from there automatically --
  the watchdog's only job is forcing the exit; it doesn't restart anything
  itself.

## Why this was tested on WSL, not Windows

systemd doesn't exist on Windows. This machine's WSL2 Ubuntu distro already
had real systemd running (confirmed `ps -p 1` reports `systemd`), so that's
where these units were installed and tested (as user units, via
`systemctl --user`, no root needed). WSL2's default networking mode doesn't
bridge to the Windows host's `127.0.0.1`, so `.wslconfig` was set to
`networkingMode=mirrored` (a machine-wide WSL networking change, not scoped
to just this project) so WSL's `localhost:8888` reaches the Windows-side
`reachy-mini-daemon --sim`.

## Why `supervised_process.py` isn't the real `main.py`

The actual `reachy_mini` SDK pulls in PyGObject/pycairo/mujoco, which need
system dev packages (`libcairo2-dev`, GObject introspection headers, a
compiler toolchain, etc.) not present on a minimal WSL image, and installing
them wasn't the point of this exercise. `supervised_process.py` is a
lightweight stand-in that's still genuinely "simulation-connected" -- it
polls the real daemon's `GET /api/daemon/status` every 5s and logs
`simulation_enabled` from the actual response -- without that dependency
chain. It also has a `REACHY_HANG_TEST=1` mode that ignores `SIGTERM` and
spins forever, used to simulate a genuine unresponsive hang (not just a
graceful-shutdown-that-took-a-while).

body/voice_loop.py's actual dependencies (sherpa-onnx, piper-tts, sounddevice)
are lighter and likely *do* have Linux wheels, so pointing `ExecStart` at the
real `main.py` instead may be feasible -- but mic/speaker access from WSL
would need WSLg audio passthrough, which is untested. On the real robot this
whole WSL detour is moot: it's native Linux with real audio hardware, so
`ExecStart` just points at `main.py` directly.

**Before reusing these units on the real robot**, update the hardcoded paths
in `reachy-supervised.service` / `reachy-watchdog.service` (currently
`/home/tmarepal/reachy_supervisor/...`, this test environment's layout) to
wherever the project actually lives, and swap `ExecStart` to the real
`main.py` if/when that's been validated on-device.

## Install (as tested, via `systemctl --user`)

```bash
mkdir -p ~/.config/systemd/user
cp reachy-supervised.service reachy-watchdog.service reachy-watchdog.timer ~/.config/systemd/user/
# adjust ExecStart paths in the .service files first if your layout differs
systemctl --user daemon-reload
systemctl --user enable --now reachy-supervised.service
systemctl --user enable --now reachy-watchdog.timer
```

## Test results (actual run, WSL2 Ubuntu, 2026-07-28)

**Direct `SIGKILL`:**
```
20:07:44  Main process exited, code=killed, status=9/KILL
20:07:44  Failed with result 'signal'
20:07:49  Scheduled restart job, restart counter is at 1   <- exactly RestartSec=5 later
20:07:49  Started reachy-supervised.service
20:07:49  daemon status: state=running simulation_enabled=True   <- reconnected immediately
```

**Hang (`REACHY_HANG_TEST=1`, `SIGTERM` ignored):**
```
t+0s    hung, heartbeat frozen; systemctl reports "active (running)" throughout --
        confirms systemd genuinely cannot detect this on its own
t+10s..t+80s   heartbeat age climbing (17s, 28s, 38s, ... 88s), watchdog logging
               "Heartbeat OK" every ~15s the whole time -- correctly not yet stale
20:10:21  Heartbeat STALE (age=91.4s > 90s) -- force-killing reachy-supervised.service
20:10:21  Sent signal SIGKILL to main process 5546 (python) on client request
20:10:21  Main process exited, code=killed, status=9/KILL
20:10:26  Scheduled restart job, restart counter is at 1   <- RestartSec=5 again
20:10:26  Started reachy-supervised.service
```

Both mechanisms recovered the process correctly and independently: systemd's
own policy for the plain-`SIGKILL` case, and the watchdog specifically for the
case systemd alone couldn't see.
