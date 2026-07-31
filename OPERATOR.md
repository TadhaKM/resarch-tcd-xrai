# Running Reachy — the simple guide

No technical knowledge needed. If something goes wrong, the fix is almost
always: close the black window, and double-click **Start Reachy** again.

## One-time setup (someone technical does this once)

1. On the laptop, run `install.ps1` (right-click → Run with PowerShell).
   It installs everything and puts two icons on the desktop.
2. Turn on the laptop's own WiFi hotspot:
   **Settings → Network & internet → Mobile hotspot → On.**
   Give it a name and password and never change them.
3. Connect the robot to that hotspot once (Pollen's WiFi setup — the robot
   broadcasts its own setup network when it can't find a known one; see
   Pollen's documentation).

Because the robot connects to the *laptop's* hotspot, the venue's WiFi never
matters. Nothing needs the internet — everything runs on the laptop.

## Every time you use it

1. Turn the laptop on. Check **Mobile hotspot** is on.
2. Plug in / power on the robot. Give it about a minute.
3. Double-click **Start Reachy** on the desktop.
4. Wait. A black window shows progress, and your browser opens the
   dashboard by itself.
5. **The robot wiggles (looks left, right, nods) when it is ready.**
6. Say **"Hey Reachy"** — its antennas perk up — then ask your question.

One "Hey Reachy" per question. The dashboard lists every phrase it responds
to, shows everything it hears and says, has a volume slider, and lets you
switch what it's doing:

- **Conversation** — answers questions (the normal mode)
- **Dance** — dances until you pick another mode
- **Greeter** — says hello to people it sees
- **Idle** — sits quietly; still wakes if spoken to

Say **"turn off"** or **"goodbye"** to put it to sleep. Any wake phrase
("Hey Reachy") wakes it again.

## On your phone

Connect the phone to the laptop's hotspot, then type the address shown in
the black window (it looks like `http://192.168.137.1:8080`).

## If it's not working

| What you see | What to do |
| --- | --- |
| Robot ignores "Hey Reachy" | Check the dashboard: does text appear when you talk? If the page says **robot offline**, wait a minute — it restarts itself. |
| Robot frozen but still talks | Wait a minute; it fixes itself. |
| Black window closed / no window | Double-click **Start Reachy** again. |
| "UNREACHABLE" in the window | The robot isn't on the hotspot. Check hotspot is on, restart the robot, try again. |
| Nothing works after 3 minutes | Power the robot off and on, close the window, **Start Reachy** again. |

Still stuck? Restart the laptop. Truly stuck? Call the person who set it up.
