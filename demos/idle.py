"""Awake, listening, and starting nothing.

Idle is what an operator selects when the robot is on a stand during a talk, in
shot for a photograph, or sitting on the table while someone explains it to a
group -- any moment where the robot speaking or moving on its own initiative
would be an interruption. Every other demo in this folder takes the initiative
somewhere: greeting whoever walks past, dancing, working through a script,
asking a visitor a question. Exactly one has to take none, or there is no
answer to "make it stop without switching it off".

WHY AN ALMOST EMPTY FILE IS THE POINT
Reading this, the natural reaction is that it looks unfinished -- a stub nobody
got round to filling in. It is finished. Its whole contract is negative: while
Idle is selected the robot does nothing a person did not ask for, and someone
mid-sentence in front of a room full of industry partners can rely on that
without watching it. A mode that is quiet *most* of the time is not a mode you
can select while the Dean is speaking; the one occasion it chirps up is the
occasion that matters. So anything added here -- a periodic glance around, an
occasional "I'm still here", a hello when a face appears -- would not extend
Idle. It would delete the only mode whose promise is silence, and there is
nowhere else for that promise to go.

Those behaviours already have homes: the greeter greets, the storyteller tells
stories, conversation converses. And if you want a quiet-but-not-silent mode,
that is a new file beside this one, which costs nothing -- discovery is "what
is in the folder" (see demokit/registry.py), so a second demo is added, not
traded for this one.

IDLE IS NOT ASLEEP
Sleep is a core state rather than a demo: "goodbye" or "turn off" makes the
robot announce it is going quiet and then ignore everything but the wake word
(demokit/runner.py). Idle stays awake and useful. It returns a listening window
every slice, and because it claims no utterances, defines no triggers and
handles nothing, the runner's ordinary dispatch applies to whatever follows the
wake word: another demo's trigger phrase switches to that demo, and anything
else is answered conversationally. That is what makes Idle usable mid-event --
a speaker can hand the room back to the robot by voice, with nobody walking
over to the laptop.
"""

from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S


class Idle(Demo):
    label = "Idle"
    help = "Sits quietly. Wakes on \"Hey Reachy\", starts nothing by itself."
    #: Sorted after the demos an operator actually reaches for, and pointedly
    #: not first: REGISTRY.default_id() is the first *available* demo by order,
    #: and that is where the runner puts a demo it has just set aside after
    #: repeated failures. Recovering from a fault into a robot that says nothing
    #: at all would look like a worse fault than the one being recovered from.
    order = 90
    #: No trigger phrases, and no phrasing would be safe to add. A visitor
    #: saying "quiet" in the middle of a sentence must not take the robot away
    #: from whatever an operator deliberately selected, and the phrase that
    #: genuinely means stop is already core (runner.SLEEP_PHRASES).
    triggers = ()
    #: Nothing, so this can never be greyed out. It is where an operator goes
    #: when something else misbehaves, and a refuge that is itself conditional
    #: on the camera working is not a refuge.
    requires = ()
    #: Must stay False. Claiming utterances would put the quiet mode *ahead of*
    #: every other demo's trigger phrase in the runner's dispatch order, so the
    #: one demo that does nothing would be the one intercepting the room.
    claims_utterances = False

    def on_enter(self, ctx: DemoContext) -> None:
        """Selected. Deliberately silent -- see the module docstring.

        Idle is chosen *because* the robot should stop being the thing in the
        room, so announcing itself would be exactly the interruption the
        operator pressed the button to avoid. The dashboard gets a written note
        instead, which is enough to see from across the room why it went quiet.
        """
        ctx.status("Idle: listening only, nothing on its own.")
        # Whatever ran before may have left the head holding an emotion pose,
        # which the motion loop keeps composing around indefinitely -- and the
        # downward-tilted ones (thinking, sad) leave it looking sulky for the
        # rest of a talk. Resetting the base pose eases it back to level over
        # about a second (_lerp_pose in body/motion.py). Reached through
        # ctx.motion rather than said, because nothing here may be audible.
        ctx.motion.express("neutral")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        """Hand the slice straight back to the core as a listening window.

        The core's maximum rather than a smaller number, and asking for it by
        name rather than repeating 3.0 here. A shorter window would not improve
        wake-word detection: capture is continuous and the spotter's stream and
        mic backlog survive across calls, so a phrase spoken across a window
        boundary is still matched (AudioIO.wait_for_wake_word documents the bug
        this fixed -- rebuilding per call made the wake word land about one time
        in five). All a shorter window buys is more trips round the loop, each
        re-polling demo availability, on a laptop that is also holding a
        language model. Idle has nothing to do between windows, so it asks for
        the longest one the operator's patience can be held to -- the cap exists
        so a dashboard press always lands within a few seconds.

        Note what is not here: no ctx.status. This runs every few seconds for as
        long as a talk lasts, and ctx.status appends unconditionally to a
        200-event ring buffer, so one line per slice would push every heard,
        said and error event off the dashboard within minutes. (state.note
        dedupes consecutive repeats and would survive that, but Idle has nothing
        new to report anyway, which is the whole idea.)
        """
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    # No on_utterance: inheriting the base's "not handled" is what lets every
    # trigger phrase and every question fall through to the runner. Overriding
    # it to return False would behave identically today and invite someone to
    # put something in it tomorrow.
    #
    # No on_exit either: nothing was started, so there is nothing to stop.
