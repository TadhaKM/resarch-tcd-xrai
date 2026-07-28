"""The main loop: body captures a turn, brain decides the reply, body acts it out.

Models (STT/KWS/TTS) are loaded once in AudioIO.__init__, not per turn -- run_once
takes already-constructed components so run_forever can reuse them across turns.
"""

from brain.interface import get_reply
from config import HardwareTarget

from .audio_io import AudioIO
from .camera import Camera
from .face import FaceIdentifier
from .motion import MotionController


def run_once(audio: AudioIO, camera: Camera, face: FaceIdentifier, motion: MotionController) -> None:
    """Run a single wake -> listen -> think -> speak/express turn."""
    audio.wait_for_wake_word()

    frame = camera.get_frame()
    person_id, active_face, _score = face.identify(frame, force=True)
    if active_face is not None and frame is not None:
        motion.track_face(active_face.bbox, frame.shape)
    message = audio.listen()

    if person_id is None and active_face is not None:
        motion.express("curious")
        audio.speak("I don't think we've met yet. What's your name?", "curious", motion=motion)
        name = audio.listen()
        person_id = face.enroll(name, active_face)

    if person_id is None:
        person_id = 0

    motion.express("thinking")
    reply_text, emotion_tag = get_reply(person_id, message)

    motion.express(emotion_tag)
    audio.speak(reply_text, emotion_tag, motion=motion)


def run_forever(target: HardwareTarget) -> None:
    """Run the voice loop until interrupted."""
    audio = AudioIO(target)
    camera = Camera(target)
    face = FaceIdentifier(target)
    motion = MotionController(target)

    try:
        while True:
            run_once(audio, camera, face, motion)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
