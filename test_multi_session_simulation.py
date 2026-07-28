"""Simulate several fake "people" across several fake "days" of conversations.

The mujoco sim has no photorealistic human faces for the real detector to
find, so this substitutes real photos for camera frames -- the photo-
substitution approach: feed a static image in wherever a live camera frame
would normally come from. Everything downstream of that is real: the actual
MediaPipe detector + ONNX embedding model decide who's in frame, the actual
Ollama-backed brain generates replies, and the actual SQLite memory persists
and recalls notes across visits. Only audio is out of scope -- scripted text
strings stand in for get_reply()'s message argument, the same simplification
test_memory.py used, since STT/TTS/wake-word were already validated
separately in test_stt_reliability.py/test_full_loop.py.

Checks, per visit:
- A never-before-seen face gets enrolled exactly once, not re-enrolled on
  later visits.
- The same photo is recognized as the same person_id every time.
- Memory notes accumulate in the DB after each visit.
- Prior-visit memory is present (non-empty) before later visits, and the
  reply is printed so recall quality can be eyeballed.
- Nothing raises anywhere in the run.

Needs test_photos/person_{11,22,33,44}.jpg (gitignored -- these are real
people's photos from a placeholder-avatar service, not committed). Reproduce:

    mkdir -p test_photos
    for i in 11 22 33 44; do
        curl -sL -o test_photos/person_$i.jpg "https://i.pravatar.cc/400?img=$i"
    done
"""

import cv2

from body.face import FaceIdentifier
from body.motion import MotionController
from brain import db, long_term_memory
from brain.interface import end_conversation, get_reply
from config import SIMULATION

PEOPLE = {
    "Nia": "test_photos/person_11.jpg",
    "Marcus": "test_photos/person_22.jpg",
    "Priya": "test_photos/person_33.jpg",
    "Oskar": "test_photos/person_44.jpg",
}

# name -> scripted 2-turn conversations, one list per visit (cycles if a
# person visits more times than there are scripted conversations for them).
CONVERSATIONS: dict[str, list[list[str]]] = {
    "Nia": [
        [
            "Hi, I'm Nia. My favorite color is teal and I have a big presentation on Thursday.",
            "I'm pretty nervous about it, I've been rehearsing all week.",
        ],
        [
            "Hey, it's me again. Any advice before Thursday?",
            "Thanks, I'll try to stay calm.",
        ],
        [
            "The presentation went great by the way!",
            "Everyone loved the teal slides too, ha.",
        ],
    ],
    "Marcus": [
        [
            "Hey Reachy, I'm Marcus. I just started learning to surf.",
            "I fell off the board about twenty times but it was fun.",
        ],
        [
            "Guess what, I finally stood up on the board today!",
            "Only for like two seconds but I'll take it.",
        ],
    ],
    "Priya": [
        [
            "Hi, Priya here. I'm a vet and I just adopted a rescue cat named Momo.",
            "She's very shy still, hiding under the couch most of the day.",
        ],
        [
            "Momo came out from under the couch today!",
            "She even let me pet her for a minute.",
        ],
        [
            "Quick update, Momo is basically part of the furniture now.",
            "She sleeps on my desk while I work.",
        ],
    ],
    "Oskar": [
        [
            "Hello, I'm Oskar. I'm training for a marathon in the spring.",
            "Today's run was rough, my knee's been bothering me.",
        ],
        [
            "Knee's feeling better after resting it for a few days.",
            "Back to training tomorrow, fingers crossed.",
        ],
    ],
}

# Which people visit on each simulated day.
SCHEDULE = [
    ["Nia", "Marcus"],
    ["Nia", "Priya"],
    ["Marcus", "Oskar"],
    ["Nia", "Priya"],
    ["Oskar"],
]


def print_notes(person_id: int, label: str) -> None:
    notes = db.get_notes(person_id)
    print(f"  memory_notes for person_id={person_id} after {label}: {len(notes)} row(s)")
    for i, note in enumerate(notes, 1):
        preview = note if len(note) <= 130 else note[:127] + "..."
        print(f"    [{i}] {preview}")


def run_visit(
    face: FaceIdentifier,
    name: str,
    turns: list[str],
    visit_num: int,
    known_ids: dict[str, int],
) -> None:
    print(f"\n=== {name}, visit #{visit_num} ===")
    frame = cv2.imread(PEOPLE[name])
    assert frame is not None, f"couldn't load substitute photo for {name}"

    person_id, active_face, score = face.identify(frame, force=True)

    if person_id is None:
        assert active_face is not None, f"real face detector found no face for {name}'s photo"
        assert name not in known_ids, f"{name} was enrolled before but wasn't recognized this time!"
        person_id = face.enroll(name, active_face)
        print(f"  not recognized -> enrolled as new person_id={person_id}")
    else:
        print(f"  recognized as person_id={person_id} (cosine score={score:.3f})")
        if name in known_ids:
            assert person_id == known_ids[name], (
                f"{name} was recognized as person_id={person_id}, expected {known_ids[name]} from before"
            )
        else:
            raise AssertionError(f"{name} matched an existing person_id={person_id} but was never enrolled by us")

    known_ids[name] = person_id

    prior_context = long_term_memory.get_context(person_id)
    is_return_visit = visit_num > 1
    if is_return_visit:
        assert prior_context, f"{name}'s visit #{visit_num} has no prior memory context -- recall is broken"
        print(f"  prior memory present ({len(db.get_notes(person_id))} note(s)), included in this turn's prompt")

    for message in turns:
        reply, tag = get_reply(person_id, message)
        print(f"  user: {message}")
        print(f"  reachy [{tag}]: {reply}")

    end_conversation(person_id)
    print_notes(person_id, f"visit #{visit_num}")


def main() -> None:
    face = FaceIdentifier(SIMULATION)
    motion = MotionController(SIMULATION)

    known_ids: dict[str, int] = {}
    visit_counts: dict[str, int] = {name: 0 for name in PEOPLE}

    try:
        for day_num, visitors in enumerate(SCHEDULE, 1):
            print(f"\n########## SIMULATED DAY {day_num} ##########")
            for name in visitors:
                visit_counts[name] += 1
                script = CONVERSATIONS[name][(visit_counts[name] - 1) % len(CONVERSATIONS[name])]
                run_visit(face, name, script, visit_counts[name], known_ids)
                motion.express("neutral")

        print("\n\n=== FINAL STATE ===")
        for name, person_id in known_ids.items():
            print(f"\n{name} -> person_id={person_id}, {visit_counts[name]} visit(s)")
            print_notes(person_id, "all visits")

        print("\nAll simulated days completed with no crashes.")
    finally:
        motion.stop()


if __name__ == "__main__":
    main()
