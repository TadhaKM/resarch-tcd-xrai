"""Test long-term memory end to end: scripted fake conversations across
multiple simulated sessions for several fake person_ids, printing
brain/db.py's memory_notes table after every session so growth (and, for
Alice, consolidation past 15 notes) is visible.

Uses the real Ollama-backed get_reply()/end_conversation() -- no audio, no
daemon needed. Messages are fixed scripted strings standing in for what a
person might say.
"""

from brain import db
from brain.interface import end_conversation, get_reply

PEOPLE = {
    1: "Alice",
    2: "Bob",
    3: "Carol",
    4: "Dave",
}

# A small pool of two-turn scripted conversations, cycled per session so notes
# vary instead of being near-identical each time.
CONVERSATION_POOL = [
    [
        "Hi Reachy, I'm feeling pretty great today!",
        "I just got back from a long hike in the mountains, it was amazing.",
    ],
    [
        "Ugh, work has been so stressful this week.",
        "My boss keeps piling on extra projects and I'm exhausted.",
    ],
    [
        "Can you remind me about my dentist appointment?",
        "It's next Tuesday at 2pm, don't let me forget.",
    ],
    [
        "I've been learning to play the guitar lately.",
        "I'm trying to master this one song by Friday.",
    ],
    [
        "I'm a bit worried about my exam tomorrow.",
        "I studied a lot but I still feel nervous about it.",
    ],
    [
        "Guess what, I adopted a puppy this weekend!",
        "Her name is Luna and she's a golden retriever.",
    ],
]


def print_notes_table(person_id: int, label: str) -> None:
    name = PEOPLE[person_id]
    notes = db.get_notes(person_id)
    print(f"\n--- memory_notes for {name} (person_id={person_id}) after {label} ---")
    if not notes:
        print("  (none)")
    for i, note in enumerate(notes, 1):
        preview = note if len(note) <= 140 else note[:137] + "..."
        print(f"  [{i}] {preview}")
    print(f"  ({len(notes)} row(s))")


def run_session(person_id: int, session_num: int) -> None:
    turns = CONVERSATION_POOL[session_num % len(CONVERSATION_POOL)]
    print(f"\n=== {PEOPLE[person_id]} session {session_num} ===")
    for message in turns:
        reply, tag = get_reply(person_id, message)
        print(f"  user: {message}")
        print(f"  reachy [{tag}]: {reply}")
    end_conversation(person_id)
    print_notes_table(person_id, f"session {session_num}")


def main() -> None:
    for person_id, name in PEOPLE.items():
        db.ensure_person(person_id, name)

    # Alice: enough sessions to cross the >15-note consolidation threshold and
    # see a couple more notes accumulate on top of the resulting profile row.
    for session_num in range(1, 18):
        run_session(1, session_num)

    # Everyone else: a couple of sessions each, to show normal accumulation
    # without hitting consolidation.
    for person_id in (2, 3, 4):
        for session_num in range(1, 3):
            run_session(person_id, session_num)

    print("\n\n=== FINAL STATE: all people ===")
    for person_id in PEOPLE:
        print_notes_table(person_id, "all sessions")


if __name__ == "__main__":
    main()
