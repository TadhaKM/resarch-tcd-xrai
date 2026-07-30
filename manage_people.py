"""List, enroll, and delete face/person records."""

import argparse

from brain import db


def enroll_person(name: str) -> None:
    """Enroll whoever is in front of the camera under a typed name.

    Enrollment lives here rather than in the voice loop because the name has
    to be right: it is stored alongside a face embedding, so a wrong one makes
    the robot confidently greet someone by the wrong name until the row is
    deleted by hand. Asking mid-conversation meant transcribing a name from
    noisy room audio, which produced entries like "Have Come" and "Harky
    Henry G". Typed here, it is exact.
    """
    from body.camera import Camera
    from body.face import FaceIdentifier
    from config import default_target

    target = default_target()
    camera = Camera(target)
    try:
        face_id = FaceIdentifier(target)
        frame = camera.get_frame()
        if frame is None:
            print("No camera frame -- is the robot reachable and the camera free?")
            return

        detected = face_id.detect_active_face(frame, force=True)
        if detected is None:
            print("No face detected. Sit in view of the camera and try again.")
            return

        person_id = db.create_person(name.strip())
        db.save_embedding(person_id, face_id.embedding_for_face(detected))
        print(f"Enrolled {name.strip()!r} as person_id={person_id}.")
    finally:
        camera.close()


def list_people() -> None:
    people = db.list_people()
    if not people:
        print("No people enrolled.")
        return

    for person in people:
        name = person["name"] or "(unnamed)"
        embedding = "yes" if person["has_embedding"] else "no"
        print(
            f"{person['person_id']}: {name} "
            f"embedding={embedding} notes={person['note_count']}"
        )


def delete_person(person_id: int) -> None:
    db.delete_person(person_id)
    print(f"Deleted person_id={person_id} embeddings and notes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List enrolled people")

    enroll_parser = subparsers.add_parser(
        "enroll", help="Enroll the face currently in view under a given name"
    )
    enroll_parser.add_argument("name")

    delete_parser = subparsers.add_parser("delete", help="Delete one enrolled person")
    delete_parser.add_argument("person_id", type=int)

    args = parser.parse_args()
    if args.command == "list":
        list_people()
    elif args.command == "enroll":
        enroll_person(args.name)
    elif args.command == "delete":
        delete_person(args.person_id)


if __name__ == "__main__":
    main()
