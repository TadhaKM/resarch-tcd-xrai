"""List and delete enrolled face/person records."""

import argparse

from brain import db


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

    delete_parser = subparsers.add_parser("delete", help="Delete one enrolled person")
    delete_parser.add_argument("person_id", type=int)

    args = parser.parse_args()
    if args.command == "list":
        list_people()
    elif args.command == "delete":
        delete_person(args.person_id)


if __name__ == "__main__":
    main()
