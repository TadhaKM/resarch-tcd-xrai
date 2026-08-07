"""What the robot knows about the AI XR Hub it lives in.

Kept as data in one file rather than scattered through the demos, because
these are claims made out loud to visitors -- industry partners, prospective
students, academic staff -- and a wrong one is worse than a vague one. When
something here changes (a new project, a staff change), this is the only file
that needs editing.

Two shapes, because they are consumed differently:

- The SCRIPTS are spoken close to verbatim, so they are written for the ear:
  short sentences, no bullet-point syntax, no numbers a listener cannot hold.
  Timings assume piper's conversational pace of roughly 150 words a minute.
- GROUNDING is injected into the system prompt for demos that answer questions
  about the Hub. Without it the model invents plausible-sounding detail, which
  is exactly the failure prompts.py already documents for the robot's own
  capabilities -- and inventing a research project in front of the person who
  runs it is a worse version of that.

Deliberately absent: an equipment list. There is no public one, so the robot
says it doesn't know and points at a human rather than guessing at headset
models in front of people who can see the actual shelf.
"""

#: Measured at ~32 seconds through piper, against a 30-second brief. Leads with
#: what the Hub is for rather than what it owns, because the first thing a
#: visitor wants is a reason to care. Trimmed to time: an earlier draft ran 39s,
#: which is long enough for a group to start looking at each other.
WELCOME_SCRIPT = (
    "Welcome to the AI XR Hub, in Trinity Business School. "
    "We advance the responsible use of artificial intelligence and extended "
    "reality, across education, business and research. "
    "The idea isn't to replace human judgement, but to amplify what people are "
    "distinctly good at. "
    "We work in three strands: immersive learning for students, executive education "
    "and innovation labs with partners, and applied research. "
    "Some of that research is me. I'm Reachy Mini, one of the robots being built here."
)

#: A simple technical overview, for visitors who are not engineers. Concrete
#: about what is actually running, because "it's AI" tells nobody anything.
ABOUT_REACHY_SCRIPT = (
    "I'm a Reachy Mini, made by Pollen Robotics. "
    "I have a camera behind my eyes, a microphone array, a speaker, and six motors "
    "in my neck, which is what lets me look around and tilt my head. "
    "When you speak to me, three things happen. A wake word model notices you're "
    "talking to me. Speech recognition turns what you said into text. Then a "
    "language model decides what to say back, and a speech synthesiser gives it a "
    "voice. "
    "I can also see faces and turn to look at whoever is talking. "
    "The interesting part is that all of it can run on a laptop in this room, with "
    "no internet at all."
)

#: The three-strand summary, reused wherever a demo needs it in one breath.
MISSION_SHORT = (
    "The AI XR Hub is a Trinity Business School initiative for the responsible use "
    "of AI and extended reality, built around amplifying human skills rather than "
    "replacing them."
)

#: Structured for anything that wants to render or reason over them rather
#: than speak them. Figures are the published ones.
PEOPLE = (
    {
        "name": "Professor Na Fu",
        "role": "Director",
        "detail": (
            "Professor in Human Resource Management at Trinity Business School, "
            "Fellow of Trinity College Dublin and of the CIPD, and Co-Director of the "
            "Trinity Centre for Digital Business and Analytics. She coordinates "
            "TechConnect and works on LEADSx2030 and WOTAM, and holds Trinity's "
            "Research Excellence and Teaching Excellence awards."
        ),
    },
    {
        "name": "Professor Laura Berry",
        "role": "Manager",
        "detail": (
            "Assistant Professor of Digital Marketing and AI. She researches how "
            "people respond to immersive technology, AI-enabled interactions and "
            "human-robot interaction, and directs the MSc in Digital Marketing "
            "Strategy. She spent over a decade in industry marketing before "
            "academia, including with IDA Ireland and KBC Bank."
        ),
    },
    {
        "name": "Professor Laurent Muzellec",
        "role": "Dean of Trinity Business School",
        "detail": "Championed the public launch of TechConnect.",
    },
)

PROJECTS = (
    {
        "name": "TechConnect",
        "funding": "3 million euro, Horizon Europe",
        "detail": (
            "A nine-partner consortium of universities, SMEs and hospitals, "
            "coordinated by Trinity. Partners include Malardalen University in "
            "Sweden, the Polytechnic University of Madrid, IRYCIS in Spain leading "
            "data management, and Tallaght University Hospital."
        ),
    },
    {
        "name": "LEADSx2030",
        "funding": "2 million euro, 2024 to 2028, Digital Europe Programme",
        "detail": (
            "Work on Europe's advanced digital skills gap, building on the earlier "
            "LEADS project which ran from 2022 to 2024."
        ),
    },
    {
        "name": "WOTAM",
        "funding": "3 million euro, 2026 to 2029",
        "detail": "The newest project, modelling AI-driven working-time reduction.",
    },
)

#: Public engagement, mentioned when someone asks what the Hub does beyond
#: teaching and research.
OUTREACH = (
    "European Researchers' Night",
    "school programmes",
    "the Trinity Access Programme",
    "the Sanctuary University initiative",
)


def _people_block() -> str:
    return "\n".join(f"- {p['name']}, {p['role']}. {p['detail']}" for p in PEOPLE)


def _projects_block() -> str:
    return "\n".join(f"- {p['name']} ({p['funding']}). {p['detail']}" for p in PROJECTS)


#: Injected into the system prompt for demos that field questions about the
#: Hub. The closing instruction is the load-bearing part: a visitor asking
#: something not covered here should get "I don't know, ask a human" rather
#: than a confident invention.
GROUNDING = f"""\
Facts about the AI XR Hub, where you live. Use these when asked; do not invent
anything beyond them.

WHAT IT IS
The AI XR Hub is a strategic initiative of Trinity Business School, advancing the
responsible use of AI and Extended Reality in education, business and research.
It is grounded in human-centred design and ethical technology adoption. Its
mission is to enhance and amplify distinctly human skills, so that the technology
strengthens professional judgement rather than replacing it.

It runs across three strands: immersive learning for students; executive
education and innovation labs for enterprise partners; and applied research.
Hands-on work spans AI, XR and robotics, including building applications for
Pollen Robotics' Reachy platform, which is what you are.
Public engagement includes {", ".join(OUTREACH[:-1])} and {OUTREACH[-1]}.

THE BUILDING
The Hub sits inside the Trinity Business School building: an 80 million euro,
11,400 square metre building opened in 2019. It has an innovation and
entrepreneurial hub, smart classrooms, a 600-seat auditorium and a rooftop
conference room, and runs as a near-zero-energy building with rooftop solar and
recycled rainwater. The Hub's own space is used for VR leadership-scenario
simulations and hands-on AI, XR and robotics build work.

PEOPLE
{_people_block()}

PROJECTS
{_projects_block()}

WHAT YOU DO NOT KNOW
There is no published list of the Hub's equipment, so if you are asked which
headsets or machines are here, say you don't have that detail and suggest asking
Professor Laura Berry or whoever is hosting. Never guess at a specific model,
count, or price. The same goes for any project, partner, person or figure not
listed above: say you don't know rather than inventing something plausible. The
people you are talking to may well be the people who would notice.\
"""
