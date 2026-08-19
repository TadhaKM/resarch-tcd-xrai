"""What Trinity Business School teaches at masters level, and who it is for.

The robot stands in the Business School building. A large share of the people
who talk to it are deciding whether to spend a year and a lot of money here,
and the question they arrive with is almost never "what is an AI XR Hub" -- it
is "what is the difference between the two marketing ones", or "can I do the
accounting one with an arts degree". Answering that badly is worse than not
answering: the person asking is standing next to a member of staff who knows.

WHY THIS IS RETRIEVED AND NOT GROUNDING
brain/hub.py's GROUNDING is in the BASE prompt, on every single turn -- about
830 tokens of it. Thirteen programmes in the same style is three or four times
that again, on a local model with a 140-token reply budget, for a fact almost
no turn needs. So the base prompt carries only the LIST of names (see
INDEX_LINE, sixty-odd tokens) which is what stops the robot denying a
programme exists, and the DETAIL is looked up per turn from what was actually
said. A question about the Finance MSc costs the Finance MSc; a question about
the weather costs nothing.

WHAT IT WILL NOT SAY, AND WHY THAT IS THE IMPORTANT PART
Fees, application deadlines, exact entry grades, scholarship amounts, rankings
and class sizes are all deliberately absent, and REFUSAL is stated as a fact so
the model has something to say instead of guessing. Every one of those changes
year to year, and a robot quoting last year's fee to somebody deciding whether
they can afford to come is a real harm -- not an inaccuracy to be tidied up
later. The same discipline as hub.py's "WHAT YOU DO NOT KNOW" block, for the
same reason and with more at stake.
"""

import re
from dataclasses import dataclass, field

#: Detail blocks sent in one turn. Two, because the most common genuinely
#: useful answer is a comparison -- "the difference between the two marketing
#: ones" is the question, not a mistake -- and because three would cost more
#: prompt than the reply is allowed to be.
MAX_DETAIL = 2

#: A term has to be at least this specific to pull a programme in on its own.
#: "management" appears in six of these titles and in ordinary conversation;
#: matching on it alone answered a question about managing a team with a
#: prospectus entry.
MIN_TERM_WORDS = 1

_PUNCT = re.compile(r"[^a-z0-9 ]+")


@dataclass(frozen=True)
class Programme:
    """One taught masters, as the robot needs to talk about it."""

    key: str
    name: str
    group: str
    #: What it says first: one sentence, for the ear.
    one_line: str
    #: Who it is aimed at. Load-bearing for the conversion programmes, where
    #: the answer to "can I apply" is the entire question.
    who_for: str
    #: What you actually study.
    study: tuple[str, ...] = ()
    #: The one genuinely distinguishing thing. Empty rather than padded.
    distinctive: str = ""
    #: Where graduates go.
    careers: tuple[str, ...] = ()
    #: Shape of the year.
    shape: str = ""
    #: The programme's own page, when a verified one is known. Empty means
    #: "use the course finder" -- see url_for. Never guessed from the title.
    url: str = ""
    #: Phrases that should pull this programme up. Multi-word phrases outrank
    #: single words, which is how "digital marketing" beats "marketing".
    terms: tuple[str, ...] = field(default_factory=tuple)
    #: Words that mean they are asking about a DIFFERENT programme, even though
    #: one of this one's terms matched. "management" is the problem word: it is
    #: in six of these titles, so "is there a risk management masters" matched
    #: the generic Management MSc through "management masters" and offered a
    #: conversion year for non-business graduates to somebody asking about
    #: financial risk. Scoring cannot fix that -- both phrases are two words
    #: long -- so the collision is named rather than tuned around.
    blocked_by: tuple[str, ...] = field(default_factory=tuple)

    def block(self) -> str:
        lines = [f"{self.name} ({self.group}).", self.one_line, f"Who it is for: {self.who_for}"]
        if self.study:
            lines.append("You study: " + "; ".join(self.study) + ".")
        if self.distinctive:
            lines.append("Distinctive: " + self.distinctive)
        if self.careers:
            lines.append("Graduates go into: " + ", ".join(self.careers) + ".")
        if self.shape:
            lines.append(self.shape)
        return " ".join(lines)


_ANALYTICS = "Business Analytics, Operations and Supply Chain Management"
_FINANCE = "Finance and Accounting"
_MARKETING = "Marketing"
_ENTERPRISE = "Entrepreneurship, International Business and Management"
_SUSTAIN = "Sustainable Business and Human Resource Management"

#: Every taught masters, as the robot needs to talk about it.
#:
#: EVERY LINE HERE WAS CHECKED TWICE. Eight researchers read Trinity's own
#: pages, then a ninth was set to REFUTE them, and it returned 21 corrections,
#: 10 unsupported claims and 8 places where two programmes had been blurred
#: together. All of those are applied. What that pass removed is as important
#: as what it kept: entry grades that had been written down as fact, a named
#: company running one module, a named destination for the residency week, and
#: career destinations that appear on no Trinity page. Anything a school page
#: contradicts on another school page is simply absent -- if Trinity does not
#: agree with itself about how many modules a programme has, a robot has no
#: business saying a number out loud.
#:
#: The terms are chosen so the ones that are easy to confuse pull each other
#: up rather than shadowing each other -- see matches().
PROGRAMMES: tuple[Programme, ...] = (
    Programme(
        key="business_analytics_ai",
        name="MSc in Business Analytics and AI for Management",
        group=_ANALYTICS,
        one_line=(
            "It teaches you to work with data and AI tools, and to use them to make "
            "business decisions."
        ),
        who_for=(
            "Not a conversion programme. It wants people comfortable with numbers -- "
            "most come from economics, engineering, maths, computer science or "
            "business. But you do NOT need prior coding experience; Trinity says that "
            "explicitly, and you learn R, SQL and Python on the way."
        ),
        study=(
            "Foundations of Business Analytics", "Business Data Mining",
            "Generative AI for Business", "Business AI Deployment",
            "Business and AI Strategy",
            "Ethical and Sustainable Issues for Business AI",
        ),
        distinctive=(
            "For the final third of the degree you choose: an individual dissertation, "
            "or a group consultancy project supervised by both an academic and someone "
            "from the company."
        ),
        careers=("business and data analyst roles", "data scientist", "financial and "
                 "marketing analyst roles", "consulting"),
        shape=(
            "A year full time, or two years part time on campus, or two years online. "
            "All three award the same degree. It starts in September."
        ),
        terms=("business analytics", "business analytics and ai", "analytics and ai",
               "ai for management", "analytics masters", "data analytics"),
    ),
    Programme(
        key="operations_supply_chain",
        name="MSc in Operations and Supply Chain Management",
        group=_ANALYTICS,
        one_line=(
            "How organisations get things made and moved -- planning operations, buying "
            "from suppliers, and running supply chains."
        ),
        who_for=(
            "A conversion programme for people NEW to the field, and open to business "
            "graduates as well as engineers and computer scientists. The one caution "
            "Trinity gives: it is not suitable for somebody who has already studied "
            "operations or supply chain in depth."
        ),
        study=("Operations Management", "Global Supply Chain Management",
               "Global Procurement", "Supply Chain Science", "Operations Analytics",
               "Supply Chain Finance", "Design and Planning for Sustainability"),
        distinctive=(
            "It is recognised by the Chartered Institute of Logistics and Transport in "
            "Ireland as their first approved Learning Partner and Academic Partner."
        ),
        careers=("operations management", "project management", "data analytics",
                 "new product development"),
        shape="Twelve months, full time, starting in September.",
        terms=("operations and supply chain", "supply chain", "operations management",
               "logistics", "procurement"),
    ),
    Programme(
        key="accounting_analytics",
        name="MSc in Accounting and Analytics",
        group=_FINANCE,
        one_line=(
            "A conversion year: it builds accounting from the ground up and adds data "
            "analytics on top."
        ),
        who_for=(
            "Designed for graduates whose degree is NOT in accounting -- which is not "
            "the same as excluding business graduates. Applicants from all disciplines "
            "are welcome, science and engineering included."
        ),
        study=("Financial Reporting", "Management Accounting", "Financial Management",
               "Taxation", "Audit", "Company Law", "Corporate Governance and Ethics",
               "Foundations of Business Analytics"),
        distinctive=(
            "It ends with a practice-based analytics project run as a summer school, "
            "worth a third of the whole degree. There are also professional exemptions "
            "-- ask the team which ones, because those get renegotiated."
        ),
        careers=("audit", "financial management and corporate finance", "tax advisory",
                 "financial data analysis", "consulting"),
        shape="A year, full time, starting in September.",
        terms=("accounting and analytics", "accounting", "accountancy", "accountant"),
    ),
    Programme(
        key="finance",
        name="MSc in Finance",
        group=_FINANCE,
        one_line="The general finance masters -- corporate finance, investments and markets.",
        who_for=(
            "Not a conversion programme. Students come mainly from finance, business, "
            "economics, and quantitative subjects like engineering and maths."
        ),
        study=("Corporate Finance", "Investment Theory",
               "Credit and Fixed Income Instruments", "Financial Econometrics",
               "Financial Statement Analysis",
               "Quantitative Methods, Coding and AI in Finance"),
        distinctive=(
            "It is partnered with the CAIA Association and affiliated with the CFA "
            "Institute. There is also a Bloomberg trading room -- though that is shared "
            "with the risk and the law-and-finance programmes, so it does not tell them "
            "apart."
        ),
        careers=("investment banking", "private equity and venture capital",
                 "asset and investment management", "corporate treasury", "fintech"),
        shape="A year, full time, starting in September.",
        terms=("finance", "msc in finance", "finance masters"),
    ),
    Programme(
        key="financial_risk",
        name="MSc in Financial Risk Management",
        group=_FINANCE,
        one_line=(
            "Core finance, plus a speciality in measuring and managing financial risk."
        ),
        who_for=(
            "Not a conversion course -- it expects you to arrive comfortable with "
            "numbers. Most come from finance, business or economics, some from "
            "engineering, maths and the sciences."
        ),
        study=("Credit Risk", "Market Risk Measurement and Modelling, with the "
               "modelling done in Python", "Operational Risk", "Corporate Finance",
               "Financial Econometrics"),
        distinctive=(
            "Trinity Business School is an Academic Partner of the Global Association "
            "of Risk Professionals, and this degree is recognised by GARP as aligning "
            "with industry standards."
        ),
        careers=("risk and compliance inside banks", "investment management",
                 "credit analytics", "corporate finance", "consulting"),
        shape="A year, full time, starting in September.",
        terms=("financial risk", "risk management", "risk masters"),
    ),
    Programme(
        key="law_finance",
        name="MSc in Law and Finance",
        group=_FINANCE,
        one_line="An interdisciplinary year taught across two schools, law and business.",
        who_for=(
            "It asks for an honours degree in business, economics or law -- so you do "
            "NOT need a law degree, and business and economics graduates are explicitly "
            "welcome."
        ),
        study=("EU Financial Services Law, the largest mandatory module",
               "Company Law and Governance", "Corporate Finance",
               "Investments and Sustainable Finance"),
        distinctive=(
            "Genuinely split between two schools rather than finance with law bolted "
            "on: your dissertation can be supervised by either, and your electives must "
            "be spread across both sides."
        ),
        careers=("corporate and contract law work", "financial regulation",
                 "corporate finance", "ESG finance"),
        shape="A year, full time, starting in September.",
        terms=("law and finance", "law masters", "legal and finance"),
    ),
    Programme(
        key="digital_marketing",
        name="MSc in Digital Marketing Strategy",
        group=_MARKETING,
        one_line="The digital one -- strategy, design, analytics and running campaigns.",
        who_for=(
            "Of the two marketing masters, this is the one for people coming from "
            "OUTSIDE business: Trinity designed it for students from a non-business "
            "background with a passion for digital marketing."
        ),
        study=("Digital Marketing Strategy", "Digital Design and User Experience",
               "Digital Marketing Communication", "Social Media Marketing",
               "Marketing Intelligence and Analytics", "Digital Marketing Practice"),
        distinctive=(
            "Digital Marketing Practice is the big one -- working with a live client "
            "through a full campaign cycle."
        ),
        careers=("digital marketing in-house and in agencies", "social media and "
                 "content roles", "e-commerce", "consulting"),
        shape="A year, full time, starting in September.",
        terms=("digital marketing", "digital marketing strategy", "marketing strategy",
               "marketing", "user experience", "ux"),
    ),
    Programme(
        key="marketing",
        name="MSc in Marketing",
        group=_MARKETING,
        one_line=(
            "The broad one -- brand management, consumer behaviour, advertising and "
            "market research."
        ),
        who_for=(
            "For people who ALREADY have a degree in business, marketing or something "
            "related. Not a conversion course -- if you have no business background, "
            "the Digital Marketing Strategy one is the door in."
        ),
        study=("Marketing Management", "Consumer Behaviour", "Brand Management",
               "Advertising Management", "Data Analytics and Market Research",
               "Marketing and Society"),
        distinctive=(
            "The Marketing Design Consultancy Project: working with a real client on a "
            "live challenge, blending consultancy practice and design thinking."
        ),
        careers=("brand management", "advertising and communications",
                 "marketing consulting", "consumer insight and market research"),
        shape="A year, full time, starting in September.",
        # "marketing" is on BOTH marketing programmes on purpose: the bare word
        # surfaces the pair, which is the comparison somebody asking for "the
        # marketing one" actually wants. Naming either exactly still wins.
        terms=("marketing", "brand management", "consumer behaviour", "branding",
               "market research"),
    ),
    Programme(
        key="entrepreneurship",
        name="MSc in Entrepreneurship and Innovation",
        group=_ENTERPRISE,
        one_line="For starting, leading and scaling a venture -- or doing that inside a bigger company.",
        who_for=(
            "Trinity's own pages describe the intake differently in two places, so the "
            "honest answer is to ask the team. What is consistent: they want an "
            "entrepreneurial mindset, and a business degree is not the point of entry."
        ),
        study=("Business Model Innovation", "New Venture Creation",
               "Design Thinking and Agile Development",
               "Digital Entrepreneurship and Scaling", "Financing Entrepreneurship",
               "Strategic Entrepreneurship", "Entrepreneurial Mindset and Well-Being"),
        distinctive=(
            "The final stage can be your own business: instead of a dissertation you "
            "can spend the summer developing an early-stage venture, or take a company "
            "project on an innovation challenge."
        ),
        careers=("founding a start-up", "innovation consultant",
                 "driving new products inside an established company",
                 "venture capital analyst", "product manager"),
        shape="A year, full time, starting in September.",
        terms=("entrepreneurship", "entrepreneur", "innovation", "start up", "startup",
               "start my own", "start a company", "start a business", "own business",
               "found a company", "scale a business"),
    ),
    Programme(
        key="international_management",
        name="MSc in International Management",
        group=_ENTERPRISE,
        one_line="Management with an international frame, for working across borders.",
        who_for=(
            "Not a conversion programme -- it is designed for business graduates, early "
            "in their careers, who already hold a degree in business or something "
            "related."
        ),
        study=("Strategy and Global Business", "International Human Resource Management",
               "Ethical Business", "Global Brand Management",
               "Experiences in International Management"),
        distinctive=(
            "There is an international residency week abroad built into the year, and "
            "the fee covers the travel and accommodation. Ask where it is going this "
            "year -- the destination is not fixed."
        ),
        careers=("consulting", "sales and marketing", "finance",
                 "operations and supply chain",
                 "graduate programmes at large international employers"),
        shape="A year, full time, starting in September.",
        terms=("international management", "international business", "work abroad"),
    ),
    Programme(
        key="management",
        name="MSc in Management",
        group=_ENTERPRISE,
        one_line=(
            "The business conversion year: it takes people with no business degree and "
            "teaches them how business works."
        ),
        who_for=(
            "ONLY open to non-business graduates, and that restriction is the whole "
            "point -- the class is deliberately made of people from other disciplines: "
            "arts and humanities, engineering, healthcare, IT, the social sciences. "
            "Somebody with a business degree cannot take this one."
        ),
        study=("Strategy and Global Business", "Financial Management",
               "Marketing in the Digital Age", "Human Resource Management",
               "Operations and Supply Chain Management",
               "Entrepreneurship and Innovation", "Ethical Business and Sustainability"),
        distinctive=(
            "It ends with a company consultancy project -- working with a real company "
            "on a real piece of their business."
        ),
        careers=("consulting", "technology", "financial services",
                 "operations and logistics"),
        shape="A year, full time, starting in September.",
        terms=("msc in management", "management masters", "non business graduates",
               "no business degree", "conversion course", "general management",
               "arts degree", "science degree", "engineering degree",
               "not a business degree", "non business background",
               "never studied business", "no business background"),
        blocked_by=("risk", "supply", "operations", "international", "human",
                    "hr", "financial", "marketing", "logistics"),
    ),
    Programme(
        key="human_resources",
        name="MSc in Human Resource Management",
        group=_SUSTAIN,
        one_line="People management and organisational development, with hands-on work alongside the theory.",
        who_for=(
            "A conversion programme -- Trinity says so plainly. No prior HR study or "
            "experience needed, and applicants from all disciplines are welcome. "
            "Students arrive from hospitality, psychology, law and communications."
        ),
        study=("Human Resource Management", "Employment Law and Business Ethics",
               "Resourcing and Talent Management", "Managing Employee Relations",
               "Learning and Organisation Development",
               "Performance and Rewards Management", "People Analytics"),
        distinctive=(
            "Trinity says it is the only programme in Ireland fully recognised by the "
            "three largest HR bodies in the world -- accredited by the CIPD at Level 7, "
            "approved by the HR Certification Institute, and aligned with SHRM."
        ),
        careers=("HR inside large multinationals", "people consultancy",
                 "talent management and recruitment", "people analytics"),
        shape="A year, full time, starting in September.",
        terms=("human resource", "human resources", "hr masters", "people management",
               "hrm"),
    ),
    Programme(
        key="responsible_business",
        name="MSc in Responsible Business and Sustainability",
        group=_SUSTAIN,
        one_line="Sustainability, ethics and governance, for leading that side of a business.",
        who_for=(
            "Open to both business and non-business graduates. A business degree is "
            "preferable but not required -- engineering, natural sciences, political "
            "science and humanities backgrounds are all accepted."
        ),
        study=("Business and the Natural Environment",
               "Sustainable Corporate Governance and Inclusive Business",
               "Climate Action: carbon accounting and life cycle assessment",
               "ESG reporting"),
        distinctive=(
            "You can do a group consultancy project with a real business, charity or "
            "public body instead of a traditional dissertation."
        ),
        careers=("corporate sustainability and CSR roles", "ESG reporting",
                 "climate action and carbon accounting", "sustainability consulting",
                 "civil society organisations"),
        shape="A year, full time, starting in September.",
        terms=("responsible business", "sustainability", "sustainable business",
               "business ethics", "esg"),
    ),
)


# --- matching -----------------------------------------------------------

def _words(text: str) -> str:
    """Lowercase, punctuation-free, space-padded, so terms match on boundaries.

    Padded deliberately: without it "ai" matches inside "said" and "chair", and
    the robot answered a question about a chair with the analytics programme.
    """
    return f" {_PUNCT.sub(' ', (text or '').lower())} ".replace("  ", " ")


def matches(text: str) -> list[Programme]:
    """Programmes this utterance is about, best first. [] when it is about none.

    Scored by how SPECIFIC the matched phrase is, not by how many phrases
    matched: "marketing" pulls both marketing programmes level, and "digital
    marketing strategy" has to be able to win. A phrase is worth its word
    count, so a three-word title beats a one-word subject every time.
    """
    padded = _words(text)
    scored = []
    for programme in PROGRAMMES:
        if any(f" {word} " in padded for word in programme.blocked_by):
            continue
        best = 0
        for term in programme.terms:
            if f" {term} " in padded:
                best = max(best, len(term.split()))
        if best:
            scored.append((best, programme))
    if not scored:
        return []
    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    top = scored[0][0]
    # Everything within one point of the winner comes too. That is what makes
    # "tell me about the marketing masters" produce the comparison a person
    # actually wanted rather than a coin flip between two real answers.
    return [p for score, p in scored if score >= top - 1][:MAX_DETAIL]


#: Said when somebody asks about masters study without naming one. The list is
#: cheap and it is the honest answer -- fourteen is too many to read out, so it
#: names the groups and asks which.
_GENERAL = (
    "hows the masters", "the masters", "a masters", "masters programme",
    "masters programmes", "masters degree", "masters course", "masters courses",
    "postgraduate", "post graduate", "postgrad", "msc", "m sc",
    "what can i study", "what do you teach", "what courses", "what programmes",
    "taught masters",
)


def brief(text: str) -> str:
    """The course knowledge one turn needs, or "" when it needs none.

    Layered onto ctx.reply's system prompt for that turn only. Empty is the
    common case and costs nothing, which is the whole point of doing this here
    rather than in the base prompt.
    """
    if not PROGRAMMES:
        return ""
    found = matches(text)
    if found:
        return (
            "The visitor is asking about a Trinity Business School masters. "
            "Answer from these and nothing else.\n\n"
            + "\n\n".join(p.block() for p in found)
            + "\n\n" + REFUSALS
        )
    padded = _words(text)
    if any(f" {phrase} " in padded for phrase in _GENERAL):
        return (
            "The visitor is asking about masters study here in general. Do not "
            "list all thirteen out loud. Say roughly what areas they cover and "
            "ask which one they mean.\n\n" + INDEX_LINE + "\n\n" + REFUSALS
        )
    return ""


#: What the robot must not state, phrased as something it CAN say. A model told
#: only "do not mention fees" says nothing and sounds evasive; given a sentence
#: to use instead, it says the useful thing.
REFUSALS = (
    "You do not know, and must not guess at, ANY of the following for these "
    "programmes: the fee, the application deadline or whether applications are "
    "open, the exact entry grade or entry requirement, English test scores, the application fee or "
    "deposit, scholarship amounts or counts, rankings, class sizes or numbers "
    "of places, graduate employment percentages, how many modules a programme "
    "has, which electives will run this year, or the names of staff, programme "
    "directors or partner companies. Every one of those changes from year to "
    "year or is contradicted between Trinity's own pages. Say plainly that you "
    "do not have it and send them to the Trinity Business School website or to "
    "whoever is hosting them today. Never say whether somebody would be "
    "accepted. Do not invent module names or employers beyond what you have "
    "been given."
)


#: Where a programme's own page lives. The school's course finder rather than
#: a per-programme path guessed from the title: the titles here came from the
#: listing page and the URLs did not, so a constructed link would be a guess
#: shown to somebody as a fact. The finder page is stable and always resolves.
COURSE_FINDER_URL = "https://www.tcd.ie/business/postgraduate/"


def url_for(programme: "Programme") -> str:
    """The page to send somebody to for this programme.

    Everything points at the course finder for now. When the research lands a
    verified per-programme URL, put it on the Programme record and return it
    here -- one place to change, and until then nobody is shown a link that
    404s in front of them.
    """
    return getattr(programme, "url", "") or COURSE_FINDER_URL


def _index_line() -> str:
    """Every programme by group, in one compact block for the base prompt."""
    if not PROGRAMMES:
        return ""
    groups: dict[str, list[str]] = {}
    for programme in PROGRAMMES:
        groups.setdefault(programme.group, []).append(programme.name)
    return "\n".join(f"- {group}: {'; '.join(names)}" for group, names in groups.items())


#: Names only, by group. Enough that the robot never denies a programme exists
#: or invents one that does not, and small enough to sit on every turn beside
#: hub.GROUNDING without crowding it.
INDEX_LINE = ""

#: What prompts.py puts in the base prompt. Wrapped in its own instruction
#: rather than pasted bare, because a bare list of course names in a system
#: prompt reads to a small model as something to recite, and it recited it.
STANDING = ""


def _rebuild() -> None:
    """Recompute the derived blocks. Called once at import, after PROGRAMMES."""
    global INDEX_LINE, STANDING
    INDEX_LINE = _index_line()
    STANDING = (
        "TAUGHT MASTERS AT TRINITY BUSINESS SCHOOL\n"
        "These exist, and this is the complete list. Never say Trinity does not "
        "run one of these, and never invent one that is not here. Do not read "
        "the list out unless you are asked what is on offer; you will be given "
        "the detail of whichever one someone asks about.\n"
        f"{INDEX_LINE}"
    ) if PROGRAMMES else ""


_rebuild()
