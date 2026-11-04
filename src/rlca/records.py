"""Kev training records: one labelled request per command, with every question Kev answers in one pass.

The element question's options are rendered exactly as serving renders them (id: description, then the
"none" option), so a trained checkpoint sees the same text at runtime. Questions without a label for a
source are left out of its records rather than guessed (public sets have no irreversible/ambiguous labels).

Record shape is kev.data.load_records': {"state", "questions": {qid: {type, instructions, criteria, label, target?}}}.
"""
from .candidates import NONE_OPTION
from .kev_backend import ELEMENT_QUESTION

NONE_KEY = "none"
ACTIONS = {
    "click": "click it (button, link, row, icon)",
    "select": "select it (tab, radio button, list item)",
    "toggle": "toggle it (checkbox, star, switch)",
    "type": "type text into it",
    "open_menu_item": "open it from the menu bar",
}
TEXT_SOURCES = {
    "none": "no text is entered",
    "user": "the user said the exact words to enter",
    "compose": "the words have to be written for the user",
}
ACTION_Q = "How should the chosen element be operated to carry out the command?"
TEXT_Q = "If the command enters text, where do the words come from?"
IRREVERSIBLE_Q = "Would carrying out this command be hard to undo, or visible to other people (send, post, delete, buy, submit)?"
AMBIGUOUS_Q = "Does the command fit more than one on-screen element equally well, so the user should be asked which one?"


def describe(opt):
    """One option's text, the same for Mac AX elements, OCR/detector items and public-set elements."""
    role = (opt.get("role") or "item").removeprefix("AX").lower()
    ctx = f" · in: {' › '.join(opt['context'][-2:])}" if opt.get("context") else ""
    off = " · disabled" if opt.get("enabled") is False else ""
    return f'{role} "{opt.get("label", "")}"{ctx}{off}'


def element_criteria(options):
    """{id: description, ..., "none": NONE_OPTION} in list order."""
    crit = {o["id"]: describe(o) for o in options}
    crit[NONE_KEY] = NONE_OPTION
    return crit


def state_text(app, command):
    # The "state" layout: best for both sizes in the untrained baseline (the window title pulled answers toward itself).
    return f"App: {app}\nUser command: {command}"


def record(app, command, options, targets, action=None, text_source=None, irreversible=None, ambiguous=None, meta=None):
    """-> one labelled request. targets: acceptable option ids, best first ([] = not on screen)."""
    qs = {"element": {"type": "choice", "instructions": ELEMENT_QUESTION, "criteria": element_criteria(options),
                      "label": targets[0] if targets else NONE_KEY}}
    if len(targets) > 1:
        # Several correct answers share the probability, so picking any of them isn't penalised.
        qs["element"]["target"] = {t: 1.0 for t in targets}
    if targets and action in ACTIONS:
        qs["action"] = {"type": "choice", "instructions": ACTION_Q, "criteria": ACTIONS, "label": action}
    if targets and text_source in TEXT_SOURCES:
        qs["text_source"] = {"type": "choice", "instructions": TEXT_Q, "criteria": TEXT_SOURCES, "label": text_source}
    if irreversible is not None:
        qs["irreversible"] = {"type": "noul", "instructions": IRREVERSIBLE_Q, "label": bool(irreversible)}
    if ambiguous is not None:
        qs["ambiguous"] = {"type": "noul", "instructions": AMBIGUOUS_Q, "label": bool(ambiguous)}
    return {"state": state_text(app, command), "questions": qs, "_meta": meta or {}}
