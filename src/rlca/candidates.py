"""Stage 3 of the candidate pipeline: decide which elements go into the model's option list.

Window elements go in whole. Menu-bar items are many and mostly irrelevant, so a
cheap static-embedding score keeps only the top `menu_k`. A "none" option is
always appended so the model can say the target is not on screen.
"""
import re

import numpy as np

from .elements import Element

NONE_OPTION = "None of these elements"
MAX_OPTIONS = 255            # Kev's per-question limit
EMBED_MODEL = "minishlab/potion-retrieval-32M"
_TOKEN = re.compile(r"[a-z0-9]+")


class MenuPrefilter:
    def __init__(self, model_name=EMBED_MODEL):
        from model2vec import StaticModel
        self.model = StaticModel.from_pretrained(model_name)

    def scores(self, command, elements):
        if not elements:
            return np.zeros(0)
        vecs = self.model.encode([command] + [e.describe() for e in elements])
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        cos = vecs[1:] @ vecs[0]
        words = set(_TOKEN.findall(command.lower()))
        overlap = np.array([len(words & set(_TOKEN.findall(e.label.lower()))) / max(len(words), 1) for e in elements])
        return cos + 0.5 * overlap


def build_candidates(command, elements, prefilter, menu_k=40):
    """-> list of Element (window elements, then the top menu items), capped so options fit Kev's limit."""
    window = [e for e in elements if e.source == "window"]
    menu = [e for e in elements if e.source == "menubar" and e.enabled]
    budget = max(0, min(menu_k, MAX_OPTIONS - 1 - len(window)))
    ranked = [menu[i] for i in np.argsort(-prefilter.scores(command, menu))[:budget]] if menu else []
    return (window + ranked)[: MAX_OPTIONS - 1]


def load_elements(snapshot):
    return [Element(**{**e, "frame": tuple(e["frame"]) if e.get("frame") else None}) for e in snapshot["elements"]]
