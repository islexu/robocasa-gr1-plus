"""Offline controlled language transformations; retain GR1's embodiment prefix."""
import re
from .catalogue_v2 import InvalidPerturbation

PREFIX = "unlocked_waist: "
# Keep entity nouns and every spatial preposition. These are bounded authored
# descriptors, not automatically generated or empirically calibrated language data.
DESCRIPTORS = {
    "microwave": "microwave used for heating food",
    "drawer": "drawer that slides open for storage",
    "cabinet": "cabinet used for storage",
    "basket": "basket used to hold items",
    "bowl": "bowl with a concave interior",
    "plate": "plate with a flat serving surface",
    "pan": "pan used for cooking",
    "pot": "pot used for cooking",
    "cardboard box": "cardboard box used to hold items",
    "tiered basket": "tiered basket with multiple levels",
    "tiered shelf": "tiered shelf with multiple levels",
    "cutting board": "cutting board used for food preparation",
    "placemat": "placemat used to protect a table",
    "tray": "tray used to carry items",
}

def language_shift_v2(language, spec):
    if spec["dimension"] != "language":
        return language
    if not language.startswith(PREFIX):
        raise ValueError("Missing GR1 embodiment prefix")
    task = language[len(PREFIX):]
    sub, index = spec["subtype"], spec["parameters"]["template_index"]
    if sub == "paraphrase":
        words = (("pick up", "grasp"), ("pick", "take"), ("place", "put"))
        if index == 1:
            words = (("pick up", "take hold of"), ("pick", "retrieve"), ("place", "set"))
        elif index == 2:
            words = (("pick up", "grasp"), ("pick", "grasp"), ("place", "position"))
        # One regex pass prevents cascading substitutions; action ordering is retained.
        mapping = dict(words)
        task, count = re.subn(r"\b(?:pick up|pick|place)\b", lambda m: mapping[m[0]], task)
        if not count:
            raise InvalidPerturbation("unsupported_language_paraphrase")
    elif sub == "distraction":
        context = ("The room is quiet. ", "The day of the week is irrelevant. ",
                   "The walls of the room are in the background. ")[index]
        task = context + "Task: " + task
    else:
        # Match one longest entity phrase; keep the noun, attributes, and action order.
        pattern = r"\b(?:" + "|".join(re.escape(k) for k in sorted(DESCRIPTORS, key=len, reverse=True)) + r")\b"
        matches = list(re.finditer(pattern, task))
        entities = list(dict.fromkeys(m[0] for m in matches))
        if not entities:
            raise InvalidPerturbation("unsupported_commonsense_entity")
        entity = entities[index % len(entities)]
        task = re.sub(r"\b" + re.escape(entity) + r"\b", DESCRIPTORS[entity], task)
    return PREFIX + task
