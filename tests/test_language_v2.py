import pytest
from robocasa_gr1_plus.catalogue_v2 import make_variant, InvalidPerturbation
from robocasa_gr1_plus.language_v2 import language_shift_v2, PREFIX

@pytest.mark.parametrize("instruction", [
    "pick the red apple from the cutting board and place it in the basket",
    "pick up the cup, place it into the drawer and close the drawer",
    "pick up the milk, place it into the microwave and close the microwave"])
def test_language_changes_preserve_entities_prepositions_and_action_order(instruction):
    for subtype in ("paraphrase", "distraction", "commonsense"):
        out = language_shift_v2(PREFIX + instruction, make_variant("language", subtype))
        assert out.startswith(PREFIX) and out != PREFIX + instruction
        for token in ("red apple", "cup", "milk", "from", "cutting board", "basket", "drawer", "microwave", "into", "in"):
            if token in instruction.split() or token in ("red apple", "cutting board") and token in instruction:
                assert token in out
        if "close" in instruction:
            assert out.index("put") < out.index("close") if subtype == "paraphrase" else out.index("place") < out.index("close")
        if subtype == "distraction":
            assert instruction in out
        if subtype == "paraphrase":
            variants = {language_shift_v2(PREFIX + instruction, make_variant("language", subtype,index=i)) for i in range(3)}
            assert len(variants) == 3

def test_unsupported_instruction_does_not_silently_pass_as_changed():
    with pytest.raises(InvalidPerturbation):
        language_shift_v2(PREFIX + "wait", make_variant("language", "paraphrase"))
