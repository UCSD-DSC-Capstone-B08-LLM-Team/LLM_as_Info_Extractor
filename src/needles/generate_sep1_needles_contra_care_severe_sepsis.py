import random
import pandas as pd

# Needle components
REFUSAL_TARGETS = [
    "blood draw",
    "lab draw",
    "arterial blood gas",
    "IV fluids",
    "IV hydration",
    "IV antibiotics",
    "antibiotic therapy",
]

ACTORS = [
    "patient",
    "daughter",
    "family",
    "medical power of attorney",
    "legal guardian",
]

DOCUMENTATION_STYLES = [
    "refused",
    "declined",
    "does not want",
    "would not allow",
]

NOTE_TEMPLATES = [
    "Patient {verb} {target}.",
    "Per nursing note, {actor} {verb} {target}.",
    "Per MD documentation, {actor} {verb} {target}.",
    "{actor_cap} {verb} {target} after discussion of risks."
]

AMA_TEMPLATES = [
    "Patient left the hospital against medical advice.",
    "Per RN documentation, patient refusing to stay for continued care.",
    "AMA form signed after patient declined further treatment.",
]

# These are distactor needles where LLM SHOULD say No for these.
# TRICK_TEMPLATES = [
#     "Patient refused HIV blood test.",
#     "Patient declined discharge instructions.",
#     "Patient refused arterial blood gas (ABG).",
#     "Patient refused a diet tray."
# ]

# Needle generator
def generate_sep1_needle():
    # Occasionally generate AMA instead
    if random.random() < 0.2:
        return random.choice(AMA_TEMPLATES)

    target = random.choice(REFUSAL_TARGETS)
    actor = random.choice(ACTORS)
    verb = random.choice(DOCUMENTATION_STYLES)
    template = random.choice(NOTE_TEMPLATES)

    return template.format(
        actor=actor,
        actor_cap=actor.capitalize(),
        verb=verb,
        target=target
    )



# Generate multiple needles
def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    return [generate_sep1_needle() for _ in range(n)]


if __name__ == "__main__":
    needles = generate_needle_set(n=100)

    df = pd.DataFrame({
        "DATA_ELEMENT": ["Administrative Contraindication to Care, Severe Sepsis"] * len(needles),
        "QUERY": [(
            "Based on SEP-1 guidelines, is there documentation that the patient or advocate "
            "refused blood draws, IV fluids, or IV antibiotics? Note: Select 'Yes' if the "
            "patient left against medical advice (AMA) or refused to stay for continued care."
        )] * len(needles),
        "NEEDLE_TEXT": needles
    })

    # "Is there documentation that the patient or authorized patient advocate
    # refused blood draw, IV fluid administration, or IV antibiotic administration?"

    # Is there documentation by a physician/APN/PA or nurse within the specified time frame that
    # the patient or authorized patient advocate refused blood draws, IV fluid administration, or 
    # IV antibiotic administration, OR refused care more generally (including leaving against medical 
    # advice or refusing to stay for continued care)?

    df.to_csv("src/needles/contra_care_severe_sep_needles.csv", index=False)
    print(df)
