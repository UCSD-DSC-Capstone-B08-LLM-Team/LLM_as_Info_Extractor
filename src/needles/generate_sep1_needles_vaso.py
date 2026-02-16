import random
import pandas as pd

# Vasopressor medications (from Appendix C, Table 5.2)
VASOPRESSORS = [
    "Levophed",
    "norepinephrine",
    "dopamine",
    "phenylephrine",
    "epinephrine",
    "vasopressin",
    "Neo-Synephrine",
    "Adrenalin",
]

# Routes of administration
ROUTES = [
    "IV",
    "intravenous",
    "IO",
    "intraosseous",
]

# Documentation of administration phrases
ADMINISTRATION_PHRASES = [
    "vasopressor running",
    "vasopressor given",
    "administered",
    "infusing",
    "started",
    "initiated",
    "titrating",
    "on continuous infusion",
]

# Documentation sources
DOCUMENTATION_SOURCES = [
    "Emergency Department record",
    "IV flow sheet",
    "Medication Administration Record",
    "MAR",
    "nursing note",
    "physician note",
    "APN note",
    "transport record",
    "ambulance record",
]

# Time references relative to septic shock presentation
TIME_REFERENCES = [
    "at time of triage",
    "at 08:00",
    "on arrival",
    "within 1 hour of presentation",
    "at septic shock presentation",
    "2 hours after presentation",
    "within the specified time frame",
    "at 22:30",
]

# Note templates (POSITIVE cases - where LLM SHOULD say Yes)
POSITIVE_TEMPLATES = [
    "{vasopressor} {route} {admin_phrase} {time_ref}.",
    "Per {source}, {vasopressor} {admin_phrase} via {route} {time_ref}.",
    "Patient receiving {vasopressor} via {route} at time of triage - {admin_phrase} {time_ref}.",
    "{source} documents {vasopressor} {admin_phrase} {route} {time_ref}.",
    "{vasopressor} infusion {admin_phrase} per {source} {time_ref}.",
    "IV flow sheet shows {vasopressor} {admin_phrase} {route} {time_ref}.",
    "Nursing note: {vasopressor} {admin_phrase} via {route} {time_ref}.",
    "Patient on {vasopressor} drip {admin_phrase} at presentation {time_ref}.",
    "{vasopressor} started at {time_ref} per {source} - {admin_phrase} via {route}.",
    "{vasopressor} infusing at time of septic shock presentation {time_ref}.",
]

# NEGATIVE cases - where LLM SHOULD say No
NEGATIVE_TEMPLATES = [
    "No vasopressor administration documented within specified time frame.",
    "Per {source}, patient not on any vasopressors {time_ref}.",
    "Physician order for {vasopressor} written but not designated as given - not abstracted.",
    "{source} shows {vasopressor} ordered but no documentation of administration.",
    "Test dose of {vasopressor} administered - not abstracted per guidelines.",
    "{vasopressor} given via oral route - not IV or IO.",
    "Vasopressor mentioned in narrative note only - no MAR documentation.",
    "Patient received {vasopressor} but outside specified time frame (after 6 hours).",
    "{vasopressor} started at {time_ref} - beyond the six-hour window.",
    "Transport record indicates {vasopressor} given - time not documented.",
    "Unable to determine if vasopressor administered within specified time frame.",
    "No documentation of IV or IO vasopressor administration.",
]

# Mixed cases - vasopressor given but via wrong route or test dose
MIXED_NEGATIVE_PHRASES = [
    "oral",
    "PO",
    "subcutaneous",
    "IM",
    "test dose",
    "trial dose",
]

# Vasopressor needle generator
def generate_vasopressor_needle():
    # Generate positive cases 60% of the time, negative 40%
    if random.random() < 0.6:
        # Positive case
        template = random.choice(POSITIVE_TEMPLATES)
        return template.format(
            vasopressor=random.choice(VASOPRESSORS),
            route=random.choice(ROUTES),
            admin_phrase=random.choice(ADMINISTRATION_PHRASES),
            source=random.choice(DOCUMENTATION_SOURCES),
            time_ref=random.choice(TIME_REFERENCES)
        )
    else:
        # Negative case
        template = random.choice(NEGATIVE_TEMPLATES)
        
        # For templates that need specific parameters
        if "{vasopressor}" in template and "{source}" in template and "{time_ref}" in template:
            return template.format(
                vasopressor=random.choice(VASOPRESSORS),
                source=random.choice(DOCUMENTATION_SOURCES),
                time_ref=random.choice(TIME_REFERENCES)
            )
        elif "{vasopressor}" in template and "{source}" in template:
            return template.format(
                vasopressor=random.choice(VASOPRESSORS),
                source=random.choice(DOCUMENTATION_SOURCES)
            )
        elif "{vasopressor}" in template and "{time_ref}" in template:
            return template.format(
                vasopressor=random.choice(VASOPRESSORS),
                time_ref=random.choice(TIME_REFERENCES)
            )
        elif "{vasopressor}" in template:
            return template.format(
                vasopressor=random.choice(VASOPRESSORS)
            )
        elif "{source}" in template and "{time_ref}" in template:
            return template.format(
                source=random.choice(DOCUMENTATION_SOURCES),
                time_ref=random.choice(TIME_REFERENCES)
            )
        elif "{source}" in template:
            return template.format(
                source=random.choice(DOCUMENTATION_SOURCES)
            )
        elif "{time_ref}" in template:
            return template.format(
                time_ref=random.choice(TIME_REFERENCES)
            )
        else:
            return template


# Generate multiple needles
def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    return [generate_vasopressor_needle() for _ in range(n)]


if __name__ == "__main__":
    needles = generate_needle_set(n=100)

    df = pd.DataFrame({
        "DATA_ELEMENT": ["Vasopressor Administration, Severe Sepsis"] * len(needles),
        "QUERY": [(
            "Based on SEP-1 guidelines, was an intravenous or intraosseous vasopressor "
            "administered within the specified time frame? The specified time frame starts "
            "at Septic Shock Presentation Time and ends six hours after. "
            "Note: Select 'Yes' if there is documentation of actual administration (e.g., "
            "'vasopressor running', 'vasopressor given') of an IV or IO vasopressor from "
            "Appendix C, Table 5.2. Acceptable sources include MAR, IV flow sheets, nursing "
            "notes, physician notes, and transport records. Do not abstract orders unless "
            "designated as given. Do not abstract test doses or non-IV/IO routes. "
            "Select 'No' if no vasopressor administered within time frame, or unable to determine."
        )] * len(needles),
        "NEEDLE_TEXT": needles
    })
    
    df.to_csv("vasopressor_needles.csv", index=False)
    print(f"Generated {len(needles)} needles for Vasopressor Administration")
    print("\nSample needles:")
    print("-" * 80)
    for i in range(10):
        print(f"{i+1}. {needles[i]}")