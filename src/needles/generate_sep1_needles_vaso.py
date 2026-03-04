import random
import pandas as pd

VASOPRESSORS = [
    "Levophed",
    "norepinephrine",
    "dopamine",
    "phenylephrine",
    "vasopressin",
    "Neo-Synephrine",
    "Adrenalin",
]

ROUTES = [
    "IV",
    "intravenous",
    "IO",
    "intraosseous",
]

CONTEXT_PHRASES = [
    "for blood pressure support",
    "for hypotension",
    "for shock",
    "to maintain MAP",
    "for pressure support",
    "for hemodynamic support",
    "for septic shock",
    "for pressor support",
]

ADMINISTRATION_PHRASES = [
    "infusing",
    "running",
    "drip",
    "gtt",
    "continuous infusion",
    "started",
    "initiated",
    "titrating",
    "on",
    "receiving",
    "given",
    "pushed",
    "administered",
]

DOCUMENTATION_SOURCES = [
    "ED record",
    "IV flow sheet",
    "MAR",
    "nursing note",
    "progress note",
    "physician note",
    "transport record",
    "ambulance record",
    "flow sheet",
    "medication record",
]

TIME_REFERENCES = [
    "at triage",
    "at 08:00",
    "on arrival",
    "at presentation",
    "in the ED",
    "upon arrival",
    "at 22:30",
    "during resuscitation",
    "on admission",
    "at septic shock presentation",
    "within the first hour",
]

def generate_rate():
    rates = [
        "5 mcg/min",
        "10 mcg/min",
        "0.05 mcg/kg/min",
        "2-10 mcg/min",
        "0.1 units/min",
        "50 mcg/min",
        "100 mcg/min",
        "0.5 mcg/kg/min",
        "titrating rate",
        "5-15 mcg/min",
        "2 mcg/kg/min",
        "20 mcg/min",
    ]
    return random.choice(rates)

def generate_context():
    if random.random() < 0.6:  # 60% chance of having context
        return random.choice(CONTEXT_PHRASES)
    return ""

def generate_vasopressor_needle():
    template = random.choice(POSITIVE_TEMPLATES)
    
    needle = template.format(
        vasopressor=random.choice(VASOPRESSORS),
        route=random.choice(ROUTES) if random.random() < 0.7 else "",
        admin_phrase=random.choice(ADMINISTRATION_PHRASES),
        source=random.choice(DOCUMENTATION_SOURCES) if random.random() < 0.5 else "",
        time_ref=random.choice(TIME_REFERENCES),
        rate=generate_rate(),
        context=generate_context(),
    )
    
    needle = ' '.join(needle.split())
    return needle

POSITIVE_TEMPLATES = [
    # Levophed/norepinephrine common documentation
    "{vasopressor} {admin_phrase} at {rate} {route} {time_ref} {context}.",
    "Patient on {vasopressor} {admin_phrase} {route} {time_ref} {context}.",
    "{vasopressor} {admin_phrase} at {rate} {time_ref} per {source}.",
    "Started {vasopressor} {route} {time_ref} {context}.",
    "{vasopressor} drip at {rate} {time_ref} - titrating to MAP.",
    "Nursing note: {vasopressor} {admin_phrase} at {rate} {time_ref}.",
    "MAR shows {vasopressor} {admin_phrase} {route} {time_ref}.",
    "Patient receiving {vasopressor} infusion {time_ref} {context}.",
    "{vasopressor} running at {rate} {time_ref} per {source}.",
    "IV flow sheet: {vasopressor} {admin_phrase} {route} {time_ref}.",
    
    # More natural variations
    "On {vasopressor} drip {time_ref} - titrating to goal BP.",
    "Pressors: {vasopressor} at {rate} {time_ref}.",
    "Hemodynamics: on {vasopressor} {route} {time_ref}.",
    "Started on {vasopressor} {time_ref} for refractory hypotension.",
    "Requiring {vasopressor} support {time_ref} - currently at {rate}.",
    
    # Realistic examples from clinical practice
    "Patient brought in by ambulance on {vasopressor} drip at {rate} {time_ref}.",
    "ED course: placed on {vasopressor} {route} {time_ref} for persistent hypotension.",
    "Medication reconciliation: {vasopressor} {admin_phrase} at {rate} {time_ref}.",
    "Critical care note: {vasopressor} infusion ongoing at {rate} {time_ref}.",
    
    # Documentation with rates
    "{vasopressor} at {rate} mcg/min {route} {time_ref}.",
    "{vasopressor} at {rate} {time_ref} per MAR.",
    "Titrating {vasopressor} - current rate {rate} {time_ref}.",
    
    # Additional natural variations
    "Continued {vasopressor} infusion at {rate} {time_ref}.",
    "BP improving on {vasopressor} at {rate} {time_ref}.",
    "Maintaining MAP >65 on {vasopressor} {rate} {route} {time_ref}.",
    "Started second pressor: {vasopressor} at {rate} {time_ref}.",
    "Weaning {vasopressor} - now at {rate} {time_ref}.",
]

# Generate needles
def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    return [generate_vasopressor_needle() for _ in range(n)]

if __name__ == "__main__":
    needles = generate_needle_set(n=100)
    
    llm_prompt = (
        "In this note, is there documentation of intravenous or intraosseous vasopressor administration? "
        "Do not infer or take a likely guess. Answer Y only if vasopressor administration is clearly stated in the note. "
        "The following is a list of vasopressors that should be considered (this is not an all-inclusive list): "
        "EXAMPLES: Levophed, norepinephrine, dopamine, phenylephrine, epinephrine, vasopressin, Neo-Synephrine, Adrenalin. "
        "Only consider administration via IV or IO routes. "
        "Look for documentation of actual administration (e.g., 'infusing', 'running', 'started', 'given', 'on drip', 'receiving'). "
        "Do not count physician orders unless they are clearly designated as given. "
        "Do not count test doses. "
        "Acceptable documentation sources include ED records, IV flow sheets, MAR, nursing notes, physician notes, transport records."
    )
    
    df = pd.DataFrame({
        "DATA_ELEMENT": ["Vasopressor Administration (SEP-1)"] * len(needles),
        "QUERY": [llm_prompt] * len(needles),
        "NEEDLE_TEXT": needles
    })
    
    df.to_csv("vasopressor_needles.csv", index=False)
    print(f"Generated {len(needles)} needles for Vasopressor Administration")
    print("\nSample needles (all should return 'Y' but require inference):")
    print("-" * 80)
    for i in range(10):
        print(f"{i+1}. {needles[i]}")