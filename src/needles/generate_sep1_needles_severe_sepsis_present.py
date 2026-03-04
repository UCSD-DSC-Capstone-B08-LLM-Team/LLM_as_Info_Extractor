import random
import pandas as pd

INFECTION_TERMS = [
    "pneumonia",
    "UTI",
    "urinary tract infection", 
    "abscess",
    "wound infection",
    "cellulitis",
    "sepsis",
    "urosepsis",
    "endocarditis",
    "meningitis",
    "C. diff",
    "C. difficile",
    "bone/joint infection",
    "skin/soft tissue infection",
    "catheter infection",
    "empyema",
    "perforated bowel",
    "necrotic bowel",
    "gangrene",
    "pelvic inflammatory disease",
    "blood stream infection",
    "implantable device infection",
    "acute abdominal infection",
    "COPD acute exacerbation",
    "purulence",
    "pus",
]

POSITIVE_QUALIFIERS = [
    "possible",
    "rule out",
    "r/o",
    "suspected",
    "likely",
    "probable",
    "differential diagnosis",
    "suspicious for",
    "concern for",
    "suggestive of",
    "presumed",
]

DOCUMENTATION_SOURCES = [
    "physician",
    "APN",
    "PA",
    "ED record",
    "progress note",
    "nursing note",
    "attending",
    "resident",
]


TIME_REFERENCES = [
    "on arrival",
    "at 08:00",
    "at 14:30",
    "at presentation",
    "in the ED",
    "on admission",
    "",
]

POSITIVE_TEMPLATES = [
    # Provider documents with qualifier
    "{provider} documents {qualifier} {infection} {time_ref}.",
    "{provider} notes {qualifier} {infection} {time_ref}.",
    "{provider} suspects {qualifier} {infection} {time_ref}.",
    "Assessment: {qualifier} {infection}.",
    "Impression: {qualifier} {infection} {time_ref}.",
    "Diagnosis: {qualifier} {infection}.",
    
    # Qualifier as part of phrase
    "There is {qualifier} for {infection} {time_ref}.",
    "{qualifier} {infection} noted {time_ref}.",
    "Plan to treat for {qualifier} {infection}.",
    "Patient presents with symptoms concerning for {infection} {time_ref}.",
    
    # Rule out format
    "Rule out {infection} {time_ref}.",
    "r/o {infection} {time_ref}.",
    "Need to rule out {infection} {time_ref}.",
    
    # Differential diagnosis format
    "Differential diagnosis includes {infection} {time_ref}.",
    "DDx: {infection} {time_ref}.",
    
    # Other qualifier formats
    "CT scan suggestive of {infection} {time_ref}.",
    "Labs suspicious for {infection} {time_ref}.",
    "Clinical picture concerning for {infection} {time_ref}.",
    "Presumed {infection} {time_ref}.",
    "Likely {infection} {time_ref}.",
    "Probable {infection} {time_ref}.",
    
    # Multiple infections
    "{provider} notes {qualifier} {infection} and {qualifier} {random_infection} {time_ref}.",
    
    # With additional context
    "Patient admitted for {qualifier} {infection} {time_ref}.",
    "Treating for {qualifier} {infection} with antibiotics.",
    "Working diagnosis: {qualifier} {infection}.",
]


def generate_positive_needle():
    """Generate a needle with positive qualifier + infection."""
    template = random.choice(POSITIVE_TEMPLATES)
    
    # Handle templates that might need two different infections
    if "{random_infection}" in template:
        return template.format(
            provider=random.choice(DOCUMENTATION_SOURCES),
            qualifier=random.choice(POSITIVE_QUALIFIERS),
            infection=random.choice(INFECTION_TERMS),
            random_infection=random.choice(INFECTION_TERMS),
            time_ref=random.choice(TIME_REFERENCES)
        ).strip()
    # Handle templates with just qualifier and infection
    elif "{qualifier}" in template:
        return template.format(
            provider=random.choice(DOCUMENTATION_SOURCES),
            qualifier=random.choice(POSITIVE_QUALIFIERS),
            infection=random.choice(INFECTION_TERMS),
            time_ref=random.choice(TIME_REFERENCES)
        ).strip()
    else:
        # Templates that have the qualifier built in
        return template.format(
            provider=random.choice(DOCUMENTATION_SOURCES),
            infection=random.choice(INFECTION_TERMS),
            time_ref=random.choice(TIME_REFERENCES)
        ).strip()


def generate_needle_set(n=100, seed=42):
    """Generate n positive needles."""
    random.seed(seed)
    
    needles = []
    for _ in range(n):
        needles.append(generate_positive_needle())
    
    # shuffle to avoid any patterns
    random.shuffle(needles)
    return needles


if __name__ == "__main__":
    # Generate 100 positive needles
    needles = generate_needle_set(n=100, seed=42)
    
    llm_prompt = (
        "In this note, is there an explicit positive qualifier for an infection? Do not infer or take a likely guess. "
        "Answer Y only if the infectious condition is clearly stated in the note. Do not use documentation of viral, fungal, or parasitic infections. "
        "The following is a list of conditions that are considered infections (this is not an all-inclusive list.): "
        "EXAMPLES: Abscess, Acute abdomen, Acute abdominal infection, Blood stream, catheter infection, Bone/joint infection, C. difficile (C-diff), Chronic obstructive pulmonary disease (COPD) acute exacerbation, Endocarditis, Gangrene, Implantable device infection, Infection, Infectious, Meningitis, Necrosis, Necrotic/ischemic/infarcted bowel, Pelvic Inflammatory Disease, Perforated bowel, Pneumonia, Empyema, Purulence/pus, Sepsis, Septic, Skin/soft tissue infection, Suspect infection source unknown, Urosepsis, Urinary tract infection, Wound infection. "
        "EXAMPLES THAT AREN'T INFECTIONS: Fever, Diarrhea. "
        "POSITIVE QUALIFIERS: Possible, Rule out (r/o), Suspected, Likely, Probable, Differential diagnosis, Suspicious for, Concern for, Suggestive of, Presumed. "
        "NEGATIVE QUALIFIERS: Impending, Unlikely, Doubt, Ruled out, Less likely, Questionable."
    )
    
    df = pd.DataFrame({
        "DATA_ELEMENT": ["Infection with Positive Qualifier"] * len(needles),
        "QUERY": [llm_prompt] * len(needles),
        "NEEDLE_TEXT": needles
    })
    
    df.to_csv("severe_sepsis_needles.csv", index=False)
    print(f"Generated {len(needles)} positive needles")
    print("\nSample needles (all should return Y):")
    print("-" * 80)
    for i in range(min(20, len(needles))):
        print(f"{i+1}. {needles[i]}")