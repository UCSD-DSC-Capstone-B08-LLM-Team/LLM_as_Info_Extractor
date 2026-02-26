import random
import pandas as pd

# Severe sepsis documentation types
SEVERE_SEPSIS_TERMS = [
    "severe sepsis",
    "septic shock",
    "severe sepsis with shock",
    "sepsis with shock",
]

# Infection terms
INFECTION_TERMS = [
    "pneumonia",
    "UTI",
    "urinary tract infection",
    "sepsis",
    "abscess",
    "wound infection",
    "cellulitis",
    "bacteremia",
]

# SIRS criteria
SIRS_INDICATORS = [
    "temperature >38.3 C",
    "temperature <36.0 C",
    "heart rate >90",
    "respiration >20",
    "WBC >12,000",
    "WBC <4,000",
]

# Organ dysfunction indicators
ORGAN_DYSFUNCTION = [
    "SBP <90 mmHg",
    "lactate >2 mmol/L",
    "creatinine >2.0 mg/dL",
    "mechanical ventilation",
    "platelets <100,000",
]

# Documentation sources
DOCUMENTATION_SOURCES = [
    "physician note",
    "ED record",
    "progress note",
    "nursing note",
]

# Time references
TIME_REFERENCES = [
    "on arrival",
    "at 08:00",
    "within 6 hours",
    "at presentation",
]

# Positive qualifiers
POSITIVE_QUALIFIERS = [
    "suspected",
    "possible",
    "likely",
    "concern for",
]

# Negative qualifiers
NEGATIVE_QUALIFIERS = [
    "unlikely",
    "ruled out",
    "no evidence of",
]

# Non-infectious causes
NON_INFECTIOUS_CAUSES = [
    "dehydration",
    "seizure",
    "medication",
    "heart failure",
]

# POSITIVE templates - with clear categories
PROVIDER_TERM_TEMPLATES = [
    "{provider} documents {term} {time_ref}.",
    "{provider} notes {term} {time_ref}.",
]

INFECTION_SIRS_ORGAN_TEMPLATES = [
    "Patient meets severe sepsis criteria: {infection}, {sirs}, and {organ_dysfunction} {time_ref}.",
    "Clinical criteria met: {infection} with {sirs} and {organ_dysfunction} {time_ref}.",
]

PROVIDER_INFECTION_TEMPLATES = [
    "{provider} notes suspected severe sepsis due to {infection} {time_ref}.",
]

# NEGATIVE templates - with clear categories
NO_DOC_TEMPLATES = [
    "No documentation of severe sepsis.",
]

PROVIDER_TIME_TEMPLATES = [
    "{provider} notes patient does not have sepsis {time_ref}.",
]

INFECTION_ONLY_TEMPLATES = [
    "Patient has {infection} but no organ dysfunction documented.",
]

SIRS_NONINFECTIOUS_TEMPLATES = [
    "{sirs} due to {non_infectious}, not infection.",
]

RULE_OUT_TEMPLATES = [
    "Initial concern for {infection} later ruled out.",
    "{provider} documents sepsis is unlikely.",
]

UNABLE_DETERMINE_TEMPLATES = [
    "Unable to determine if severe sepsis present.",
]

ORGAN_DYSFUNCTION_ONLY_TEMPLATES = [
    "Organ dysfunction ({organ_dysfunction}) but no infection source identified.",
]

# Combine all templates for random selection
POSITIVE_TEMPLATES = PROVIDER_TERM_TEMPLATES + INFECTION_SIRS_ORGAN_TEMPLATES + PROVIDER_INFECTION_TEMPLATES
NEGATIVE_TEMPLATES = NO_DOC_TEMPLATES + PROVIDER_TIME_TEMPLATES + INFECTION_ONLY_TEMPLATES + SIRS_NONINFECTIOUS_TEMPLATES + RULE_OUT_TEMPLATES + UNABLE_DETERMINE_TEMPLATES + ORGAN_DYSFUNCTION_ONLY_TEMPLATES

# Severe Sepsis needle generator
def generate_severe_sepsis_needle():
    # Always generate positive case
    template = random.choice(POSITIVE_TEMPLATES)
    # Match template to appropriate formatter
    if template in PROVIDER_TERM_TEMPLATES:
        return template.format(
            provider=random.choice(["physician", "APN", "PA"]),
            term=random.choice(SEVERE_SEPSIS_TERMS),
            time_ref=random.choice(TIME_REFERENCES)
        )
    elif template in INFECTION_SIRS_ORGAN_TEMPLATES:
        return template.format(
            infection=random.choice(INFECTION_TERMS),
            sirs=random.choice(SIRS_INDICATORS),
            organ_dysfunction=random.choice(ORGAN_DYSFUNCTION),
            time_ref=random.choice(TIME_REFERENCES)
        )
    elif template in PROVIDER_INFECTION_TEMPLATES:
        return template.format(
            provider=random.choice(["physician", "APN", "PA"]),
            infection=random.choice(INFECTION_TERMS),
            time_ref=random.choice(TIME_REFERENCES)
        )
    else:
        # Fallback
        return f"{random.choice(['physician', 'APN'])} documents {random.choice(SEVERE_SEPSIS_TERMS)} {random.choice(TIME_REFERENCES)}."


# Generate multiple needles
def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    return [generate_severe_sepsis_needle() for _ in range(n)]


if __name__ == "__main__":
    needles = generate_needle_set(n=100)

    llm_prompt = (
        "In this note, is there explicit positive qualifier for an infection? Do not infer or take a likely guess. "
        "Answer Y only if the infectious condition is clearly stated in the note. Do not use documentation of viral, fungal, or parasitic infections. "
        "The following is a list of conditions commonly associated with severe sepsis that are considered infections (this is not an all-inclusive list.): "
        "EXAMPLES: Abscess, Acute abdomen, Acute abdominal infection, Blood stream, catheter infection, Bone/joint infection, C. difficile (C-diff), Chronic obstructive pulmonary disease (COPD) acute exacerbation, Endocarditis, Gangrene, Implantable device infection, Infection, Infectious, Meningitis, Necrosis, Necrotic/ischemic/infarcted bowel, Pelvic Inflammatory Disease, Perforated bowel, Pneumonia, Empyema, Purulence/pus, Sepsis, Septic, Skin/soft tissue infection, Suspect infection source unknown, Urosepsis, Urinary tract infection, Wound infection. "
        "EXAMPLES THAT AREN'T INFECTIONS: Fever, Diarrhea. "
        "POSITIVE QUALIFIERS: Possible, Rule out (r/o), Suspected, Likely, Probable, Differential diagnosis, Suspicious for, Concern for, Suggestive of, Presumed. "
        "NEGATIVE QUALIFIERS: Impending, Unlikely, Doubt, Ruled out, Less likely, Questionable."
    )
    df = pd.DataFrame({
        "DATA_ELEMENT": ["Severe Sepsis Present"] * len(needles),
        "QUERY": [llm_prompt] * len(needles),
        "NEEDLE_TEXT": needles
    })
    
    df.to_csv("severe_sepsis_needles_new.csv", index=False)
    print(f"Generated {len(needles)} needles")
    print("\nSample needles:")
    print("-" * 80)
    for i in range(10):
        print(f"{i+1}. {needles[i]}")