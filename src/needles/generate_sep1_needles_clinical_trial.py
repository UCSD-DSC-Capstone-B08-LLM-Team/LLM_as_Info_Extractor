import random
import pandas as pd

# Inclusion terms for Clinical Trial
INCLUSION_TERMS = [
    "clinical trial",
    "experimental study",
    "research study",
    "investigational treatment",
    "randomized controlled trial",
    "phase III trial",
    "study protocol",
]

# Conditions matching the measure set
MATCHING_CONDITIONS = [
    "sepsis",
    "severe sepsis",
    "septic shock",
    "sepsis syndrome",
    "sepsis-3 criteria",
]

# Trial interventions
INTERVENTIONS = [
    "investigational drug",
    "experimental antibiotic",
    "novel therapeutic",
    "study medication",
    "experimental protocol",
    "investigational device",
    "randomized treatment assignment",
]

# Documentation sources
DOCUMENTATION_SOURCES = [
    "signed consent form",
    "research consent",
    "informed consent",
    "clinical trial consent",
    "study enrollment form",
]

# Time references
TIME_REFERENCES = [
    "during this hospital stay",
    "today",
    "on admission",
    "prior to arrival (continued participation)",
    "at time of sepsis diagnosis",
]

# Note templates (POSITIVE cases - where LLM SHOULD say Yes)
POSITIVE_TEMPLATES = [
    "Signed consent form for {term} studying patients with {condition}.",
    "Patient enrolled in {term} for {condition} - consent signed {time_ref}.",
    "Per {source}, patient participating in {term} evaluating {intervention} for {condition}.",
    "{source} documents patient enrollment in {term} - study population: {condition}.",
    "Patient continues active participation in {term} for {condition} (enrolled prior to arrival).",
    "Research consent signed: {term} investigating {intervention} in {condition} patients.",
]

# NEGATIVE cases - where LLM SHOULD say No
# These include observational studies, unclear study types, or wrong conditions

# Observational studies (not clinical trials)
OBSERVATIONAL_TERMS = [
    "observational study",
    "registry",
    "data collection only",
    "patient registry",
    "cohort study",
    "prospective observational study",
    "biobank consent",
]

# Unacceptable contexts
UNACCEPTABLE_CONTEXTS = [
    "observational study only",
    "unclear if experimental or observational",
    "study type not specified",
    "unclear which study population",
    "data collection registry",
]

# Wrong condition options
WRONG_CONDITIONS = [
    "diabetes",
    "hypertension",
    "COPD",
    "heart failure",
    "pneumonia (non-sepsis)",
    "post-surgical recovery",
    "oncology (solid tumors)",
]

NEGATIVE_TEMPLATES = [
    # Observational studies
    "Signed patient consent form for {observational} - no intervention.",
    "Patient enrolled in {observational} with data collection only.",
    "{source} for {observational} - study participants observed, not allocated to treatment groups.",
    
    # Unclear cases
    "{source} - unclear if experimental or observational.",
    "Patient enrolled in study - {unacceptable_context}.",
    "Research consent signed - {unacceptable_context}.",
    
    # Wrong condition cases
    "Signed consent for {term} studying patients with {wrong_condition}.",
    "Patient enrolled in {term} - study population: {wrong_condition}.",
    "Research consent: {term} for {wrong_condition} (not sepsis population).",
    
    # Mixed cases
    "{source} indicates {observational} but also mentions {term} - not clearly experimental.",
]

# needle generator
def generate_clinical_trial_needle():
    # Always generate positive cases only
    template = random.choice(POSITIVE_TEMPLATES)
    return template.format(
        term=random.choice(INCLUSION_TERMS),
        condition=random.choice(MATCHING_CONDITIONS),
        intervention=random.choice(INTERVENTIONS),
        source=random.choice(DOCUMENTATION_SOURCES),
        time_ref=random.choice(TIME_REFERENCES)
    )

def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    return [generate_clinical_trial_needle() for _ in range(n)]


if __name__ == "__main__":
    needles = generate_needle_set(n=100)

    df = pd.DataFrame({
        "DATA_ELEMENT": ["Clinical Trial, Severe Sepsis"] * len(needles),
        "QUERY": [(
            "During this hospital stay, was the patient enrolled "
            "in a clinical trial in which patients with the same condition as the measure set "
            "were being studied? Note: Select 'Yes' ONLY if BOTH of the following are true: "
            "1. There is a signed consent form for a clinical trial (experimental study with "
            "treatment/intervention assignment, often randomized). "
            "2. The consent form documents that during this hospital stay the patient was enrolled "
            "in a clinical trial studying patients with sepsis, severe sepsis, or septic shock. "
            "Select 'No' for observational studies (registries, data collection only), unclear "
            "study types, clinical trials studying conditions other than sepsis, or if no documentation."
        )] * len(needles),
        "NEEDLE_TEXT": needles
    })

    df.to_csv("clinical_trial_needles_new.csv", index=False)
    print(f"Generated {len(needles)} needles for Clinical Trial")
    print("\nSample needles:")
    print("-" * 80)
    for i in range(10):
        print(f"{i+1}. {needles[i]}")