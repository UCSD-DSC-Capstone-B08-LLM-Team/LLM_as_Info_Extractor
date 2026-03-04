import random
import pandas as pd

TRIAL_TERMS = [
    "study",
    "trial",
    "research protocol",
    "investigation",
    "research study",
    "protocol",
    "study protocol",
]

EXPERIMENTAL_PHRASES = [
    "randomized",
    "double-blind",
    "placebo-controlled",
    "investigational drug",
    "experimental treatment",
    "treatment arm",
    "randomization",
    "blinded study",
    "controlled trial",
    "phase 3",
    "phase III",
    "investigational protocol",
]

MATCHING_CONDITIONS = [
    "sepsis",
    "severe sepsis",
    "septic shock",
    "sepsis-3 criteria",
    "sepsis syndrome",
    "sepsis patients",
    "septic patients",
]

INTERVENTIONS = [
    "new antibiotic",
    "investigational therapy",
    "study medication",
    "experimental protocol",
    "novel treatment approach",
    "investigational agent",
    "study drug",
    "treatment protocol",
]
DOCUMENTATION_REFERENCES = [
    "consent signed",
    "informed consent",
    "consent form",
    "enrollment form",
    "research consent",
    "study consent",
    "patient consented",
    "consent obtained",
    "enrolled in",
    "participating in",
    "signed up for",
]

TIME_REFERENCES = [
    "today",
    "on admission",
    "during this admission",
    "in the ED",
    "prior to arrival (continued participation)",
    "enrolled previously, continuing",
    "at time of sepsis diagnosis",
    "this hospitalization",
    "upon presentation",
]

ENROLLMENT_CONTEXT = [
    "for sepsis study",
    "for septic shock trial",
    "for severe sepsis research",
    "as part of sepsis protocol",
    "in sepsis investigation",
    "for the sepsis trial",
]

POSITIVE_TEMPLATES = [
    # Basic enrollment documentation
    "Patient {doc_ref} {trial} {context}.",
    "{doc_ref} for {trial} studying {condition} patients.",
    "Patient {doc_ref} in {trial} evaluating {intervention} for {condition}.",
    
    # time references
    "{doc_ref} {time_ref} - {trial} for {condition}.",
    "Patient {doc_ref} {time_ref} to participate in {trial} {context}.",
    
    # intervention details
    "{trial} {context}: patient {doc_ref} to receive {intervention}.",
    "Research consent: {trial} with {intervention} for {condition} - {doc_ref} {time_ref}.",
    
    # Continued participation
    "Patient continues participation in {trial} {context} (enrolled prior to arrival).",
    "Previously enrolled in {trial} {context}, continuing during this admission.",
    
    "Enrolled in {trial} {context} - consent on file.",
    "Consented for {trial} {time_ref} - investigating {intervention} in {condition}.",
    "Study participation: {trial} {context}, {doc_ref} {time_ref}.",
    "Randomized to {intervention} arm of {trial} {context}.",
    
    # sepsis/septic shock
    "Septic shock patient {doc_ref} for {trial} of {intervention}.",
    "Severe sepsis: patient enrolled in {trial} evaluating {intervention}.",
    "Protocol 2025-03: {trial} for {condition} - patient {doc_ref} {time_ref}.",
    
    # Documentation examples
    "Progress note: patient {doc_ref} in {trial} {context}.",
    "Admission orders include {trial} protocol for {condition} - consent signed.",
    "Research team consulted: patient eligible for {trial} {context}, consent obtained.",
    
    # Consent-specific
    "Signed consent form for {trial} studying {condition} with {intervention}.",
    "Informed consent signed: {trial} investigating {intervention} in {condition} patients.",
]

def generate_trial_description():
    if random.random() < 0.7:  # 70% chance of having experimental phrase
        exp = random.choice(EXPERIMENTAL_PHRASES)
        term = random.choice(TRIAL_TERMS)
        return f"{exp} {term}"
    else:
        return random.choice(TRIAL_TERMS)

def generate_intervention():
    if random.random() < 0.6:  # 60% chance of having intervention
        return random.choice(INTERVENTIONS)
    return ""

def generate_context():
    if random.random() < 0.5:  # 50% chance of having context
        return random.choice(ENROLLMENT_CONTEXT)
    return ""

def generate_condition():
    return random.choice(MATCHING_CONDITIONS)

def generate_clinical_trial_needle():
    template = random.choice(POSITIVE_TEMPLATES)
    
    trial_desc = generate_trial_description()
    intervention = generate_intervention()
    context = generate_context()
    condition = generate_condition()
    doc_ref = random.choice(DOCUMENTATION_REFERENCES)
    time_ref = random.choice(TIME_REFERENCES)
    
    needle = template.format(
        trial=trial_desc,
        condition=condition,
        intervention=intervention,
        doc_ref=doc_ref,
        time_ref=time_ref,
        context=context,
    )
    
    # Cleaning
    needle = ' '.join(needle.split())
    needle = needle.replace("  ", " ")
    return needle

# Generate needles
def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    return [generate_clinical_trial_needle() for _ in range(n)]

if __name__ == "__main__":
    needles = generate_needle_set(n=100)
    
    llm_prompt = (
        "In this note, is there documentation that during this hospital stay the patient was enrolled "
        "in a clinical trial in which patients with the same condition as the measure set were being studied? "
        "Do not infer or take a likely guess. Answer Y only if clinical trial enrollment is clearly documented. "
        "A clinical trial is an experimental study where research subjects are assigned a treatment/intervention "
        "(drugs, procedures, devices) and outcomes are measured. Often randomized with control groups. "
        "To select Y, BOTH of the following must be true: "
        "1. There is a signed consent form for a clinical trial (experimental study with intervention assignment). "
        "2. The consent form documents enrollment in a trial studying patients with sepsis, severe sepsis, or septic shock. "
        "Patients may be newly enrolled during this stay OR enrolled prior to arrival with continued participation. "
        "Select N for: "
        "- Observational studies only (registries, data collection, no intervention) "
        "- Unclear if experimental or observational "
        "- Unclear which study population "
        "- Clinical trials studying conditions other than sepsis. "
        "Look for documentation of signed consent, enrollment, or participation in trials related to sepsis."
    )
    
    # Create a list of dictionaries, each representing one row
    data = []
    for needle in needles:
        data.append({
            "DATA_ELEMENT": "Clinical Trial (SEP-1)",
            "QUERY": llm_prompt,
            "NEEDLE_TEXT": needle
        })
    
    # Create DataFrame from the list (this creates one row per entry)
    df = pd.DataFrame(data)
    
    df.to_csv("clinical_trial_needles.csv", index=False)
    print(f"Generated {len(needles)} needles for Clinical Trial")
    print("-" * 80)
    for i in range(min(10, len(needles))):
        print(f"{i+1}. {needles[i]}")