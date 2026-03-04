import random
import pandas as pd

REFUSAL_TARGETS = [
    "blood draw",
    "labs",
    "blood work",
    "IV",
    "IV fluids",
    "fluids",
    "hydration",
    "antibiotics",
    "IV antibiotics",
    "medications",
    "treatment",
    "further care",
    "staying",
]

REFUSAL_VERBS = [
    "refused",
    "declined",
    "would not allow",
    "does not want",
    "not willing to",
    "unwilling to",
    "declined to consent for",
    "refusing",
    "refuses",
]

ACTORS = [
    "patient",
    "daughter",
    "son",
    "family",
    "wife",
    "husband",
    "family member",
    "power of attorney",
    "POA",
    "healthcare proxy",
    "legal guardian",
    "authorized advocate",
]

DOCUMENTATION_SOURCES = [
    "RN",
    "nurse",
    "physician",
    "MD",
    "APN",
    "PA",
    "provider",
    "nursing",
]

TIME_REFERENCES = [
    "on arrival",
    "at 08:00",
    "upon presentation",
    "in the ED",
    "during initial assessment",
    "at time of triage",
    "within first hour",
    "at septic shock presentation",
    "before fluids initiated",
]

AMA_PHRASES = [
    "left against medical advice",
    "leaving AMA",
    "refusing to stay",
    "declined further care",
    "would not remain for treatment",
    "insisted on leaving",
    "signed out AMA",
    "left before treatment complete",
    "refusing admission",
]

CONTEXT_PHRASES = [
    "after discussion of risks",
    "despite recommendations",
    "per patient preference",
    "after explaining benefits",
    "following shared decision making",
    "due to personal reasons",
    "",
]

POSITIVE_TEMPLATES = [
    # Basic refusal documentation
    "{actor} {verb} {target} {time_ref}.",
    "{source} notes: {actor} {verb} {target} {time_ref}.",
    "{actor} {verb} {target} {context}.",
    
    # With time reference
    "{time_ref}: {actor} {verb} {target}.",
    "Per {source} documentation {time_ref}, {actor} {verb} {target}.",
    
    # With context
    "{actor} {verb} {target} {context} {time_ref}.",
    "{source} documents {actor} {verb} {target} {context}.",
    
    # Specific to IV/fluids
    "Patient {verb} IV access {time_ref}.",
    "{actor} {verb} IV fluids despite recommendation.",
    "Would not allow IV to be started {time_ref}.",
    "Declined antibiotics {time_ref} - will treat symptomatically.",
    
    # Multiple refusals
    "{actor} {verb} {target} and further interventions {time_ref}.",
    "Refusing all {target} at this time.",
    
    # AMA documentation
    "{actor} {ama_phrase} {time_ref}.",
    "{source} documents patient {ama_phrase} {time_ref}.",
    "Patient {ama_phrase} after discussion {time_ref}.",
    "{actor} {verb} to stay for continued care {time_ref}.",
    "AMA form signed: patient leaving {time_ref}.",
    
    # Nursing notes
    "Nursing note: {actor} {verb} {target} {time_ref}.",
    "RN documents {actor} refusing {target} {time_ref}.",
    
    # Physician notes
    "MD note: patient {verb} {target} {time_ref}.",
    "Provider documents refusal of {target} by {actor} {time_ref}.",
]

BLOOD_DRAW_TEMPLATES = [
    "Patient {verb} {blood_test} {time_ref}.",
    "{actor} {verb} {blood_test} - will not consent.",
    "Per {source}, {actor} {verb} {blood_test}.",
]

BLOOD_TESTS = [
    "HIV blood test",
    "arterial blood gas",
    "ABG",
    "blood draw",
    "lab work",
    "blood cultures",
    "lactic acid",
    "CBC",
]

def generate_blood_draw_refusal():
    template = random.choice(BLOOD_DRAW_TEMPLATES)
    return template.format(
        actor=random.choice(ACTORS),
        verb=random.choice(REFUSAL_VERBS),
        blood_test=random.choice(BLOOD_TESTS),
        source=random.choice(DOCUMENTATION_SOURCES),
        time_ref=random.choice(TIME_REFERENCES)
    )

def generate_refusal_needle():
    # 20% chance of using blood draw specific templates (for spec examples)
    if random.random() < 0.2:
        return generate_blood_draw_refusal()
    
    template = random.choice(POSITIVE_TEMPLATES)
    
    # Determine if this is an AMA case (30% chance)
    is_ama = random.random() < 0.3
    
    if is_ama:
        # For AMA cases, use ama_phrase
        return template.format(
            actor=random.choice(ACTORS),
            source=random.choice(DOCUMENTATION_SOURCES),
            time_ref=random.choice(TIME_REFERENCES),
            context="",
            ama_phrase=random.choice(AMA_PHRASES),
            target="",
            verb="",
        ).replace("  ", " ").strip()
    else:
        # For regular refusal cases
        return template.format(
            actor=random.choice(ACTORS),
            verb=random.choice(REFUSAL_VERBS),
            target=random.choice(REFUSAL_TARGETS),
            source=random.choice(DOCUMENTATION_SOURCES),
            time_ref=random.choice(TIME_REFERENCES),
            context=random.choice(CONTEXT_PHRASES) if random.random() < 0.5 else "",
            ama_phrase="",
        ).replace("  ", " ").strip()

# Generate needles
def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    needles = []
    for _ in range(n):
        needle = generate_refusal_needle()
        # Cleaning up
        needle = ' '.join(needle.split())
        needles.append(needle)
    return needles

if __name__ == "__main__":
    needles = generate_needle_set(n=100)
    
    llm_prompt = (
        "In this note, is there documentation that the patient or authorized patient advocate "
        "refused blood draw, IV fluid administration, or IV antibiotic administration within the specified time frame? "
        "Answer Y only if clearly documented by a physician/APN/PA or nurse. "
        "Acceptable documentation includes: patient refused blood draws, IV fluids, or IV antibiotics; "
        "patient left AMA (against medical advice); patient refusing to stay for continued care; "
        "or general documentation of refusal of care that could impact these elements (e.g., pulling out IV). "
        "Do not count refusal of specific blood tests that don't impact SEP-1 requirements (like HIV test). "
        "Authorized advocates include family members, medical power of attorney, legal guardian, or healthcare proxy. "
        "Documentation must be within six hours after Severe Sepsis Presentation Time."
    )
    
    # Create a list of dictionaries, each representing one row
    data = []
    for needle in needles:
        data.append({
            "DATA_ELEMENT": "Administrative Contraindication to Care, Severe Sepsis",
            "QUERY": llm_prompt,
            "NEEDLE_TEXT": needle
        })
    
    # Create DataFrame from the list
    df = pd.DataFrame(data)
    
    df.to_csv("contra_care_needles_new.csv", index=False)
    print(f"Generated {len(needles)} needles for Administrative Contraindication to Care")
    print(f"CSV has {len(df)} rows (one per query+needle combination)")
    print("\nSample needles (all should return 'Y' but require inference):")
    print("-" * 80)
    for i in range(min(10, len(needles))):
        print(f"{i+1}. {needles[i]}")