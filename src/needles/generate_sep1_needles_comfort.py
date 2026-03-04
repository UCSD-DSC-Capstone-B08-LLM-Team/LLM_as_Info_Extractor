import random
import pandas as pd

INCLUSION_TERMS = [
    "comfort care",
    "comfort measures",
    "comfort measures only",
    "comfort only",
    "palliative care",
    "palliative",
    "hospice",
    "hospice care",
    "end of life care",
    "terminal care",
    "withdraw care",
    "withhold care",
    "terminal extubation",
]

CLINICAL_SHORTHAND = [
    "CMO",
    "comfort care",
    "palliative",
    "hospice",
]

ACCEPTABLE_CONTEXTS = [
    "recommendation for",
    "order for",
    "plan for",
    "patient requests",
    "family requests",
    "goal of care is",
    "care plan:",
    "plan to pursue",
    "transition to",
    "goals of care discussion - family prefers",
]

# State-authorized portable orders
SAPO_FORMS = [
    "DNR-Comfort Care form",
    "MOLST",
    "POLST",
    "OOH DNR",
    "advance directive",
]

PROVIDERS = [
    "physician",
    "APN",
    "PA",
    "MD",
    "attending",
    "hospitalist",
    "intensivist",
]

DOCUMENTATION_SOURCES = [
    "ED note",
    "H&P",
    "progress note",
    "consult note",
    "physician order",
    "admission note",
    "critical care note",
]

TIME_REFERENCES = [
    "on arrival",
    "at presentation",
    "upon admission",
    "in the ED",
    "within first hour",
    "at 08:00",
    "during initial assessment",
    "",
]

POSITIVE_TEMPLATES = [
    # Basic documentation
    "{provider} {source}: {context} {term} {time_ref}.",
    "Per {provider}, {context} {term} {time_ref}.",
    "{provider} documents {context} {term} {time_ref}.",
    "Goals of care: {context} {term} {time_ref}.",
    
    # With specific phrasing
    "After discussion with family, {provider} {context} {term}.",
    "{provider} recommending {term} - family agrees {time_ref}.",
    "Plan: {term} per {provider} note {time_ref}.",
    
    # Order/consult formats
    "Order placed for {term} consultation {time_ref}.",
    "Consult to {term} service placed {time_ref}.",
    "{provider} ordered {term} protocol {time_ref}.",
    
    # Patient/family request formats
    "Family requesting {term} for patient {time_ref}.",
    "Patient prefers {term} approach {time_ref}.",
    "Spouse requests {term} only {time_ref}.",
    
    # CMO specific
    "Patient is {term} - full support withdrawn {time_ref}.",
    "Decision made for {term} after goals discussion.",
    "Transitioning to {term} effective {time_ref}.",
    
    # SAPO documentation
    "{sapo_form} in chart dated {time_ref} with {term} option selected.",
    "Prior {sapo_form} indicates {term} - on file {time_ref}.",
    "Advance directive: {term} selected on {sapo_form}.",
    
    # Mixed natural examples
    "Palliative care consulted - recommending comfort measures.",
    "Hospice referral placed by attending physician.",
    "Patient to be made CMO per family request.",
    "Withdrawing life support - transitioning to comfort care.",
]

SAPO_TEMPLATES = [
    "{sapo_form} from prior admission with {term} checked.",
    "Patient has {sapo_form} on file indicating {term} preference.",
    "Pre-arrival {sapo_form} documents {term} selection.",
    "Old {sapo_form} in chart: {term} option selected.",
]

def generate_term():
    if random.random() < 0.3:  # 30% chance of using shorthand
        return random.choice(CLINICAL_SHORTHAND)
    return random.choice(INCLUSION_TERMS)

def generate_sapo_needle():
    template = random.choice(SAPO_TEMPLATES)
    return template.format(
        sapo_form=random.choice(SAPO_FORMS),
        term=generate_term(),
        time_ref=random.choice(TIME_REFERENCES) if random.random() < 0.5 else "prior to arrival"
    )

def generate_positive_needle():
    # 20% chance of SAPO case
    if random.random() < 0.2:
        return generate_sapo_needle()
    
    template = random.choice(POSITIVE_TEMPLATES)
    
    # Build the needle
    provider = random.choice(PROVIDERS)
    source = random.choice(DOCUMENTATION_SOURCES) if random.random() < 0.6 else ""
    context = random.choice(ACCEPTABLE_CONTEXTS)
    term = generate_term()
    time_ref = random.choice(TIME_REFERENCES)
    sapo_form = random.choice(SAPO_FORMS) if random.random() < 0.3 else ""
    
    needle = template.format(
        provider=provider,
        source=source,
        context=context,
        term=term,
        term_cap=term.capitalize() if term else "",
        time_ref=time_ref,
        sapo_form=sapo_form,
    )
    
    # Cleaning up
    needle = ' '.join(needle.split())
    return needle

# Generate needles
def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    needles = []
    for _ in range(n):
        needle = generate_positive_needle()
        # Cleaning up
        needle = ' '.join(needle.split())
        needles.append(needle)
    return needles

if __name__ == "__main__":
    needles = generate_needle_set(n=100)
    
    llm_prompt = (
        "In this note, is there physician/APN/PA documentation of comfort measures only, palliative care, "
        "or another acceptable inclusion term within an acceptable context? Answer Y only if clearly documented. "
        "Acceptable contexts include: comfort measures only recommendation, order for hospice consultation, "
        "patient or family request for comfort measures only, plan for comfort measures only, or referral to hospice. "
        "Also accept state-authorized portable orders (SAPOs) like POLST/MOLST with CMO option checked if dated and signed prior to arrival, "
        "unless contradicted by documentation on day of arrival. "
        "Do NOT count documentation that is only discussion/consideration (e.g., 'consider palliative care'), "
        "negative/conditional statements (e.g., 'no comfort care'), or pre-arrival documentation referring to prior admission. "
        "Inclusion terms include: comfort care, comfort measures only, CMO, palliative care, hospice, end of life care, withdraw care, terminal extubation."
    )
    
    # Create a list of dictionaries, each representing one row
    data = []
    for needle in needles:
        data.append({
            "DATA_ELEMENT": "Directive for Comfort Care or Palliative Care, Severe Sepsis",
            "QUERY": llm_prompt,
            "NEEDLE_TEXT": needle
        })
    
    # Create DataFrame from the list
    df = pd.DataFrame(data)
    
    df.to_csv("comfort_care_needles.csv", index=False)
    print(f"Generated {len(needles)} needles for Directive for Comfort Care/Palliative Care")
    print(f"CSV has {len(df)} rows (one per query+needle combination)")
    print("\nSample needles (all should return 'Y' but require inference):")
    print("-" * 80)
    for i in range(min(10, len(needles))):
        print(f"{i+1}. {needles[i]}")
    