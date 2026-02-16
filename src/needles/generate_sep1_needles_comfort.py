import random
import pandas as pd

# Inclusion terms for Comfort Care/Palliative Care
INCLUSION_TERMS = [
    "brain dead",
    "brain death",
    "comfort care",
    "comfort measures",
    "comfort measures only",
    "comfort only",
    "DNR-CC",
    "end of life care",
    "hospice",
    "hospice care",
    "organ harvest",
    "palliative care",
    "palliative consult",
    "terminal care",
    "terminal extubation",
    "withdraw care",
    "withhold care",
]

# Acceptable contexts for documentation
ACCEPTABLE_CONTEXTS = [
    "comfort measures only recommendation",
    "order for consultation or evaluation by a hospice care service",
    "patient or patient representative request for comfort measures only",
    "plan for comfort measures only",
    "referral to hospice care service",
]

# State-authorized portable orders (SAPOs)
SAPO_FORMS = [
    "DNR-Comfort Care form",
    "MOLST (Medical Orders for Life-Sustaining Treatment)",
    "POLST (Physician Orders for Life-Sustaining Treatment)",
    "Out-of-Hospital DNR (OOH DNR)",
]

# Healthcare providers who can document
PROVIDERS = [
    "physician",
    "APN",
    "PA",
    "MD",
    "nurse practitioner",
    "physician assistant",
]

# Documentation sources
DOCUMENTATION_SOURCES = [
    "consultation note",
    "discharge summary",
    "ED record",
    "history and physical",
    "physician order",
    "progress note",
    "emergency department note",
]

# Note templates (POSITIVE cases - where LLM SHOULD say Yes)
POSITIVE_TEMPLATES = [
    "{provider} documents {context}: {term}.",
    "Per {provider} {source}, {context}: {term}.",
    "{provider} {source}: {context} - {term_cap}.",
    "After discussion with family, {provider} documents {context}: {term}.",
    "{provider} orders {context}: {term_cap}.",
    "{sapo_form} signed prior to arrival with {term} option selected.",
    "Most recent {sapo_form} (dated and signed by {provider}) indicates {term}.",
]

# SAPO-specific templates
SAPO_TEMPLATES = [
    "{sapo_form} dated prior to arrival with {term} option checked.",
    "Patient has {sapo_form} signed by {provider} with {term} selected.",
    "Medical record includes {sapo_form} indicating {term_cap}.",
]

# NEGATIVE cases - where LLM SHOULD say No
# These include unacceptable contexts, negative terms, or pre-arrival documentation
NEGATIVE_CONTEXTS = [
    "discussion of comfort measures",
    "consider palliative care",
    "no comfort care",
    "not appropriate for hospice care",
    "comfort care would also be reasonable - defer decision for now",
    "DNRCCA (Do Not Resuscitate – Comfort Care Arrest)",
    "family requests comfort measures only should the patient arrest",
]

NEGATIVE_TEMPLATES = [
    "{provider} documents '{negative_context}' in {source}.",
    "Per {provider} {source}: '{negative_context}'.",
    "ED note states '{negative_context}'.",
    "{provider} {source} indicates '{negative_context}'.",
    "Patient has {sapo_form} but {provider} documents '{negative_context}' on day of arrival.",
    "Previous hospitalization record shows {term} order (dated prior to arrival).",
    "MD ED note: 'Pt. on hospice at home' (pre-arrival).",
    "Chart shows 'hx dilated CMO' (cardiomyopathy context).",
]

# Time frame references
TIME_REFERENCES = [
    "",
    "within 6 hours of presentation",
    "documented before severe sepsis presentation",
    "noted in initial assessment",
    "recorded upon admission",
]

# Comfort Care/Palliative Care needle generator
def generate_comfort_care_needle():
    # Generate positive cases 70% of the time, negative 30%
    if random.random() < 0.7:
        # Positive case
        if random.random() < 0.3:
            # SAPO case
            template = random.choice(SAPO_TEMPLATES)
            return template.format(
                sapo_form=random.choice(SAPO_FORMS),
                provider=random.choice(PROVIDERS),
                term=random.choice(INCLUSION_TERMS),
                term_cap=random.choice(INCLUSION_TERMS).capitalize()
            )
        else:
            # Regular positive documentation
            template = random.choice(POSITIVE_TEMPLATES)
            time_ref = random.choice(TIME_REFERENCES)
            if time_ref:
                time_ref = " " + time_ref
            
            return template.format(
                provider=random.choice(PROVIDERS),
                context=random.choice(ACCEPTABLE_CONTEXTS),
                term=random.choice(INCLUSION_TERMS),
                term_cap=random.choice(INCLUSION_TERMS).capitalize(),
                source=random.choice(DOCUMENTATION_SOURCES),
                sapo_form=random.choice(SAPO_FORMS),
                time_ref=time_ref
            )
    else:
        # Negative case
        template = random.choice(NEGATIVE_TEMPLATES)
        return template.format(
            provider=random.choice(PROVIDERS),
            negative_context=random.choice(NEGATIVE_CONTEXTS),
            source=random.choice(DOCUMENTATION_SOURCES),
            sapo_form=random.choice(SAPO_FORMS),
            term=random.choice(INCLUSION_TERMS)
        )

# Generate multiple needles
def generate_needle_set(n=20, seed=42):
    random.seed(seed)
    return [generate_comfort_care_needle() for _ in range(n)]


if __name__ == "__main__":
    needles = generate_needle_set(n=100)

    df = pd.DataFrame({
        "DATA_ELEMENT": ["Directive for Comfort Care or Palliative Care, Severe Sepsis"] * len(needles),
        "QUERY": [(
            "Based on SEP-1 guidelines, is there physician/APN/PA documentation of "
            "comfort measures only, palliative care, or another acceptable inclusion term "
            "within an acceptable context? Note: Acceptable contexts include comfort measures "
            "only recommendation, order for hospice consultation, patient request for comfort "
            "measures only, plan for comfort measures only, or referral to hospice. "
            "Also accept SAPOs with CMO option checked if dated and signed prior to arrival, "
            "unless contradicted by documentation on day of arrival. "
            "Do NOT accept documentation that is only discussion/consideration, "
            "negative/conditional statements, or dated prior to arrival."
        )] * len(needles),
        "NEEDLE_TEXT": needles
    })

    df.to_csv("comfort_care_needles.csv", index=False)
    print(f"Generated {len(needles)} needles for Comfort Care/Palliative Care")
    print("\nSample needles:")
    print("-" * 80)
    for i in range(5):
        print(f"{i+1}. {needles[i]}")