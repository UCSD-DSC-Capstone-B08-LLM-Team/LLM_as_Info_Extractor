from typing import List, Dict, Tuple

class MedicalData:
    """Medical data and templates for the generator"""
    
    COMMON_CONDITIONS = [
        "multiple sclerosis", "lupus", "sarcoidosis", "sjogren's syndrome",
        "rheumatoid arthritis", "celiac disease", "addison's disease",
        "hashimoto's thyroiditis", "type 1 diabetes", "crohn's disease",
        "ulcerative colitis", "myasthenia gravis", "guillain-barre syndrome",
        "parkinson's disease", "alzheimer's disease", "pulmonary fibrosis",
        "endometriosis", "interstitial cystitis", "mast cell activation syndrome",
        "ehlers-danlos syndrome", "postural orthostatic tachycardia syndrome"
    ]
    
    DEMOGRAPHICS = {
        'ages': ['25-34', '35-44', '45-54', '55-64', '65-74', '75+'],
        'genders': ['Male', 'Female', 'Other'],
        'ethnicities': ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other']
    }
    
    COMORBIDITIES = [
        'hypertension', 'hyperlipidemia', 'type 2 diabetes', 'obesity', 
        'anxiety', 'depression', 'osteoarthritis', 'asthma', 'COPD'
    ]
    
    MEDICAL_STRUCTURES = [
        "General Medicine Clinic", "Internal Medicine", "Family Practice",
        "Neurology Department", "Rheumatology Clinic", "Gastroenterology Center",
        "Endocrinology Clinic", "Cardiology Department"
    ]
    

    SUBTLETY_PROMPTS = {
        "high": "Include only very subtle hints that would require expert medical knowledge to detect. The clues should be easily missed by non-specialists.",
        "medium": "Include moderate clues that a careful reader could piece together with basic medical knowledge. Balance subtlety with detectability.",
        "low": "Include more obvious clinical clues while still not stating the condition directly. Should be detectable by most medical professionals."
    }
    
    @classmethod
    def get_condition_categories(cls) -> Dict[str, List[str]]:
        """Group conditions by category"""
        return {
            "autoimmune": ["lupus", "rheumatoid arthritis", "multiple sclerosis", "celiac disease", "hashimoto's thyroiditis"],
            "neurological": ["parkinson's disease", "alzheimer's disease", "myasthenia gravis", "guillain-barre syndrome"],
            "gastrointestinal": ["crohn's disease", "ulcerative colitis", "celiac disease"],
            "endocrine": ["type 1 diabetes", "addison's disease", "hashimoto's thyroiditis"],
            "rheumatological": ["lupus", "rheumatoid arthritis", "sjogren's syndrome"]
            }