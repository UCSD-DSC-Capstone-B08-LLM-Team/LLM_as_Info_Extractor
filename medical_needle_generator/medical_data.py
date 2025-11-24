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
    
    # Fallback notes for when API fails
    FALLBACK_NOTES = {
        "multiple sclerosis": """SUBJECTIVE: Patient presents with intermittent numbness in right leg lasting 2-3 days at a time. Reports occasional blurred vision and fatigue that interferes with daily activities. Denies pain or trauma.

OBJECTIVE: Neurological exam: mild hyperreflexia in lower extremities. Romberg test: mild unsteadiness with eyes closed. MRI brain: few non-specific white matter hyperintensities. Visual evoked potentials: slightly delayed.

ASSESSMENT: Neurological symptoms of unclear etiology. Rule out demyelinating process.

PLAN: Refer to neurology. Repeat MRI in 6 months. Symptomatic management for fatigue. Follow up in 3 months.""",

        "lupus": """SUBJECTIVE: Patient reports joint pain in hands and wrists, worse in mornings. Notes facial rash after sun exposure. Complains of persistent low-grade fever and increased hair loss over past month.

OBJECTIVE: BP 142/88. Malar rash noted. Joint exam: tenderness in MCP joints without swelling. Labs: ANA positive 1:160, anti-dsDNA positive. CBC: mild leukopenia. Urinalysis: trace protein.

ASSESSMENT: Multisystem inflammatory symptoms. Autoimmune process suspected.

PLAN: Rheumatology referral. Sun protection recommended. Start hydroxychloroquine 200mg daily. Monitor renal function.""",

        "celiac disease": """SUBJECTIVE: Patient describes intermittent abdominal bloating and loose stools. Notes unintentional 10lb weight loss over 3 months. Reports improvement when avoiding bread and pasta.

OBJECTIVE: Abdomen: soft, non-tender, active bowel sounds. BMI 19.2. Labs: Iron deficiency anemia (Hgb 10.2). Celiac panel: elevated tTG IgA at 45 U/mL.

ASSESSMENT: Malabsorption syndrome. Likely gluten-sensitive enteropathy.

PLAN: Gastroenterology referral. Gluten-free diet trial. Repeat serology in 3 months. Nutritional counseling.""",

        "rheumatoid arthritis": """SUBJECTIVE: Patient presents with bilateral hand stiffness lasting over an hour each morning. Reports swelling in metacarpophalangeal joints. Symptoms gradually worsening over 6 months.

OBJECTIVE: Joint exam: swelling and tenderness in MCP and wrist joints bilaterally. Reduced grip strength. Labs: Rheumatoid factor positive, CRP elevated at 2.8 mg/L. X-rays: periarticular osteopenia.

ASSESSMENT: Inflammatory polyarthritis consistent with autoimmune etiology.

PLAN: Rheumatology consultation. Start methotrexate 15mg weekly. Folic acid supplementation. Physical therapy referral."""
    }
    
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