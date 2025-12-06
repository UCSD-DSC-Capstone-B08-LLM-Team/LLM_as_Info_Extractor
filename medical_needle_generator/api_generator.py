import random
import time
import csv
from typing import List, Dict, Any
from base_generator import BaseMedicalGenerator
from config import GeneratorConfig, APIConfig
from medical_data import MedicalData
from evaluator import NeedleEvaluator

class APIMedicalGenerator(BaseMedicalGenerator):
    """Medical needle generator using API calls"""
    
    def __init__(self, api_client, config: GeneratorConfig = None, api_config: APIConfig = None):
        super().__init__(config)
        self.client = api_client
        self.api_config = api_config or APIConfig()
        self.evaluator = NeedleEvaluator(api_client, api_config)
    
    def generate_needle_condition(self) -> str:
        """Generate a variety of medical condition using API"""
        
        # Track conditions we've already used in this session
        if not hasattr(self, '_used_conditions'):
            self._used_conditions = []
        
        # Build prompt that asks for different conditions
        used_str = ", ".join(self._used_conditions[-5:]) if self._used_conditions else "none"
        
        prompt = f"""Generate one specific medical condition that could be discovered in a patient's medical record.
        
        IMPORTANT: The condition must be DIFFERENT from these recently used: {used_str}
        
        Choose from these categories (pick RANDOMLY):
        1. Autoimmune diseases
        2. Neurological disorders
        3. Rare genetic conditions
        4. Metabolic disorders
        5. Chronic inflammatory conditions
        6. Endocrine disorders
        7. Gastrointestinal diseases
        8. Rheumatological conditions
        
        Examples of acceptable conditions (but be creative):
        - Sarcoidosis, Sjögren's syndrome, Addison's disease
        - Guillain-Barré syndrome, Parkinson's disease, ALS
        - Ehlers-Danlos syndrome, Marfan syndrome
        - Cushing's syndrome, Hashimoto's thyroiditis
        - Crohn's disease, Ulcerative colitis
        - Rheumatoid arthritis, Lupus, Scleroderma
        
        Return ONLY the condition name in lowercase, nothing else."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.api_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=50
            )
            condition = response.choices[0].message.content.strip().lower()
            
            # Clean up the condition
            condition = condition.replace('condition:', '').replace('diagnosis:', '').strip()
            
            # Filter out problematic responses
            if len(condition) < 3 or condition in ['ok', 'yes', 'no', 'unknown', 'n/a', '']:
                condition = self._get_random_condition()
            
            # Avoid repeats
            if condition in self._used_conditions[-3:]:  # If used in last 3
                print(f"  Warning: {condition} repeated, forcing change...")
                condition = self._get_random_condition(exclude=self._used_conditions)
            
            # Add to used conditions
            self._used_conditions.append(condition)
            
            return condition
            
        except Exception as e:
            print(f"Error generating condition: {e}")
            return self._get_random_condition()

    def _get_random_condition(self, exclude: list = None) -> str:
        """Get a random condition from MedicalData, excluding recent ones"""
        exclude = exclude or []
        
        # List of conditions
        all_conditions = [
            # Autoimmune
            "sarcoidosis", "sjögren's syndrome", "addison's disease", "hashimoto's thyroiditis",
            "celiac disease", "type 1 diabetes", "graves disease", "vitiligo",
            # Neurological
            "parkinson's disease", "alzheimer's disease", "huntington's disease", 
            "amyotrophic lateral sclerosis", "guillain-barré syndrome", "multiple system atrophy",
            # Gastrointestinal
            "crohn's disease", "ulcerative colitis", "irritable bowel syndrome",
            # Rheumatological
            "rheumatoid arthritis", "lupus", "scleroderma", "ankylosing spondylitis",
            "polymyalgia rheumatica", "gout",
            # Rare/Other
            "ehlers-danlos syndrome", "marfan syndrome", "cystic fibrosis",
            "pulmonary fibrosis", "interstitial lung disease", "endometriosis",
            "mast cell activation syndrome", "postural orthostatic tachycardia syndrome"
        ]
        
        # Remove excluded conditions
        available = [c for c in all_conditions if c not in exclude]
        
        if not available:
            available = all_conditions  # Reset if all are excluded
        
        return random.choice(available)
    
    def generate_medical_note(self, condition: str) -> str:
        """Generate a unique medical note using API"""
        prompt = f"""Generate a realistic medical note in SOAP format for a patient.
        
        CONTEXT: The patient may have {condition}, but DO NOT mention "{condition}" explicitly.
        
        REQUIREMENTS:
        1. Make it UNIQUE - vary patient demographics, symptoms, and findings
        2. Include subtle clues that could suggest {condition}
        3. Format with: SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN
        4. Keep between 300-600 words
        5. Use different details each time (age, gender, symptom duration, test results)
        
        Examples of variations:
        - Age: 28F, 45M, 62F, 35M, 51F
        - Symptoms: fatigue, joint pain, neurological changes, gastrointestinal issues
        - Duration: 2 weeks, 3 months, 6 months, 1 year
        - Findings: specific lab abnormalities, imaging results, physical exam findings
        
        Generate the medical note:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.api_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=600
            )
            
            note = response.choices[0].message.content.strip()
            
            # Validate the note
            if self.validate_medical_note(note, condition):
                return note
                
        except Exception as e:
            print(f"Error generating note: {e}")
            return self._generate_simple_note(condition)
    
    def generate_dataset(self, num_samples: int = None, output_file: str = None) -> List[Dict]:
        """Generate a dataset using API calls"""
        num_samples = num_samples or self.config.num_samples
        output_file = output_file or self.config.output_file
        
        dataset = []
        
        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}...")
            
            try:
                # Generate condition and medical note
                condition = self.generate_needle_condition()
                print(f"  Condition: {condition}")
                
                medical_note = self.generate_medical_note(condition)
                print(f"  Note length: {len(medical_note)} characters")
                
                # Check if needle was found
                evaluation_result = self.evaluator.evaluate_needle_detection(medical_note, condition)
                
                # Add to dataset
                dataset.append({
                    "medical_note": medical_note,
                    "true_condition": condition,
                    "needle_found": evaluation_result["correct"],
                    "detected_condition": evaluation_result["detected_condition"],
                    "confidence": evaluation_result.get("confidence", 0)
                })
                
                print(f"  Needle Found: {evaluation_result['correct']}")
                print("-" * 50)
                
            except Exception as e:
                print(f"  Error in sample {i+1}: {e}")
                continue
            
            time.sleep(1)
        
        self._save_to_csv(dataset, output_file)
        print(f"Dataset saved to {output_file}")
        
        self._print_summary(dataset)
        
        return dataset