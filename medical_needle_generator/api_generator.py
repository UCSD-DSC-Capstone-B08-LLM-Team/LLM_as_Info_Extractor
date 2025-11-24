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
        """Generate a specific medical condition to use as the needle"""
        # Use a mix of API and local conditions for reliability
        if random.random() < 0.7:  # 70% chance to use API
            prompt = """Generate one specific medical condition that could be discovered in a patient's medical record. 
            The condition should be one that might not be immediately obvious and could be discovered through 
            various symptoms and test results over time. Choose from autoimmune, neurological, or chronic conditions.
            
            Return ONLY the condition name, nothing else."""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.api_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=50
                )
                condition = response.choices[0].message.content.strip().lower()
                # Basic validation to ensure it's a reasonable condition
                if len(condition) < 3 or condition in ['ok', 'yes', 'no', 'unknown']:
                    return random.choice(self.medical_data.COMMON_CONDITIONS)
                return condition
            except Exception as e:
                print(f"Error generating condition: {e}")
        
        return random.choice(self.medical_data.COMMON_CONDITIONS)
    
    def generate_medical_note(self, condition: str) -> str:
        """Generate a realistic medical note that subtly hints at the condition"""
        
        subtlety_prompt = self.medical_data.SUBTLETY_PROMPTS.get(
            self.config.subtlety_level, 
            self.medical_data.SUBTLETY_PROMPTS["medium"]
        )
        
        prompt = f"""Generate a realistic medical progress note for a patient. The patient has {condition}, 
        but this should NOT be explicitly stated in the note. Instead, include subtle clues through:
        
        - Symptoms that are characteristic of the condition
        - Physical exam findings
        - Laboratory results or imaging findings
        - Medication mentions that hint at the condition
        - Specialist referrals
        - Patient history elements
        
        {subtlety_prompt}
        
        Format the note professionally with sections like:
        - Subjective: Patient's reported symptoms
        - Objective: Exam findings, test results
        - Assessment: Clinical impression (be subtle)
        - Plan: Treatment plan, referrals, follow-up
        
        Keep the note between 200-400 words. Make it realistic and not overly dramatic."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.api_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.api_config.temperature,
                max_tokens=self.api_config.max_tokens
            )
            note = response.choices[0].message.content.strip()
            
            # Validate the generated note
            if self.validate_medical_note(note, condition):
                return note
            else:
                print("Generated note failed validation, using fallback")
                return self._generate_fallback_note(condition)
                
        except Exception as e:
            print(f"Error generating medical note: {e}")
            return self._generate_fallback_note(condition)
    
    def _generate_fallback_note(self, condition: str) -> str:
        """Fallback note generation if API fails"""
        return self.medical_data.FALLBACK_NOTES.get(
            condition, 
            f"""SUBJECTIVE: Patient presents with various symptoms requiring further evaluation.

OBJECTIVE: Examination reveals findings needing specialist input. Initial lab work shows abnormalities.

ASSESSMENT: Symptoms of unclear etiology requiring additional workup.

PLAN: Specialist referral and further testing indicated. Follow up to review results."""
        )
    
    def generate_dataset(self, num_samples: int = None, output_file: str = None) -> List[Dict]:
        """Generate a dataset using API calls"""
        num_samples = num_samples or self.config.num_samples
        output_file = output_file or self.config.output_file
        
        dataset = []
        
        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}...")
            
            # Generate condition and medical note
            condition = self.generate_needle_condition()
            medical_note = self.generate_medical_note(condition)
            
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
            
            print(f"  Condition: {condition}")
            print(f"  Needle Found: {evaluation_result['correct']}")
            print(f"  Note length: {len(medical_note)} characters")
            print("-" * 50)
            
            # Add delay to avoid rate limiting
            time.sleep(2)
        
        # Save to CSV
        self._save_to_csv(dataset, output_file)
        print(f"Dataset saved to {output_file}")
        
        # Print summary statistics
        self._print_summary(dataset)
        
        return dataset
    
    def _save_to_csv(self, dataset: List[Dict], output_file: str) -> None:
        """Save dataset to CSV file with extended fields"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['medical_note', 'true_condition', 'needle_found', 'detected_condition', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in dataset:
                writer.writerow(row)