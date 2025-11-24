import random
from typing import List, Dict, Any
import csv
from base_generator import BaseMedicalGenerator
from config import GeneratorConfig
from medical_data import MedicalData

class SimpleMedicalGenerator(BaseMedicalGenerator):
    """Simple generator that creates synthetic data without API calls"""
    
    def __init__(self, config: GeneratorConfig = None):
        super().__init__(config)
        self.conditions_notes = [
            ("multiple sclerosis", "Patient presents with intermittent numbness and blurred vision. Neurological exam shows hyperreflexia. MRI reveals non-specific white matter changes."),
            ("lupus", "Patient reports joint pain and facial rash after sun exposure. Labs show positive ANA and leukopenia. Malar rash noted on exam."),
            ("celiac disease", "Patient describes abdominal bloating and weight loss. Iron deficiency anemia present. Symptoms improve with gluten avoidance."),
            ("rheumatoid arthritis", "Bilateral hand stiffness lasting over an hour in mornings. Joint swelling in MCPs. Rheumatoid factor positive."),
            ("parkinson's disease", "Patient notes resting tremor and slow movement. Mild cogwheel rigidity on exam. Family reports reduced facial expression."),
            ("myasthenia gravis", "Patient reports fluctuating muscle weakness, worse with activity. Ptosis noted on exam. Symptoms improve with rest."),
            ("addison's disease", "Patient presents with fatigue, hyperpigmentation, and orthostatic hypotension. Labs show hyponatremia and hyperkalemia.")
        ]
    
    def generate_needle_condition(self) -> str:
        """Generate condition from predefined list"""
        return random.choice(self.medical_data.COMMON_CONDITIONS)
    
    def generate_medical_note(self, condition: str) -> str:
        """Generate medical note from predefined templates"""
        # Find matching note or use random one
        for cond, note in self.conditions_notes:
            if cond == condition:
                return note
        return random.choice(self.conditions_notes)[1]
    
    def generate_dataset(self, num_samples: int = None, output_file: str = None) -> List[Dict]:
        """Generate simple dataset for testing"""
        num_samples = num_samples or self.config.num_samples
        output_file = output_file or self.config.output_file
        
        dataset = []
        for i in range(num_samples):
            condition = self.generate_needle_condition()
            note = self.generate_medical_note(condition)
            
            # Simulate detection with some randomness
            # Conditions with more obvious clues have higher detection rate
            obvious_conditions = ["multiple sclerosis", "lupus", "rheumatoid arthritis"]
            base_rate = 0.8 if condition in obvious_conditions else 0.6
            needle_found = random.random() < base_rate
            
            dataset.append({
                "medical_note": note,
                "true_condition": condition,
                "needle_found": needle_found
            })
        
        # Save to CSV
        self._save_to_csv(dataset, output_file)
        print(f"Simple dataset with {num_samples} samples saved to {output_file}")
        
        # Print summary
        self._print_summary(dataset)
        
        return dataset