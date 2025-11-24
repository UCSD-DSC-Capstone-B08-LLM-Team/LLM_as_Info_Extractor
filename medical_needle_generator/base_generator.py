from abc import ABC, abstractmethod
from typing import List, Dict, Any
import csv
import pandas as pd
from config import GeneratorConfig
from medical_data import MedicalData

class BaseMedicalGenerator(ABC):
    """Base class for medical needle generators"""
    
    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self.medical_data = MedicalData()
    
    @abstractmethod
    def generate_needle_condition(self) -> str:
        """Generate a specific medical condition to use as the needle"""
        pass
    
    @abstractmethod
    def generate_medical_note(self, condition: str) -> str:
        """Generate a realistic medical note that subtly hints at the condition"""
        pass
    
    def validate_medical_note(self, note: str, condition: str) -> bool:
        """Validate generated note meets quality standards"""
        validation_checks = [
            len(note) > 100,  # Minimum length
            condition.lower() not in note.lower(),  # Condition not explicitly stated
            any(section in note.lower() for section in ['subjective', 'objective', 'assessment', 'plan', 'history']),  # Structure
            note.count('\n') > 2,  # Reasonable formatting
            len(note) < 5000  # Not excessively long
        ]
        return all(validation_checks)
    
    def _save_to_csv(self, dataset: List[Dict], output_file: str) -> None:
        """Save dataset to CSV file"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['medical_note', 'true_condition', 'needle_found']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in dataset:
                writer.writerow(row)
    
    def _print_summary(self, dataset: List[Dict]) -> None:
        """Print summary statistics"""
        total_samples = len(dataset)
        needles_found = sum(1 for row in dataset if row['needle_found'])
        detection_rate = (needles_found / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"\n=== SUMMARY ===")
        print(f"Total samples: {total_samples}")
        print(f"Needles found: {needles_found}")
        print(f"Detection rate: {detection_rate:.1f}%")
        
        # Count conditions
        condition_counts = {}
        for row in dataset:
            condition = row['true_condition']
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        print(f"\nCondition distribution:")
        for condition, count in condition_counts.items():
            print(f"  {condition}: {count}")
    
    @abstractmethod
    def generate_dataset(self, num_samples: int = None, output_file: str = None) -> List[Dict]:
        """Generate dataset - to be implemented by subclasses"""
        raise NotImplementedError