"""
Medical Needle Dataset Generator

A tool for generating medical notes with hidden conditions (needles) 
to test LLM capabilities in medical information extraction.
"""

__version__ = "1.0.0"

from .config import GeneratorConfig, APIConfig
from .base_generator import BaseMedicalGenerator
from .api_generator import APIMedicalGenerator
from .evaluator import NeedleEvaluator
from .medical_data import MedicalData
from . import utils

__all__ = [
    'GeneratorConfig',
    'APIConfig', 
    'BaseMedicalGenerator',
    'APIMedicalGenerator',
    'SimpleMedicalGenerator',
    'NeedleEvaluator',
    'MedicalData',
    'utils'
]