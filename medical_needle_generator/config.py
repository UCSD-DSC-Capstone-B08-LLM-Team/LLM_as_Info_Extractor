from dataclasses import dataclass
from typing import List, Dict, Any

GLOBAL_NUM_SAMPLES = 50  # Global variable for number of samples
GLOBAL_OUTPUT_FILE = "api_medical_needles.csv"
GLOBAL_SUBTLETY_LEVEL = "medium"

@dataclass
class GeneratorConfig:
    """Configuration for the medical needle generator"""
    num_samples: int = GLOBAL_NUM_SAMPLES
    output_file: str = GLOBAL_OUTPUT_FILE
    subtlety_level: str = GLOBAL_SUBTLETY_LEVEL
    api_timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7
    batch_size: int = 10
    evaluation_model: str = "deepseek-chat"
    
    # Medical context configuration
    include_demographics: bool = True
    include_medications: bool = True
    include_lab_results: bool = True

@dataclass
class APIConfig:
    """API configuration"""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    max_tokens: int = 800
    temperature: float = 0.8
    timeout: int = 30