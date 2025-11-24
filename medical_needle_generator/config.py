from dataclasses import dataclass
from typing import List, Dict, Any

# Define global variables at the module level
GLOBAL_NUM_SAMPLES = 20  # ← This is your global variable
GLOBAL_OUTPUT_FILE = "medical_needles.csv"
GLOBAL_SUBTLETY_LEVEL = "medium"

@dataclass
class GeneratorConfig:
    """Configuration for the medical needle generator"""
    num_samples: int = GLOBAL_NUM_SAMPLES  # Use the global variable as default
    output_file: str = GLOBAL_OUTPUT_FILE
    subtlety_level: str = GLOBAL_SUBTLETY_LEVEL
    api_timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7
    batch_size: int = 10
    evaluation_model: str = "deepseek-chat"
    
    # Medical context settings
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