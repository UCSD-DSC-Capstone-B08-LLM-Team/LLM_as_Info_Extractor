import pandas as pd
from typing import List, Dict, Any
import json

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load generated dataset from CSV"""
    return pd.read_csv(file_path)

def analyze_detection_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze detection rates by condition"""
    analysis = {
        "overall_detection_rate": df['needle_found'].mean(),
        "by_condition": {},
        "total_samples": len(df)
    }
    
    for condition in df['true_condition'].unique():
        condition_data = df[df['true_condition'] == condition]
        analysis["by_condition"][condition] = {
            "detection_rate": condition_data['needle_found'].mean(),
            "sample_count": len(condition_data)
        }
    
    return analysis

def save_analysis(analysis: Dict[str, Any], output_file: str) -> None:
    """Save analysis results to JSON"""
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

def print_detailed_analysis(df: pd.DataFrame) -> None:
    """Print detailed analysis of the dataset"""
    analysis = analyze_detection_rates(df)
    
    print("\n" + "="*50)
    print("DETAILED ANALYSIS")
    print("="*50)
    print(f"Overall Detection Rate: {analysis['overall_detection_rate']:.1%}")
    print(f"Total Samples: {analysis['total_samples']}")
    
    print("\nDetection by Condition:")
    for condition, stats in analysis['by_condition'].items():
        print(f"  {condition}: {stats['detection_rate']:.1%} ({stats['sample_count']} samples)")