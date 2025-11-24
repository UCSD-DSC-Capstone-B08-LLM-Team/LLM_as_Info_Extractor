import sys
import os
import pandas as pd

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from simple_generator import SimpleMedicalGenerator
    from config import GeneratorConfig
    import utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all files are in the same directory")
    sys.exit(1)

def example_simple_generator():
    """Example using simple generator (no API needed)"""
    config = GeneratorConfig(
        num_samples=5,
        output_file="simple_medical_needles.csv",
        subtlety_level="medium"
    )
    
    generator = SimpleMedicalGenerator(config)
    dataset = generator.generate_dataset()
    
    # Load and analyze the results
    try:
        df = pd.read_csv(config.output_file)
        print("\nGenerated CSV preview:")
        print(df.head())
        
        # Detailed analysis
        utils.print_detailed_analysis(df)
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
    
    return dataset

def example_analysis():
    """Example of analyzing existing dataset"""
    try:
        df = pd.read_csv("simple_medical_needles.csv")
        utils.print_detailed_analysis(df)
        
        # Save detailed analysis
        analysis = utils.analyze_detection_rates(df)
        utils.save_analysis(analysis, "detection_analysis.json")
        print(f"\nDetailed analysis saved to detection_analysis.json")
        
    except FileNotFoundError:
        print("No dataset found. Run the generator first.")

if __name__ == "__main__":
    print("Medical Needle Generator Examples")
    print("=" * 40)
    
    # Run simple generator example
    print("\n1. Running Simple Generator...")
    df = example_simple_generator()
    
    print("\n2. Analysis Example...")
    example_analysis()
    
    print("\n3. Dataset Preview:")
    print(df.head() if hasattr(df, 'head') else "Dataset generated successfully")