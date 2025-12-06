# api_usage.py
import os
import sys
import pandas as pd

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from api_generator import APIMedicalGenerator
from config import GeneratorConfig, APIConfig, GLOBAL_NUM_SAMPLES, GLOBAL_OUTPUT_FILE, GLOBAL_SUBTLETY_LEVEL

def setup_api_client():
    """Set up the DeepSeek API client"""
    try:
        from openai import OpenAI
        
        # Get API key from environment variable or set directly
        api_key = os.getenv('DEEPSEEK_API_KEY', 'your-api-key-here')
        
        if api_key == 'your-api-key-here':
            print("Please set your DeepSeek API key:")
            print("1. Set environment variable: export DEEPSEEK_API_KEY='your-key'")
            print("2. Or replace 'your-api-key-here' with your actual key")
            return None
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=30
        )
        
        # Test the connection
        try:
            test_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Say hello in one word."}],
                max_tokens=10
            )
            print("✓ DeepSeek API connection successful")
            return client
        except Exception as e:
            print(f"✗ API connection failed: {e}")
            return None
            
    except ImportError:
        print("OpenAI package not installed. Run: pip install openai")
        return None

def generate_with_api():
    """Generate medical needles using API"""
    
    # Setup API client
    client = setup_api_client()
    if not client:
        raise ConnectionError(
            "Cannot connect to DeepSeek API. "
            "Please check your API key and internet connection. "
            "API is required for this generator."
        )
    
    # Configure the generator
    config = GeneratorConfig(
        num_samples=GLOBAL_NUM_SAMPLES,
        output_file=GLOBAL_OUTPUT_FILE,
        subtlety_level="medium"
    )
    
    api_config = APIConfig(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=800
    )
    
    # Create API generator
    generator = APIMedicalGenerator(client, config, api_config)
    
    print("Generating medical needles with DeepSeek API...")
    
    dataset = generator.generate_dataset()
    return dataset

def analyze_api_results():
    """Analyze results from API generation"""
    try:
        df = pd.read_csv(GLOBAL_OUTPUT_FILE)
        
        print("\n" + "="*60)
        print("API GENERATION RESULTS")
        print("="*60)
        
        detection_rate = df['needle_found'].mean()
        print(f"Detection Rate: {detection_rate:.1%}")
        print(f"Total Samples: {len(df)}")
        
        # Show results
        print("\nDetailed Results:")
        for i, row in df.iterrows():
            status = "✓ FOUND" if row['needle_found'] else "✗ MISSED"
            confidence = row.get('confidence', 'N/A')
            print(f"{i+1}. {row['true_condition']} - {status} (Confidence: {confidence}%)")
            
        return df
    except FileNotFoundError:
        print("No API dataset found. Run the generator first.")
        return None

if __name__ == "__main__":
    print("DeepSeek API Medical Needle Generator")
    print("=" * 50)
    
    # Generate dataset using API
    dataset = generate_with_api()
    
    df = analyze_api_results()
    
    if df is not None:
        print(f"\nDataset saved to: {GLOBAL_OUTPUT_FILE}")