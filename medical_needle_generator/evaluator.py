from typing import Dict, Any
from config import APIConfig

class NeedleEvaluator:
    """Evaluates if the needle condition can be detected from medical notes"""
    
    def __init__(self, api_client, api_config: APIConfig = None):
        self.client = api_client
        self.api_config = api_config or APIConfig()
    
    def evaluate_needle_detection(self, medical_note: str, true_condition: str) -> Dict[str, Any]:
        """Comprehensive evaluation of needle detection"""
        
        prompt = f"""Analyze this medical note and:
        1. Identify the most likely medical condition
        2. Provide confidence level (0-100%)
        3. List 2-3 key clues that support this diagnosis
        
        Return your response in this exact format:
        Condition: [condition name]
        Confidence: [number]%
        Clues: [clue 1], [clue 2], [clue 3]

        Medical Note:
        {medical_note}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.api_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            response_text = response.choices[0].message.content.strip()
            
            return self._parse_evaluation_response(response_text, true_condition)
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                "detected_condition": "unknown",
                "confidence": 0,
                "clues": [],
                "correct": False
            }
    
    def _parse_evaluation_response(self, response: str, true_condition: str) -> Dict[str, Any]:
        """Parse the evaluation response"""
        detected_condition = "unknown"
        confidence = 0
        clues = []
        
        try:
            lines = response.split('\n')
            for line in lines:
                if line.lower().startswith('condition:'):
                    detected_condition = line.split(':', 1)[1].strip()
                elif line.lower().startswith('confidence:'):
                    confidence_str = line.split(':', 1)[1].strip().rstrip('%')
                    confidence = int(confidence_str) if confidence_str.isdigit() else 0
                elif line.lower().startswith('clues:'):
                    clues_str = line.split(':', 1)[1].strip()
                    clues = [clue.strip() for clue in clues_str.split(',')]
        except Exception as e:
            print(f"Error parsing evaluation response: {e}")
        
        # Check if detection is correct
        correct = true_condition.lower() in detected_condition.lower() if detected_condition != "unknown" else False
        
        return {
            "detected_condition": detected_condition,
            "confidence": confidence,
            "clues": clues,
            "correct": correct
        }