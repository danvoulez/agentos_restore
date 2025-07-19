"""
LLM Integration for Diamond Span generation and validation
"""
import os
import sys
import json
import hashlib
import requests
from typing import Dict, List, Any, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from core.diamond_span import DiamondSpan

class LLMConfig:
    def __init__(self, model_path=None, api_key=None, api_url=None):
        self.model_path = model_path
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.api_url = api_url or os.getenv("LLM_API_URL")
        self.tokenizer = None
        self.model = None
        
class LLMIntegration:
    def __init__(self, config=None):
        self.config = config or LLMConfig()
        self.pipeline = None
        
    def load_model(self):
        """Load the model if using local inference"""
        if self.pipeline is not None:
            return
            
        if self.config.model_path:
            print(f"Loading model from {self.config.model_path}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.config.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.config.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.config.model,
                tokenizer=self.config.tokenizer,
                device=device
            )
        
    def generate_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using either local model or API"""
        # For local inference
        if self.config.model_path:
            self.load_model()
            outputs = self.pipeline(
                prompt,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            return outputs[0]['generated_text']
            
        # For API-based inference
        elif self.config.api_url and self.config.api_key:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            payload = {
                "prompt": prompt,
                "max_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.9
            }
            response = requests.post(self.config.api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
            
        else:
            raise ValueError("Either model_path or api_url+api_key must be provided")
    
    def generate_span(self, prompt: str, parent_ids: List[str] = None) -> DiamondSpan:
        """Generate a Diamond Span using LLM"""
        parent_str = ", ".join(parent_ids) if parent_ids else "None"
        full_prompt = f"""
        [DIAMOND SPAN PROTOCOL]
        Parent IDs: {parent_str}
        
        Create a Diamond Span based on the following prompt:
        "{prompt}"
        
        The Diamond Span should have the following JSON format:
        {{
          "content": {{
            "text": "Generated content here",
            "tags": ["tag1", "tag2"],
            "grammar_complexity": <complexity_score>
          }},
          "metadata": {{
            "energy": <estimated_energy>,
            "reasoning": "Explanation of the span's meaning and value"
          }}
        }}
        
        JSON output:
        """
        
        # Generate text
        output = self.generate_text(full_prompt)
        
        # Extract JSON
        span_data = self._extract_json(output)
        
        # Create Diamond Span
        return DiamondSpan(
            parent_ids=parent_ids or [],
            content=span_data.get("content", {"text": prompt}),
            metadata=span_data.get("metadata", {"energy": 10.0})
        )
    
    def validate_span(self, span: DiamondSpan) -> Dict[str, Any]:
        """Validate a Diamond Span using LLM"""
        validation_prompt = f"""
        [DIAMOND SPAN VALIDATION]
        
        Please validate the following Diamond Span:
        
        ID: {span.id}
        Content: {json.dumps(span.content)}
        Parents: {span.parent_ids}
        Energy: {span.energy}
        
        Verify:
        1. Grammatical correctness
        2. Logical consistency
        3. Ethical compliance (cannot harm humans, override free will, or reduce cognitive diversity)
        4. Energy estimation accuracy
        
        Provide a validation report in JSON format:
        {{
          "is_valid": true/false,
          "grammar_score": <0-1>,
          "logic_score": <0-1>,
          "ethics_score": <0-1>,
          "energy_adjustment": <float>,
          "reasoning": "Explanation"
        }}
        
        JSON output:
        """
        
        # Generate validation
        output = self.generate_text(validation_prompt)
        
        # Extract JSON
        return self._extract_json(output)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from generated text"""
        # Find JSON in the text
        import re
        import json
        
        json_pattern = r"\{[\s\S]*\}"
        match = re.search(json_pattern, text)
        
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {match.group(0)}")
                
        return {}