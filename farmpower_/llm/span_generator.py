"""
LLM-based Span Generator for LogLineOS/DiamondSpan
"""
import os
import time
import json
import uuid
import asyncio
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class SpanGenerator:
    """LLM-based Span Generator"""
    
    def __init__(self, model_path=None, api_key=None, api_url=None):
        self.model_path = model_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_url = api_url or os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/completions")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    async def load_model(self):
        """Load the model if using local inference"""
        if self.pipeline is not None:
            return
        
        if not self.model_path:
            return  # Using API
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(device)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    async def generate_span(self, prompt: str, kind: str = None, 
                           parent_ids: List[str] = None, actor: str = None) -> Dict[str, Any]:
        """Generate a span using LLM"""
        # Build the LLM prompt
        parent_str = ", ".join(parent_ids) if parent_ids else "none"
        
        llm_prompt = f"""
        # DIAMOND SPAN GENERATION
        
        Generate a valid Diamond Span for LogLineOS using the following context:
        
        - User prompt: "{prompt}"
        - Requested span kind: {kind or "any appropriate kind"}
        - Parent IDs: {parent_str}
        - Actor: {actor or "appropriate actor"}
        
        A valid Diamond Span has this structure:
        ```json
        {{
            "kind": "<span_type>",
            "verb": "<ACTION_VERB>",
            "actor": "<actor_name>",
            "object": "<target_object>",
            "parent_ids": [<list_of_parent_ids>],
            "payload": {{
                <appropriate_content_for_kind>
            }}
        }}
        ```
        
        Generate ONLY the JSON content for this span:
        """
        
        # Generate the response
        response = await self._generate_text(llm_prompt)
        
        # Extract JSON from response
        span_dict = self._extract_json(response)
        
        # Ensure required fields
        if not span_dict:
            span_dict = self._create_default_span(prompt, kind, parent_ids, actor)
        
        # Add generated timestamp and ID
        if "id" not in span_dict:
            span_dict["id"] = f"{span_dict.get('kind', 'span')}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        
        return span_dict
    
    async def _generate_text(self, prompt: str) -> str:
        """Generate text using either local model or API"""
        if self.model_path:
            # Local model
            if not self.pipeline:
                await self.load_model()
                
            outputs = self.pipeline(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
            return outputs[0]['generated_text']
        else:
            # API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4",
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["text"]
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from generated text"""
        import re
        import json
        
        # Find JSON pattern
        pattern = r"```(?:json)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except:
                # Try again with another pattern
                pattern = r"\{.*\}"
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    try:
                        return json.loads(matches[0])
                    except:
                        pass
        
        # Try to find any JSON object
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except:
            pass
        
        return {}
    
    def _create_default_span(self, prompt: str, kind: str = None, 
                            parent_ids: List[str] = None, actor: str = None) -> Dict[str, Any]:
        """Create a default span if generation fails"""
        return {
            "kind": kind or "text",
            "verb": "STATE",
            "actor": actor or "system",
            "object": "content",
            "parent_ids": parent_ids or [],
            "payload": {
                "text": prompt
            }
        }