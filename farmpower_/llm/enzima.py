"""
Enzima System for LogLineOS/DiamondSpan
Spans that can execute external LLM agents in WASM runtime
"""
import os
import json
import logging
import asyncio
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import uuid
from datetime import datetime

from llm.llm_service import LLMService, LLMConfig

# Configure logging
logger = logging.getLogger("Enzima")

class EnzimaEngine:
    """
    Enzima execution engine for Diamond Spans
    """
    
    def __init__(self, llm_service: LLMService = None, wasm_runtime_path: str = None):
        self.llm = llm_service or LLMService(
            LLMConfig(model_id=os.getenv("LLM_MODEL_ID", "gpt-3.5-turbo"))
        )
        self.wasm_runtime_path = wasm_runtime_path or os.getenv("WASM_RUNTIME_PATH", "./wasm_runtime")
        self.enzimas: Dict[str, "Enzima"] = {}
        self.results_cache: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize the Enzima engine"""
        await self.llm.initialize()
        
        # Register built-in enzimas
        self.register_enzima("llama2-7b", Enzima(
            name="llama2-7b",
            description="Llama2 7B language model for text generation",
            version="1.0",
            capabilities=["text_generation", "summarization", "qa"],
            executor=self._execute_llama2
        ))
        
        self.register_enzima("span-generator", Enzima(
            name="span-generator",
            description="Diamond Span generator from text prompts",
            version="1.0",
            capabilities=["span_generation"],
            executor=self._execute_span_generator
        ))
        
        self.register_enzima("diamond-miner", Enzima(
            name="diamond-miner",
            description="Diamond Span mining and valuation",
            version="1.0",
            capabilities=["span_mining", "span_valuation"],
            executor=self._execute_diamond_miner
        ))
        
        logger.info(f"Enzima engine initialized with {len(self.enzimas)} built-in enzimas")
    
    def register_enzima(self, name: str, enzima: "Enzima"):
        """Register a new enzima"""
        self.enzimas[name] = enzima
        logger.info(f"Registered enzima: {name}")
    
    def get_enzima(self, name: str) -> Optional["Enzima"]:
        """Get an enzima by name"""
        return self.enzimas.get(name)
    
    async def execute_span(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an enzima span"""
        # Validate span
        if "kind" not in span or span["kind"] != "enzima":
            raise ValueError("Not an enzima span")
        
        # Extract enzima name
        enzima_name = span.get("payload", {}).get("enzyme") or span.get("metadata", {}).get("enzyme")
        if not enzima_name or enzima_name not in self.enzimas:
            raise ValueError(f"Unknown or missing enzima: {enzima_name}")
        
        # Get the enzima
        enzima = self.enzimas[enzima_name]
        
        # Create execution context
        context = {
            "span_id": span.get("id", f"span-{uuid.uuid4()}"),
            "timestamp": datetime.now().isoformat(),
            "parameters": span.get("payload", {}).get("params", {}),
            "parent_ids": span.get("parent_ids", []),
            "actor": span.get("actor", "system")
        }
        
        # Generate cache key
        cache_key = hashlib.sha256(json.dumps({
            "enzima": enzima_name,
            "params": context["parameters"],
            "timestamp": context["timestamp"]
        }).encode()).hexdigest()
        
        # Check cache
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # Execute the enzima
        try:
            start_time = time.time()
            result = await enzima.execute(context)
            execution_time = time.time() - start_time
            
            # Add execution metadata
            result["metadata"] = result.get("metadata", {})
            result["metadata"].update({
                "execution_time_ms": int(execution_time * 1000),
                "enzima": enzima_name,
                "executed_at": datetime.now().isoformat()
            })
            
            # Cache the result
            self.results_cache[cache_key] = result
            
            logger.info(f"Executed enzima {enzima_name} in {execution_time*1000:.0f}ms")
            
            return result
        except Exception as e:
            logger.error(f"Error executing enzima {enzima_name}: {str(e)}", exc_info=True)
            raise
    
    async def list_enzimas(self) -> List[Dict[str, Any]]:
        """List all available enzimas with their metadata"""
        return [
            {
                "name": enzima.name,
                "description": enzima.description,
                "version": enzima.version,
                "capabilities": enzima.capabilities
            }
            for enzima in self.enzimas.values()
        ]
    
    async def _execute_llama2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Llama2 enzima"""
        params = context["parameters"]
        prompt = params.get("input", "")
        max_tokens = params.get("max_tokens", 256)
        temperature = params.get("temperature", 0.7)
        
        # Generate text
        generated_text, stats = await self.llm.generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "output": generated_text,
            "metadata": {
                "tokens_generated": stats.tokens_generated,
                "generation_time_ms": stats.generation_time_ms
            }
        }
    
    async def _execute_span_generator(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Span Generator enzima"""
        params = context["parameters"]
        prompt = params.get("input", "")
        kind = params.get("kind", "diamond")
        
        # Generate prompt for span creation
        gen_prompt = f"""
        # DIAMOND SPAN GENERATION
        
        Generate a high-quality Diamond Span based on the following:
        
        Prompt: "{prompt}"
        Kind: {kind}
        
        Output the span as a JSON object.
        """
        
        # Generate text
        generated_text, stats = await self.llm.generate_text(gen_prompt)
        
        # Try to parse as JSON
        try:
            span_data = json.loads(generated_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            import re
            json_match = re.search(r'({.*})', generated_text, re.DOTALL)
            if json_match:
                try:
                    span_data = json.loads(json_match.group(1))
                except:
                    span_data = {"error": "Failed to parse generated span"}
            else:
                span_data = {"error": "Failed to parse generated span"}
        
        return {
            "span": span_data,
            "metadata": {
                "tokens_generated": stats.tokens_generated,
                "generation_time_ms": stats.generation_time_ms
            }
        }
    
    async def _execute_diamond_miner(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Diamond Miner enzima"""
        params = context["parameters"]
        content = params.get("content", "")
        
        # Generate a mining difficulty
        import random
        difficulty = random.uniform(1.0, 10.0)
        
        # Calculate energy based on content length and complexity
        base_energy = 10.0
        length_factor = min(3.0, len(content) / 500)
        energy = base_energy * length_factor * random.uniform(0.8, 1.2)
        
        # Simulate mining time
        mining_time = difficulty * random.uniform(1.0, 3.0)
        time.sleep(min(2.0, mining_time / 10))  # Don't actually wait the full time
        
        return {
            "mined_span": {
                "id": f"mined-{uuid.uuid4()}",
                "content": content,
                "energy": energy,
                "difficulty": difficulty
            },
            "metadata": {
                "mining_time_seconds": mining_time,
                "energy": energy
            }
        }


class Enzima:
    """
    Enzima - executable agent for Diamond Spans
    """
    
    def __init__(self, name: str, description: str, version: str,
                capabilities: List[str], executor: Callable):
        self.name = name
        self.description = description
        self.version = version
        self.capabilities = capabilities
        self.executor = executor
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this enzima with the given context"""
        return await self.executor(context)