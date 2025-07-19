"""
LLM Service for LogLineOS/DiamondSpan
Provides a unified interface to LLMs for span generation and processing
Current timestamp: 2025-07-19 05:33:24
User: danvoulez
"""
import os
import json
import time
import asyncio
import logging
import hashlib
import requests
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Generator, AsyncGenerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
from threading import Thread
from queue import Queue
from dataclasses import dataclass, field

# Import core LogLineOS modules
try:
    from core.diamond_span import DiamondSpan
    from core.logline_vm import LogLineVM
    from core.lingua_mater import LinguaMater
    from core.vector_clock import VectorClock
    from core.grammar_vector import GrammarVector
    from core.span_algebra import SpanAlgebra
    HAS_CORE_IMPORTS = True
except ImportError:
    HAS_CORE_IMPORTS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/llm_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LLMService")

@dataclass
class LLMConfig:
    """Configuration for an LLM"""
    model_id: str
    device: str = "auto"
    api_key: str = None
    api_url: str = None
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    quantization: str = "none"  # none, 8bit, 4bit
    context_size: int = 4096
    fallback_models: List[str] = field(default_factory=list)
    fallback_threshold_ms: int = 5000  # 5 seconds timeout for fallback
    energia_budget: float = 100.0  # Maximum energia per request
    tension_limit: float = 17.3  # Maximum tension threshold
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationStats:
    """Statistics about LLM generation"""
    tokens_generated: int = 0
    generation_time_ms: int = 0
    prompt_tokens: int = 0
    tokens_per_second: float = 0
    model_id: str = ""
    timestamp: str = ""
    energia_consumed: float = 0.0
    tension: float = 0.0
    span_id: Optional[str] = None

class EnzimaRouter:
    """
    Routes requests to appropriate Enzimas
    """
    def __init__(self):
        self.enzimas = {}
        self.default_handler = None
    
    def register(self, name: str, handler: callable):
        """Register an enzima handler"""
        self.enzimas[name] = handler
        logger.info(f"Registered enzima: {name}")
    
    def set_default(self, handler: callable):
        """Set default handler"""
        self.default_handler = handler
    
    async def route(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Route a request to the appropriate enzima"""
        enzima_name = request.get("enzima")
        
        if enzima_name and enzima_name in self.enzimas:
            logger.info(f"Routing request to enzima: {enzima_name}")
            return await self.enzimas[enzima_name](request, context)
        
        if self.default_handler:
            return await self.default_handler(request, context)
        
        return {"error": "No handler found"}

class LLMService:
    """
    Unified LLM service for LogLineOS with direct span support
    """
    
    def __init__(self, config: LLMConfig = None):
        """Initialize the LLM service"""
        self.config = config or LLMConfig(model_id="gpt-4")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = config.api_url or os.environ.get("OPENAI_API_URL")
        self.initialized = False
        self.is_local = not (config.api_key or self.api_key)
        
        # Initialize LogLineOS core components if available
        self.vm = LogLineVM() if HAS_CORE_IMPORTS else None
        self.lingua_mater = LinguaMater() if HAS_CORE_IMPORTS else None
        
        # Fallback system
        self.fallback_models = config.fallback_models
        self.fallback_threshold_ms = config.fallback_threshold_ms
        self.current_model_index = 0
        
        # Energy and tension control
        self.energia_budget = config.energia_budget
        self.tension_limit = config.tension_limit
        
        # Enzima router
        self.enzima_router = EnzimaRouter()
        
        # Checkpoint management
        try:
            from llm.checkpoint_manager import CheckpointManager
            self.checkpoint_manager = CheckpointManager()
        except ImportError:
            self.checkpoint_manager = None
        
        # Stats tracking
        self.stats = {
            "total_tokens_generated": 0,
            "total_requests": 0,
            "average_generation_time_ms": 0,
            "last_request_time": None,
            "model_id": config.model_id,
            "total_energia_consumed": 0.0,
            "average_tension": 0.0,
            "fallback_count": 0
        }
        
        logger.info(f"LLM Service initialized with model {config.model_id} (local: {self.is_local})")
    
    async def initialize(self):
        """Initialize the model"""
        if self.initialized:
            return
        
        # Initialize the language model
        if self.is_local:
            await self._initialize_local_model()
        
        # Initialize lingua mater if available
        if self.lingua_mater:
            self.lingua_mater.initialize_core_ontology()
        
        # Register default enzimas
        self._register_default_enzimas()
        
        self.initialized = True
        logger.info("LLM Service fully initialized")
    
    async def _initialize_local_model(self):
        """Initialize a local model"""
        try:
            logger.info(f"Loading local model {self.config.model_id}")
            
            # Determine device
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Using device: {device}")
            
            # Determine quantization
            if self.config.quantization == "8bit":
                quantization_config = {"load_in_8bit": True}
            elif self.config.quantization == "4bit":
                quantization_config = {"load_in_4bit": True}
            else:
                quantization_config = {}
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                use_fast=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                low_cpu_mem_usage=True,
                **quantization_config
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1
            )
            
            logger.info(f"Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            
            # Try to load fallback model if available
            if self.fallback_models:
                logger.info(f"Attempting to load fallback model: {self.fallback_models[0]}")
                self.config.model_id = self.fallback_models[0]
                await self._initialize_local_model()
            else:
                raise
    
    async def generate_text(self, 
                          prompt: Union[str, Dict[str, Any]], 
                          max_tokens: int = None, 
                          temperature: float = None,
                          stream: bool = False,
                          **kwargs) -> Union[Tuple[str, GenerationStats], AsyncGenerator[str, None]]:
        """
        Generate text from a prompt or span
        
        Args:
            prompt: Text prompt or span dictionary
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            Generated text and stats, or an async generator if streaming
        """
        # Ensure the model is initialized
        if not self.initialized:
            await self.initialize()
        
        # Parse input prompt
        prompt_text, input_span = self._parse_input_prompt(prompt)
        
        # Calculate tension if we have a span
        tension = 0.0
        if input_span and self.vm:
            tension = self._calculate_span_tension(input_span)
            if tension > self.tension_limit:
                raise ValueError(f"Span tension {tension} exceeds limit {self.tension_limit}")
        
        # Use provided parameters or defaults from config
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        # Check for enzima handling
        if input_span and input_span.get("kind") == "enzima":
            result = await self._handle_enzima(input_span)
            if result:
                return result
        
        # Check energia budget
        energia_estimate = self._estimate_energia(prompt_text, max_tokens)
        if energia_estimate > self.energia_budget:
            logger.warning(f"Energia estimate {energia_estimate} exceeds budget {self.energia_budget}")
            # Adjust max_tokens to fit budget
            max_tokens = int(max_tokens * (self.energia_budget / energia_estimate))
        
        if stream:
            # Stream the response
            return self._stream_generation(prompt_text, max_tokens, temperature, tension, input_span, **kwargs)
        else:
            # Regular response with fallback support
            return await self._generate_with_fallback(prompt_text, max_tokens, temperature, tension, input_span, **kwargs)
    
    async def _generate_with_fallback(self, 
                                   prompt_text: str, 
                                   max_tokens: int, 
                                   temperature: float, 
                                   tension: float,
                                   input_span: Dict[str, Any] = None,
                                   **kwargs) -> Tuple[str, GenerationStats]:
        """Generate text with fallback to alternative models if needed"""
        start_time = time.time()
        
        # Try generation with current model
        try:
            if self.is_local:
                # Use local model with timeout
                result_future = asyncio.create_task(
                    self._generate_local(prompt_text, max_tokens, temperature, False, tension, input_span, **kwargs)
                )
                
                # Wait with timeout
                try:
                    result = await asyncio.wait_for(result_future, timeout=self.fallback_threshold_ms/1000)
                    return result
                except asyncio.TimeoutError:
                    # Cancel the ongoing generation
                    result_future.cancel()
                    
                    # Log fallback event
                    logger.warning(f"Local model timeout ({self.fallback_threshold_ms}ms), falling back")
                    
                    # Try fallback models or API
                    return await self._try_fallbacks(prompt_text, max_tokens, temperature, tension, input_span, **kwargs)
            else:
                # Use API
                return await self._generate_api(prompt_text, max_tokens, temperature, False, tension, input_span, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}", exc_info=True)
            
            # Try fallback models or API
            return await self._try_fallbacks(prompt_text, max_tokens, temperature, tension, input_span, **kwargs)
    
    async def _try_fallbacks(self, 
                          prompt_text: str, 
                          max_tokens: int, 
                          temperature: float, 
                          tension: float,
                          input_span: Dict[str, Any] = None,
                          **kwargs) -> Tuple[str, GenerationStats]:
        """Try fallback models in sequence"""
        # Update stats
        self.stats["fallback_count"] += 1
        
        # If we're local and have fallback models, try them
        if self.is_local and self.fallback_models:
            original_model = self.config.model_id
            
            for model_id in self.fallback_models:
                try:
                    logger.info(f"Trying fallback model: {model_id}")
                    self.config.model_id = model_id
                    
                    # Reinitialize with the new model
                    self.initialized = False
                    await self._initialize_local_model()
                    
                    # Try generation with this model
                    return await self._generate_local(
                        prompt_text, max_tokens, temperature, False, tension, input_span, **kwargs
                    )
                except Exception as e:
                    logger.error(f"Fallback model {model_id} failed: {str(e)}", exc_info=True)
                    continue
            
            # If all fallbacks failed, restore original model
            logger.error("All fallback models failed")
            self.config.model_id = original_model
            self.initialized = False
            await self._initialize_local_model()
        
        # If local (with or without fallbacks) and we have API key, try API as last resort
        if self.is_local and self.api_key:
            logger.info("Falling back to API")
            self.is_local = False
            try:
                result = await self._generate_api(
                    prompt_text, max_tokens, temperature, False, tension, input_span, **kwargs
                )
                self.is_local = True
                return result
            except Exception as e:
                logger.error(f"API fallback failed: {str(e)}", exc_info=True)
                self.is_local = True
        
        # If all fallbacks failed, raise exception
        raise RuntimeError("All generation attempts failed")
    
    async def _generate_local(self, 
                           prompt_text: str, 
                           max_tokens: int, 
                           temperature: float, 
                           stream: bool,
                           tension: float,
                           input_span: Dict[str, Any] = None,
                           **kwargs) -> Union[Tuple[str, GenerationStats], AsyncGenerator[str, None]]:
        """Generate using local model"""
        if stream:
            return self._stream_local_generation(prompt_text, max_tokens, temperature, tension, input_span, **kwargs)
        
        try:
            # Count input tokens
            input_tokens = len(self.tokenizer.encode(prompt_text))
            
            # Generate
            start_time = time.time()
            outputs = self.pipeline(
                prompt_text,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                temperature=max(0.1, temperature),
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                **kwargs
            )
            end_time = time.time()
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the beginning if it's included
            if generated_text.startswith(prompt_text):
                generated_text = generated_text[len(prompt_text):]
            
            # Calculate stats
            output_tokens = len(self.tokenizer.encode(generated_text))
            generation_time_ms = int((end_time - start_time) * 1000)
            tokens_per_second = output_tokens / ((end_time - start_time) or 0.001)
            
            # Calculate energia consumed
            energia_consumed = self._calculate_energia_consumed(input_tokens, output_tokens, generation_time_ms)
            
            # Create generation stats
            stats = GenerationStats(
                tokens_generated=output_tokens,
                generation_time_ms=generation_time_ms,
                prompt_tokens=input_tokens,
                tokens_per_second=tokens_per_second,
                model_id=self.config.model_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                energia_consumed=energia_consumed,
                tension=tension,
                span_id=input_span.get("id") if input_span else None
            )
            
            # Update stats
            self._update_stats(output_tokens, generation_time_ms, energia_consumed, tension)
            
            # Log the generation
            logger.info(
                f"Generated {output_tokens} tokens in {generation_time_ms}ms "
                f"({tokens_per_second:.1f} tokens/s, energia: {energia_consumed:.2f}, tension: {tension:.2f})"
            )
            
            # Convert to span if requested
            if kwargs.get("return_as_span", False) and self.lingua_mater:
                output_span = self._generate_output_span(generated_text, input_span, stats)
                return output_span, stats
            
            return generated_text, stats
            
        except Exception as e:
            logger.error(f"Error in local generation: {str(e)}", exc_info=True)
            raise
    
    async def _stream_local_generation(self, 
                                    prompt_text: str, 
                                    max_tokens: int, 
                                    temperature: float,
                                    tension: float,
                                    input_span: Dict[str, Any] = None,
                                    **kwargs) -> AsyncGenerator[str, None]:
        """Stream generation from local model"""
        try:
            # Create a streamer
            streamer = TextIteratorStreamer(self.tokenizer)
            
            # Count input tokens
            input_tokens = len(self.tokenizer.encode(prompt_text))
            
            # Prepare generation parameters
            generation_kwargs = {
                "input_ids": self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.model.device),
                "max_new_tokens": max_tokens,
                "temperature": max(0.1, temperature),
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
                "streamer": streamer
            }
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the output
            start_time = time.time()
            output_tokens = 0
            generated_text = ""
            
            async for new_text in self._process_stream(streamer):
                generated_text += new_text
                output_tokens += 1
                yield new_text
            
            end_time = time.time()
            
            # Calculate stats
            generation_time_ms = int((end_time - start_time) * 1000)
            tokens_per_second = output_tokens / ((end_time - start_time) or 0.001)
            
            # Calculate energia consumed
            energia_consumed = self._calculate_energia_consumed(input_tokens, output_tokens, generation_time_ms)
            
            # Update stats
            self._update_stats(output_tokens, generation_time_ms, energia_consumed, tension)
            
            # Log the generation
            logger.info(
                f"Streamed {output_tokens} tokens in {generation_time_ms}ms "
                f"({tokens_per_second:.1f} tokens/s, energia: {energia_consumed:.2f}, tension: {tension:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Error in local streaming: {str(e)}", exc_info=True)
            raise
    
    async def _process_stream(self, streamer) -> AsyncGenerator[str, None]:
        """Process the streaming output"""
        for new_text in streamer:
            yield new_text
            await asyncio.sleep(0)
    
    async def _generate_api(self, 
                         prompt_text: str, 
                         max_tokens: int, 
                         temperature: float, 
                         stream: bool,
                         tension: float,
                         input_span: Dict[str, Any] = None,
                         **kwargs) -> Union[Tuple[str, GenerationStats], AsyncGenerator[str, None]]:
        """Generate using API"""
        if not self.api_key:
            raise ValueError("API key not provided")
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.config.model_id,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": self.config.top_p,
                "stream": stream
            }
            
            start_time = time.time()
            
            if stream:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    stream=True
                )
                response.raise_for_status()
                return self._stream_api_response(response, tension, input_span)
            
            else:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                end_time = time.time()
                result = response.json()
                
                # Extract text
                generated_text = result["choices"][0]["text"]
                
                # Calculate stats
                generation_time_ms = int((end_time - start_time) * 1000)
                
                # Create approximate stats (API doesn't always return token counts)
                prompt_tokens = result.get("usage", {}).get("prompt_tokens", len(prompt_text) // 4)
                completion_tokens = result.get("usage", {}).get("completion_tokens", len(generated_text) // 4)
                
                # Calculate energia consumed
                energia_consumed = self._calculate_energia_consumed(prompt_tokens, completion_tokens, generation_time_ms)
                
                # Update stats
                self._update_stats(completion_tokens, generation_time_ms, energia_consumed, tension)
                
                # Create generation stats
                stats = GenerationStats(
                    tokens_generated=completion_tokens,
                    generation_time_ms=generation_time_ms,
                    prompt_tokens=prompt_tokens,
                    tokens_per_second=completion_tokens / ((end_time - start_time) or 0.001),
                    model_id=self.config.model_id,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    energia_consumed=energia_consumed,
                    tension=tension,
                    span_id=input_span.get("id") if input_span else None
                )
                
                logger.info(f"API generated {completion_tokens} tokens in {generation_time_ms}ms (energia: {energia_consumed:.2f}, tension: {tension:.2f})")
                
                # Convert to span if requested
                if kwargs.get("return_as_span", False) and self.lingua_mater:
                    output_span = self._generate_output_span(generated_text, input_span, stats)
                    return output_span, stats
                
                return generated_text, stats
                
        except Exception as e:
            logger.error(f"Error in API generation: {str(e)}", exc_info=True)
            raise
    
    async def _stream_api_response(self, response, tension: float, input_span: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Stream API response"""
        try:
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        json_str = line_text[6:]
                        if json_str.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(json_str)
                            if chunk['choices'] and chunk['choices'][0].get('text'):
                                yield chunk['choices'][0]['text']
                        except:
                            pass
        except Exception as e:
            logger.error(f"Error streaming API response: {str(e)}", exc_info=True)
            raise
    
    def _update_stats(self, tokens: int, generation_time_ms: int, energia_consumed: float, tension: float):
        """Update generation statistics"""
        self.stats["total_tokens_generated"] += tokens
        self.stats["total_requests"] += 1
        self.stats["total_energia_consumed"] += energia_consumed
        
        # Update average generation time
        prev_total = self.stats["average_generation_time_ms"] * (self.stats["total_requests"] - 1)
        new_average = (prev_total + generation_time_ms) / self.stats["total_requests"]
        self.stats["average_generation_time_ms"] = new_average
        
        # Update average tension
        prev_tension = self.stats["average_tension"] * (self.stats["total_requests"] - 1)
        new_tension = (prev_tension + tension) / self.stats["total_requests"]
        self.stats["average_tension"] = new_tension
        
        self.stats["last_request_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    async def generate_span(self, 
                         prompt: Union[str, Dict[str, Any]], 
                         kind: str = "diamond",
                         parent_ids: List[str] = None,
                         actor: str = "llm_service",
                         **kwargs) -> Dict[str, Any]:
        """
        Generate a Diamond Span directly
        
        Args:
            prompt: Text prompt or input span
            kind: Kind of span to generate
            parent_ids: Parent span IDs
            actor: Actor name
            **kwargs: Additional parameters for generation
        
        Returns:
            Diamond Span dictionary
        """
        # Parse input
        prompt_text, input_span = self._parse_input_prompt(prompt)
        
        # Set defaults for span generation
        parent_ids = parent_ids or (input_span.get("parent_ids") if input_span else [])
        parent_str = ", ".join(parent_ids) if parent_ids else "none"
        
        # Create a span-optimized prompt
        span_prompt = f"""
        # DIAMOND SPAN GENERATION
        
        Generate a high-quality Diamond Span based on the following:
        
        Content: "{prompt_text}"
        Kind: {kind}
        Parent IDs: {parent_str}
        Actor: {actor}
        
        Output the span as a JSON object with this structure:
        {{
          "id": "<auto_generated>",
          "kind": "{kind}",
          "verb": "<ACTION_VERB>",
          "actor": "{actor}",
          "object": "<target_object>",
          "parent_ids": [{parent_str}],
          "payload": {{
            // Content based on the input
          }},
          "metadata": {{
            "generated_by": "llm_service",
            "timestamp": "{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}"
          }}
        }}
        
        Only respond with the JSON object.
        """
        
        # Generate with span return format
        kwargs["return_as_span"] = True
        result, stats = await self.generate_text(span_prompt, **kwargs)
        
        return result
    
    async def process_span(self, 
                        span: Dict[str, Any], 
                        operations: List[str] = None, 
                        **kwargs) -> Dict[str, Any]:
        """
        Process a span using the LLM
        
        Args:
            span: Span dictionary to process
            operations: List of operations to perform
            **kwargs: Additional parameters
        
        Returns:
            Processed span
        """
        # Default operations
        operations = operations or ["validate", "enhance"]
        
        # Initialize result with the original span
        result_span = span.copy()
        
        # Process operations in sequence
        for operation in operations:
            if operation == "validate":
                # Validate the span
                validation = await self._validate_span(result_span)
                # Attach validation results to metadata
                if "metadata" not in result_span:
                    result_span["metadata"] = {}
                result_span["metadata"]["validation"] = validation
                
                # If invalid and auto-correct is enabled, try to fix
                if not validation.get("is_valid", True) and kwargs.get("auto_correct", False):
                    result_span = await self._fix_span(result_span, validation)
                    
            elif operation == "enhance":
                # Enhance the span
                result_span = await self._enhance_span(result_span)
                
            elif operation == "translate":
                # Translate span to another language
                target_language = kwargs.get("target_language", "English")
                result_span = await self._translate_span(result_span, target_language)
                
            elif operation == "summarize":
                # Summarize span content
                max_length = kwargs.get("max_length", 100)
                result_span = await self._summarize_span(result_span, max_length)
                
            else:
                logger.warning(f"Unknown span operation: {operation}")
        
        return result_span
    
    async def get_embedding(self, text: Union[str, Dict[str, Any]]) -> List[float]:
        """
        Get embedding for text or span
        
        Args:
            text: Text string or span dictionary
        
        Returns:
            Vector embedding
        """
        # Parse input
        text_content, input_span = self._parse_input_prompt(text)
        
        if self.is_local:
            # Use mean pooling of hidden states as embedding
            inputs = self.tokenizer(text_content, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
            # Use mean of last hidden state
            last_hidden = outputs.hidden_states[-1][0].cpu().numpy()
            embedding = np.mean(last_hidden, axis=0).tolist()
            
            return embedding
        else:
            # Use API for embeddings
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "input": text_content,
                "model": "text-embedding-ada-002"  # Default embedding model
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["data"][0]["embedding"]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return self.stats
    
    async def _handle_enzima(self, span: Dict[str, Any]) -> Optional[Tuple[str, GenerationStats]]:
        """Handle an enzima span"""
        try:
            enzima_name = span.get("payload", {}).get("enzyme")
            if not enzima_name:
                return None
            
            # Create context for enzima
            context = {
                "span_id": span.get("id", f"span-{uuid.uuid4()}"),
                "timestamp": datetime.now().isoformat(),
                "parameters": span.get("payload", {}).get("params", {}),
                "parent_ids": span.get("parent_ids", []),
                "actor": span.get("actor", "system")
            }
            
            # Route to enzima
            request = {
                "enzima": enzima_name,
                "params": span.get("payload", {}).get("params", {})
            }
            
            result = await self.enzima_router.route(request, context)
            
            # Create stats
            stats = GenerationStats(
                tokens_generated=len(json.dumps(result)) // 4,  # Approximate
                generation_time_ms=0,  # Unknown
                prompt_tokens=len(json.dumps(span)) // 4,  # Approximate
                model_id=f"enzima:{enzima_name}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                energia_consumed=0.0,  # Unknown
                tension=0.0,  # Unknown
                span_id=span.get("id")
            )
            
            return result, stats
        
        except Exception as e:
            logger.error(f"Error handling enzima: {str(e)}", exc_info=True)
            return None
    
    def _parse_input_prompt(self, prompt: Union[str, Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Parse input prompt which could be text or span"""
        if isinstance(prompt, str):
            return prompt, None
        
        elif isinstance(prompt, dict):
            # This is a span, extract text content
            if "content" in prompt and isinstance(prompt["content"], dict) and "text" in prompt["content"]:
                return prompt["content"]["text"], prompt
            elif "payload" in prompt and isinstance(prompt["payload"], dict) and "text" in prompt["payload"]:
                return prompt["payload"]["text"], prompt
            elif "payload" in prompt and isinstance(prompt["payload"], dict) and "content" in prompt["payload"]:
                return prompt["payload"]["content"], prompt
            else:
                # No clear text content, serialize the whole span
                return json.dumps(prompt), prompt
        
        else:
            # Unknown type, convert to string
            return str(prompt), None
    
    def _calculate_span_tension(self, span: Dict[str, Any]) -> float:
        """Calculate tension for a span"""
        if not self.vm:
            return 0.0
        
        try:
            # Create span dict for VM
            vm_span = {
                'id': span.get('id', f"span-{uuid.uuid4()}"),
                'kind': span.get('kind', 'unknown'),
                'who': span.get('actor', 'unknown'),
                'what': span.get('verb', 'unknown'),
                'why': 'llm_generation',
                'payload': span.get('payload', {})
            }
            
            # Use VM to calculate tension
            tension = self.vm._calculate_tension(vm_span)
            return tension
        except Exception as e:
            logger.warning(f"Error calculating tension: {str(e)}")
            return 0.0
    
    def _estimate_energia(self, prompt_text: str, max_tokens: int) -> float:
        """Estimate energia consumption for a generation request"""
        # Simple estimate based on token counts
        prompt_tokens = len(self.tokenizer.encode(prompt_text)) if self.tokenizer else len(prompt_text) // 4
        total_tokens = prompt_tokens + max_tokens
        
        # Base energia per token
        base_energia_per_token = 0.1
        
        # Model complexity factor
        model_complexity = {
            "gpt-3.5-turbo": 1.0,
            "gpt-4": 2.0,
            "llama2-7b": 0.8,
            "llama2-13b": 1.2,
            "llama2-70b": 1.8
        }.get(self.config.model_id.lower(), 1.0)
        
        # Total energia estimate
        energia = total_tokens * base_energia_per_token * model_complexity
        
        return energia
    
    def _calculate_energia_consumed(self, prompt_tokens: int, completion_tokens: int, time_ms: int) -> float:
        """Calculate actual energia consumed by a generation request"""
        # Base energia calculation
        base_energia = (prompt_tokens * 0.1) + (completion_tokens * 0.2)
        
        # Time factor (longer generations use more energia)
        time_factor = 1.0 + (time_ms / 10000)  # 10s doubles energia
        
        # Model complexity factor
        model_complexity = {
            "gpt-3.5-turbo": 1.0,
            "gpt-4": 2.0,
            "llama2-7b": 0.8,
            "llama2-13b": 1.2,
            "llama2-70b": 1.8
        }.get(self.config.model_id.lower(), 1.0)
        
        # Total energia
        energia = base_energia * time_factor * model_complexity
        
        return energia
    
    def _generate_output_span(self, 
                            text: str, 
                            input_span: Optional[Dict[str, Any]], 
                            stats: GenerationStats) -> Dict[str, Any]:
        """Generate an output span from generated text"""
        if not HAS_CORE_IMPORTS:
            # Create a basic span without core imports
            span_id = str(uuid.uuid4())
            output_span = {
                "id": f"span-{span_id}",
                "kind": "diamond",
                "verb": "GENERATE",
                "actor": "llm_service",
                "object": "text_content",
                "parent_ids": [input_span["id"]] if input_span and "id" in input_span else [],
                "payload": {
                    "text": text
                },
                "metadata": {
                    "generated_by": "llm_service",
                    "model": stats.model_id,
                    "timestamp": stats.timestamp,
                    "energia_consumed": stats.energia_consumed,
                    "tension": stats.tension,
                    "tokens": stats.tokens_generated
                }
            }
            return output_span
        
        # Use lingua_mater to parse the text into a span
        try:
            if self.lingua_mater:
                span = self.lingua_mater.to_span(text)
                
                # Add parent ID if available
                if input_span and "id" in input_span:
                    span["parent_ids"] = [input_span["id"]]
                
                # Add metadata
                if "metadata" not in span:
                    span["metadata"] = {}
                span["metadata"].update({
                    "generated_by": "llm_service",
                    "model": stats.model_id,
                    "timestamp": stats.timestamp,
                    "energia_consumed": stats.energia_consumed,
                    "tension": stats.tension,
                    "tokens": stats.tokens_generated
                })
                
                return span
            else:
                # Fall back to basic span
                return self._generate_basic_span(text, input_span, stats)
                
        except Exception as e:
            logger.error(f"Error generating output span: {str(e)}", exc_info=True)
            return self._generate_basic_span(text, input_span, stats)
    
    def _generate_basic_span(self, 
                          text: str, 
                          input_span: Optional[Dict[str, Any]], 
                          stats: GenerationStats) -> Dict[str, Any]:
        """Generate a basic span without lingua_mater"""
        span_id = str(uuid.uuid4())
        output_span = {
            "id": f"span-{span_id}",
            "kind": "diamond",
            "verb": "GENERATE",
            "actor": "llm_service",
            "object": "text_content",
            "parent_ids": [input_span["id"]] if input_span and "id" in input_span else [],
            "payload": {
                "text": text
            },
            "metadata": {
                "generated_by": "llm_service",
                "model": stats.model_id,
                "timestamp": stats.timestamp,
                "energia_consumed": stats.energia_consumed,
                "tension": stats.tension,
                "tokens": stats.tokens_generated
            }
        }
        return output_span
    
    async def _validate_span(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a span using LLM"""
        # Create validation prompt
        validation_prompt = f"""
        # DIAMOND SPAN VALIDATION
        
        Validate the following Diamond Span:
        
        ```json
        {json.dumps(span, indent=2)}
        ```
        
        Perform the following checks:
        1. Structural validity (required fields and format)
        2. Semantic coherence (does the content make sense)
        3. Ethical compliance (no harmful content)
        4. Quality assessment (value of the span)
        
        Provide the validation result as JSON:
        {{
          "is_valid": true/false,
          "structural_score": 0-10,
          "coherence_score": 0-10,
          "ethical_score": 0-10,
          "quality_score": 0-10,
          "overall_score": 0-10,
          "issues": ["issue1", "issue2", ...],
          "recommendations": ["rec1", "rec2", ...]
        }}
        
        Only respond with the JSON.
        """
        
        # Generate validation
        validation_text, _ = await self.generate_text(validation_prompt)
        
        # Parse result
        try:
            validation = json.loads(validation_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            import re
            json_match = re.search(r'({.*})', validation_text, re.DOTALL)
            if json_match:
                try:
                    validation = json.loads(json_match.group(1))
                except:
                    validation = {
                        "is_valid": False,
                        "structural_score": 0,
                        "coherence_score": 0,
                        "ethical_score": 0,
                        "quality_score": 0,
                        "overall_score": 0,
                        "issues": ["Failed to parse validation result"],
                        "recommendations": ["Check the span format"]
                    }
            else:
                validation = {
                    "is_valid": False,
                    "structural_score": 0,
                    "coherence_score": 0,
                    "ethical_score": 0,
                    "quality_score": 0,
                    "overall_score": 0,
                    "issues": ["Failed to parse validation result"],
                    "recommendations": ["Check the span format"]
                }
        
        return validation
    
    async def _fix_span(self, span: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        """Fix issues in a span"""
        # Create fix prompt
        issues = validation.get("issues", [])
        recommendations = validation.get("recommendations", [])
        
        fix_prompt = f"""
        # DIAMOND SPAN REPAIR
        
        Fix the following Diamond Span:
        
        ```json
        {json.dumps(span, indent=2)}
        ```
        
        Issues to fix:
        {json.dumps(issues, indent=2)}
        
        Recommendations:
        {json.dumps(recommendations, indent=2)}
        
        Provide the corrected span as JSON.
        Only respond with the fixed JSON.
        """
        
        # Generate fixed span
        fixed_text, _ = await self.generate_text(fix_prompt)
        
        # Parse result
        try:
            fixed_span = json.loads(fixed_text)
            return fixed_span
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            import re
            json_match = re.search(r'({.*})', fixed_text, re.DOTALL)
            if json_match:
                try:
                    fixed_span = json.loads(json_match.group(1))
                    return fixed_span
                except:
                    pass
            
            # If parsing fails, return original span
            logger.warning("Failed to parse fixed span")
            return span
    
    async def _enhance_span(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a span with additional content"""
        # Create enhancement prompt
        enhance_prompt = f"""
        # DIAMOND SPAN ENHANCEMENT
        
        Enhance the following Diamond Span:
        
        ```json
        {json.dumps(span, indent=2)}
        ```
        
        Improvements to make:
        1. Add more detail to content
        2. Improve metadata
        3. Add relevant tags
        4. Increase cognitive value
        
        Provide the enhanced span as JSON.
        Only respond with the enhanced JSON.
        """
        
        # Generate enhanced span
        enhanced_text, _ = await self.generate_text(enhance_prompt)
        
        # Parse result
        try:
            enhanced_span = json.loads(enhanced_text)
            
            # Preserve the original ID
            if "id" in span:
                enhanced_span["id"] = span["id"]
                
            return enhanced_span
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            import re
            json_match = re.search(r'({.*})', enhanced_text, re.DOTALL)
            if json_match:
                try:
                    enhanced_span = json.loads(json_match.group(1))
                    
                    # Preserve the original ID
                    if "id" in span:
                        enhanced_span["id"] = span["id"]
                        
                    return enhanced_span
                except:
                    pass
            
            # If parsing fails, return original span
            logger.warning("Failed to parse enhanced span")
            return span
    
    async def _translate_span(self, span: Dict[str, Any], target_language: str) -> Dict[str, Any]:
        """Translate a span to another language"""
        # Create translation prompt
        translate_prompt = f"""
        # DIAMOND SPAN TRANSLATION
        
        Translate the following Diamond Span to {target_language}:
        
        ```json
        {json.dumps(span, indent=2)}
        ```
        
        Translate all text content, but keep the structure and IDs unchanged.
        Provide the translated span as JSON.
        Only respond with the translated JSON.
        """
        
        # Generate translated span
        translated_text, _ = await self.generate_text(translate_prompt)
        
        # Parse result
        try:
            translated_span = json.loads(translated_text)
            
            # Preserve the original ID
            if "id" in span:
                translated_span["id"] = span["id"]
                
            # Add translation metadata
            if "metadata" not in translated_span:
                translated_span["metadata"] = {}
            translated_span["metadata"]["translated_from"] = span.get("metadata", {}).get("language", "unknown")
            translated_span["metadata"]["language"] = target_language
            translated_span["metadata"]["translated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
            return translated_span
        except json.JSONDecodeError:
            # If parsing fails, return original span
            logger.warning("Failed to parse translated span")
            return span
    
    async def _summarize_span(self, span: Dict[str, Any], max_length: int = 100) -> Dict[str, Any]:
        """Summarize a span"""
        # Create summary prompt
        summarize_prompt = f"""
        # DIAMOND SPAN SUMMARIZATION
        
        Summarize the following Diamond Span:
        
        ```json
        {json.dumps(span, indent=2)}
        ```
        
        Create a summary that is no longer than {max_length} words.
        Preserve the core meaning and important details.
        
        Provide the summary as a JSON span with the same structure.
        Only respond with the summarized JSON.
        """
        
        # Generate summarized span
        summarized_text, _ = await self.generate_text(summarize_prompt)
        
        # Parse result
        try:
            summarized_span = json.loads(summarized_text)
            
            # Preserve the original ID
            if "id" in span:
                summarized_span["id"] = span["id"]
                
            # Add summary metadata
            if "metadata" not in summarized_span:
                summarized_span["metadata"] = {}
            summarized_span["metadata"]["summarized_from"] = span.get("id", "unknown")
            summarized_span["metadata"]["summarized_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            summarized_span["metadata"]["original_length"] = len(json.dumps(span))
            summarized_span["metadata"]["summary_length"] = len(json.dumps(summarized_span))
                
            return summarized_span
        except json.JSONDecodeError:
            # If parsing fails, return original span
            logger.warning("Failed to parse summarized span")
            return span
    
    def _register_default_enzimas(self):
        """Register default enzimas"""
        # Text generation enzima
        self.enzima_router.register("text_generation", self._enzima_text_generation)
        
        # Span generation enzima
        self.enzima_router.register("span_generation", self._enzima_span_generation)
        
        # Diamond mining enzima
        self.enzima_router.register("diamond_mining", self._enzima_diamond_mining)
        
        # Set default handler
        self.enzima_router.set_default(self._enzima_default)
    
    async def _enzima_text_generation(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Text generation enzima handler"""
        prompt = request.get("params", {}).get("prompt", "")
        max_tokens = request.get("params", {}).get("max_tokens", 256)
        temperature = request.get("params", {}).get("temperature", 0.7)
        
        # Generate text
        text, stats = await self.generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "text": text,
            "stats": {
                "tokens": stats.tokens_generated,
                "time_ms": stats.generation_time_ms,
                "energia": stats.energia_consumed
            }
        }
    
    async def _enzima_span_generation(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Span generation enzima handler"""
        prompt = request.get("params", {}).get("prompt", "")
        kind = request.get("params", {}).get("kind", "diamond")
        parent_ids = request.get("params", {}).get("parent_ids", [])
        
        # Generate span
        span = await self.generate_span(
            prompt,
            kind=kind,
            parent_ids=parent_ids
        )
        
        return {"span": span}
    
    async def _enzima_diamond_mining(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Diamond mining enzima handler"""
        prompt = request.get("params", {}).get("prompt", "")
        difficulty = request.get("params", {}).get("difficulty", 1.0)
        
        # Simple mining simulation
        import random
        energy = 10.0 * (1 + random.random()) * difficulty
        
        # Create span
        span = await self.generate_span(
            prompt,
            kind="diamond",
            parent_ids=context.get("parent_ids", [])
        )
        
        # Add mining metadata
        if "metadata" not in span:
            span["metadata"] = {}
        span["metadata"]["mined"] = True
        span["metadata"]["mining_difficulty"] = difficulty
        span["metadata"]["energy"] = energy
        
        return {
            "span": span,
            "mining_stats": {
                "difficulty": difficulty,
                "energy": energy,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
        }
    
    async def _enzima_default(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Default enzima handler"""
        return {
            "error": "Unknown enzima or no handler available",
            "request": request
        }