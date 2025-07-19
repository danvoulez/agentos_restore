"""
Lingua Mater Tensor Module for LogLineOS
Integrates tensor operations with Lingua Mater and Diamond Spans
Created: 2025-07-19 05:53:01 UTC
User: danvoulez
"""
import os
import json
import time
import logging
import asyncio
import hashlib
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Import LogLineOS core components if available
try:
    from core.diamond_span import DiamondSpan
    from core.logline_vm import LogLineVM
    from core.lingua_mater import LinguaMater
    from core.grammar_vector import GrammarVector
    from core.span_algebra import SpanAlgebra
    HAS_CORE_IMPORTS = True
except ImportError:
    HAS_CORE_IMPORTS = False

# Import tensor backends
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/lingua_tensor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LinguaMaterTensor")

@dataclass
class TensorConfig:
    """Configuration for tensor operations within Lingua Mater"""
    default_device: str = "auto"  # auto, cpu, cuda, mps, tpu
    precision: str = "float32"  # float32, float16, bfloat16
    memory_limit_mb: int = 1024
    enable_optimization: bool = True
    cache_size: int = 100
    tensor_energy_factor: float = 0.5  # Energy cost for tensor operations
    backend: str = "auto"  # auto, numpy, torch, tensorflow
    vector_dimensions: int = 768  # Default dimensions for grammar vectors

@dataclass
class TensorStats:
    """Statistics for tensor operations"""
    total_operations: int = 0
    total_energy: float = 0.0
    total_memory_bytes: int = 0
    total_time_ms: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    operations_by_type: Dict[str, int] = field(default_factory=dict)
    spans_created: int = 0
    start_time: float = field(default_factory=time.time)

class LinguaMaterTensor:
    """
    Tensor operations integrated with Lingua Mater and Diamond Spans
    Allows for seamless conversion between linguistic and mathematical representations
    """
    
    def __init__(self, 
                lingua_mater: Any = None, 
                logline_vm: Any = None,
                config: TensorConfig = None):
        """Initialize LinguaMaterTensor"""
        self.config = config or TensorConfig()
        self.stats = TensorStats()
        
        # Connect to LogLineOS components
        self.lingua_mater = lingua_mater
        self.vm = logline_vm
        
        if not self.lingua_mater and HAS_CORE_IMPORTS:
            self.lingua_mater = LinguaMater()
            logger.info("Created new LinguaMater instance")
            
        if not self.vm and HAS_CORE_IMPORTS:
            self.vm = LogLineVM()
            logger.info("Created new LogLineVM instance")
        
        # Initialize tensor storage
        self.tensor_registry: Dict[str, Dict[str, Any]] = {}
        self.grammar_vectors: Dict[str, Any] = {}
        self.tensor_cache: Dict[str, Any] = {}
        self.tensor_to_span_map: Dict[str, str] = {}  # Maps tensor IDs to span IDs
        self.span_to_tensor_map: Dict[str, str] = {}  # Maps span IDs to tensor IDs
        
        # Select backend
        self.backend = self._determine_backend()
        self.device = self._determine_device()
        
        logger.info(f"LinguaMaterTensor initialized with backend={self.backend}, "
                   f"device={self.device}, dimensions={self.config.vector_dimensions}")
    
    async def initialize(self):
        """Initialize the tensor system"""
        if HAS_CORE_IMPORTS and self.lingua_mater:
            # Initialize Lingua Mater if available
            self.lingua_mater.initialize_core_ontology()
            
            # Create base grammar vectors
            await self._initialize_base_grammar_vectors()
        
        logger.info("LinguaMaterTensor fully initialized")
    
    async def text_to_vector(self, 
                           text: str, 
                           dimensions: int = None,
                           span_id: str = None,
                           actor: str = "system") -> Tuple[str, Dict[str, Any]]:
        """
        Convert text to a grammar vector in tensor form
        
        Args:
            text: Text to vectorize
            dimensions: Vector dimensions (uses config default if not specified)
            span_id: Optional span ID to associate with this vector
            actor: Who is creating the vector
            
        Returns:
            Tuple of (tensor_id, tensor_info)
        """
        dimensions = dimensions or self.config.vector_dimensions
        
        start_time = time.time()
        
        # Create grammar vector from text
        if HAS_CORE_IMPORTS and self.lingua_mater:
            grammar_vector = GrammarVector.from_natural_language(text)
            tensor_data = grammar_vector.to_vector(dimensions).numpy()
        else:
            # Fallback if Lingua Mater not available
            tensor_data = self._fallback_text_to_vector(text, dimensions)
        
        # Generate tensor ID
        tensor_id = f"gramvec-{uuid.uuid4()}"
        
        # Convert to backend tensor format
        tensor_data = self._convert_to_backend_tensor(tensor_data)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Store in registry
        tensor_info = {
            "data": tensor_data,
            "shape": self._get_shape(tensor_data),
            "dtype": self.config.precision,
            "created_at": time.time(),
            "source_text": text[:100] + "..." if len(text) > 100 else text,
            "dimensions": dimensions,
            "creator": actor,
            "memory_bytes": self._estimate_memory_usage(tensor_data)
        }
        
        self.tensor_registry[tensor_id] = tensor_info
        
        # Create Diamond Span for this operation
        if span_id is None and HAS_CORE_IMPORTS and self.vm:
            span_id = await self._create_vector_span(
                "text_to_vector", 
                text, 
                actor, 
                processing_time_ms,
                dimensions
            )
            
            # Map the tensor to the span
            self.tensor_to_span_map[tensor_id] = span_id
            self.span_to_tensor_map[span_id] = tensor_id
        elif span_id:
            # Use provided span ID for mapping
            self.tensor_to_span_map[tensor_id] = span_id
            self.span_to_tensor_map[span_id] = tensor_id
        
        # Update statistics
        self.stats.total_operations += 1
        self.stats.total_time_ms += processing_time_ms
        self.stats.total_memory_bytes += tensor_info["memory_bytes"]
        self._update_operation_stats("text_to_vector")
        
        logger.info(f"Created grammar vector tensor {tensor_id} from text ({processing_time_ms:.2f}ms)")
        
        return tensor_id, tensor_info
    
    async def vector_to_text(self, 
                          tensor_id: str,
                          quality: float = 0.8,
                          actor: str = "system") -> Tuple[str, Dict[str, Any]]:
        """
        Convert a grammar vector tensor back to text
        
        Args:
            tensor_id: ID of the grammar vector tensor
            quality: Quality factor for text generation (0.0-1.0)
            actor: Who is performing the conversion
            
        Returns:
            Tuple of (generated_text, generation_info)
        """
        if tensor_id not in self.tensor_registry:
            raise ValueError(f"Tensor not found: {tensor_id}")
        
        start_time = time.time()
        
        # Get tensor data
        tensor_data = self.tensor_registry[tensor_id]["data"]
        
        # Convert to numpy array if needed
        vector_data = self._convert_to_numpy(tensor_data)
        
        # Convert to text using Lingua Mater
        if HAS_CORE_IMPORTS and self.lingua_mater:
            # Create grammar vector from numpy array
            grammar_vector = GrammarVector.from_vector(vector_data)
            
            # Generate text from grammar vector
            text = grammar_vector.to_natural_language(quality=quality)
        else:
            # Fallback without Lingua Mater
            text = self._fallback_vector_to_text(vector_data)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Get associated span ID if it exists
        span_id = self.tensor_to_span_map.get(tensor_id)
        
        # Create Diamond Span for this operation
        if HAS_CORE_IMPORTS and self.vm:
            new_span_id = await self._create_vector_span(
                "vector_to_text", 
                text, 
                actor, 
                processing_time_ms,
                vector_data.shape[0],
                parent_ids=[span_id] if span_id else []
            )
        
        # Update statistics
        self.stats.total_operations += 1
        self.stats.total_time_ms += processing_time_ms
        