"""
LogLine Tensor Engine for LogLineOS
High-performance tensor operations integrated with LogLine spans
Created: 2025-07-19 05:51:54 UTC
User: danvoulez
"""
import os
import numpy as np
import time
import logging
import asyncio
import threading
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import json

# Try to import core LogLine components
try:
    from core.diamond_span import DiamondSpan
    from core.logline_vm import LogLineVM, SpanStatus
    HAS_LOGLINE = True
except ImportError:
    HAS_LOGLINE = False

# Try to import optional GPU libraries
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
        logging.FileHandler("logs/logline_tensor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogLineTensor")

@dataclass
class TensorConfig:
    """Configuration for LogLine tensor engine"""
    default_device: str = "auto"  # auto, cpu, cuda, mps, tpu
    precision: str = "float32"  # float32, float16, bfloat16
    memory_limit_mb: int = 1024
    enable_optimization: bool = True
    cache_size: int = 100
    parallel_operations: int = 4
    enable_distributed: bool = False
    distributed_backend: str = "nccl"
    backend: str = "auto"  # auto, numpy, torch, tensorflow
    create_tensor_spans: bool = True  # Whether to create LogLine spans for tensor operations
    energy_factor: float = 0.5  # Energy cost for tensor operations

@dataclass
class TensorStats:
    """Statistics for tensor operations"""
    total_operations: int = 0
    total_flops: int = 0
    total_memory_bytes: int = 0
    total_time_ms: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    operations_by_type: Dict[str, int] = field(default_factory=dict)
    total_energy: float = 0.0
    total_spans_created: int = 0
    start_time: float = field(default_factory=time.time)

class LogLineTensor:
    """
    LogLine Tensor Engine - High-performance tensor operations integrated with LogLineOS
    """
    
    def __init__(self, config: TensorConfig = None, vm: Optional[Any] = None):
        self.config = config or TensorConfig()
        self.stats = TensorStats()
        self.tensor_cache: Dict[str, Any] = {}
        self.operation_queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.tensor_registry: Dict[str, Any] = {}
        
        # Store LogLine VM reference if provided
        self.vm = vm if HAS_LOGLINE else None
        
        # Determine optimal backend and device
        self.backend = self._determine_backend()
        self.device = self._determine_device()
        
        logger.info(f"LogLine Tensor Engine initialized with backend={self.backend}, "
                   f"device={self.device}, logline_integration={HAS_LOGLINE}")
    
    async def start(self):
        """Start the tensor engine workers"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.config.parallel_operations):
            task = asyncio.create_task(self._worker(i))
            self.workers.append(task)
            
        logger.info(f"Started {len(self.workers)} tensor workers")
    
    async def stop(self):
        """Stop the tensor engine workers"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.workers:
            task.cancel()
            
        self.workers = []
        logger.info("LogLine Tensor Engine stopped")
    
    async def create_tensor(self, 
                         data: Union[List, np.ndarray, Any],
                         tensor_id: str = None,
                         dtype: str = None,
                         name: str = None,
                         creator: str = "system") -> str:
        """
        Create a new tensor
        
        Args:
            data: Tensor data
            tensor_id: Optional ID for the tensor
            dtype: Optional data type
            name: Optional name for the tensor
            creator: Creator of the tensor
            
        Returns:
            Tensor ID
        """
        # Generate ID if not provided
        if not tensor_id:
            tensor_id = f"tensor-{uuid.uuid4()}"
        
        # Determine dtype
        if not dtype:
            dtype = self.config.precision
        
        # Convert data to appropriate format for backend
        start_time = time.time()
        tensor_data = self._convert_to_backend_tensor(data, dtype)
        conversion_time_ms = (time.time() - start_time) * 1000
        
        # Get shape and size
        shape = self._get_shape(tensor_data)
        memory_bytes = self._estimate_memory_usage(tensor_data)
        
        # Store in registry
        self.tensor_registry[tensor_id] = {
            "data": tensor_data,
            "shape": shape,
            "dtype": dtype,
            "name": name or tensor_id,
            "created_at": time.time(),
            "creator": creator,
            "memory_bytes": memory_bytes
        }
        
        # Update stats
        with self._update_stats_context("create"):
            self.stats.total_memory_bytes += memory_bytes
        
        # Create LogLine span if enabled
        if self.config.create_tensor_spans and self.vm:
            span_id = await self._create_tensor_span(
                "create_tensor",
                creator,
                {"tensor_id": tensor_id, "shape": shape, "dtype": dtype, "name": name},
                conversion_time_ms,
                memory_bytes
            )
            self.tensor_registry[tensor_id]["span_id"] = span_id
        
        logger.debug(f"Created tensor {tensor_id} with shape {shape}")
        return tensor_id
    
    async def get_tensor(self, tensor_id: str) -> Optional[Any]:
        """Get a tensor by ID"""
        if tensor_id not in self.tensor_registry:
            return None
        
        tensor_entry = self.tensor_registry[tensor_id]
        
        # Update access time
        tensor_entry["last_accessed"] = time.time()
        
        return tensor_entry["data"]
    
    async def delete_tensor(self, tensor_id: str, actor: str = "system") -> bool:
        """
        Delete a tensor by ID
        
        Args:
            tensor_id: ID of the tensor to delete
            actor: Who is deleting the tensor
            
        Returns:
            True if deleted, False if not found
        """
        if tensor_id not in self.tensor_registry:
            return False
        
        # Get memory usage before deleting
        tensor_entry = self.tensor_registry[tensor_id]
        memory_freed = tensor_entry["memory_bytes"]
        
        # Create LogLine span if enabled
        if self.config.create_tensor_spans and self.vm:
            await self._create_tensor_span(
                "delete_tensor",
                actor,
                {"tensor_id": tensor_id, "name": tensor_entry.get("name")},
                0,
                -memory_freed  # Negative because we're freeing memory
            )
        
        # Delete the tensor
        del self.tensor_registry[tensor_id]
        
        # Update stats
        with self._update_stats_context("delete"):
            self.stats.total_memory_bytes -= memory_freed
        
        logger.debug(f"Deleted tensor {tensor_id}")
        return True
    
    async def apply_operation(self, 
                           operation: str,
                           input_ids: List[str],
                           output_id: str = None,
                           options: Dict[str, Any] = None,
                           actor: str = "system") -> str:
        """
        Apply an operation to tensors
        
        Args:
            operation: Operation name
            input_ids: List of input tensor IDs
            output_id: Optional ID for output tensor
            options: Optional operation-specific options
            actor: Who is performing the operation
            
        Returns:
            Output tensor ID
        """
        # Check if inputs exist
        for input_id in input_ids:
            if input_id not in self.tensor_registry:
                raise ValueError(f"Input tensor not found: {input_id}")
        
        # Generate output ID if not provided
        if not output_id:
            output_id = f"tensor-{uuid.uuid4()}"
        
        # Default options
        if options is None:
            options = {}
        
        # Try cache lookup
        cache_key = self._generate_cache_key(operation, input_ids, options)
        if cache_key in self.tensor_cache:
            # Cache hit
            cached_result = self.tensor_cache[cache_key]
            
            # Copy cached result to requested output ID
            self.tensor_registry[output_id] = {
                "data": self._copy_tensor(