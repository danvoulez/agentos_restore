"""
Tensor Engine for LogLineOS
High-performance tensor operations for ML workloads
Created: 2025-07-19 05:47:12 UTC
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
        logging.FileHandler("logs/tensor_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TensorEngine")

@dataclass
class TensorConfig:
    """Configuration for tensor engine"""
    default_device: str = "auto"  # auto, cpu, cuda, mps, tpu
    precision: str = "float32"  # float32, float16, bfloat16
    memory_limit_mb: int = 1024
    enable_optimization: bool = True
    cache_size: int = 100
    parallel_operations: int = 4
    enable_distributed: bool = False
    distributed_backend: str = "nccl"
    backend: str = "auto"  # auto, numpy, torch, tensorflow

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
    start_time: float = field(default_factory=time.time)

class TensorEngine:
    """
    High-performance tensor operations engine
    """
    
    def __init__(self, config: TensorConfig = None):
        self.config = config or TensorConfig()
        self.stats = TensorStats()
        self.tensor_cache: Dict[str, Any] = {}
        self.operation_queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.tensor_registry: Dict[str, Any] = {}
        
        # Determine optimal backend and device
        self.backend = self._determine_backend()
        self.device = self._determine_device()
        
        logger.info(f"Tensor Engine initialized with backend={self.backend}, device={self.device}")
    
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
        logger.info("Tensor Engine stopped")
    
    async def create_tensor(self, 
                         data: Union[List, np.ndarray, Any],
                         tensor_id: str = None,
                         dtype: str = None) -> str:
        """
        Create a new tensor
        
        Args:
            data: Tensor data
            tensor_id: Optional ID for the tensor
            dtype: Optional data type
            
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
        tensor_data = self._convert_to_backend_tensor(data, dtype)
        
        # Store in registry
        self.tensor_registry[tensor_id] = {
            "data": tensor_data,
            "shape": self._get_shape(tensor_data),
            "dtype": dtype,
            "created_at": time.time()
        }
        
        # Update stats
        with self._update_stats_context("create"):
            self.stats.total_memory_bytes += self._estimate_memory_usage(tensor_data)
        
        logger.debug(f"Created tensor {tensor_id} with shape {self._get_shape(tensor_data)}")
        return tensor_id
    
    async def get_tensor(self, tensor_id: str) -> Optional[Any]:
        """Get a tensor by ID"""
        if tensor_id not in self.tensor_registry:
            return None
        
        tensor_entry = self.tensor_registry[tensor_id]
        
        # Update access time
        tensor_entry["last_accessed"] = time.time()
        
        return tensor_entry["data"]
    
    async def delete_tensor(self, tensor_id: str) -> bool:
        """Delete a tensor by ID"""
        if tensor_id not in self.tensor_registry:
            return False
        
        # Get memory usage before deleting
        tensor_data = self.tensor_registry[tensor_id]["data"]
        memory_freed = self._estimate_memory_usage(tensor_data)
        
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
                           options: Dict[str, Any] = None) -> str:
        """
        Apply an operation to tensors
        
        Args:
            operation: Operation name
            input_ids: List of input tensor IDs
            output_id: Optional ID for output tensor
            options: Optional operation-specific options
            
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
                "data": self._copy_tensor(cached_result["data"]),
                "shape": cached_result["shape"],
                "dtype": cached_result["dtype"],
                "created_at": time.time(),
                "from_cache": True
            }
            
            # Update stats
            with self._update_stats_context(operation):
                self.stats.cache_hits += 1
                
            logger.debug(f"Cache hit for operation {operation}, output: {output_id}")
            return output_id
        
        # Cache miss, perform operation
        with self._update_stats_context(operation):
            self.stats.cache_misses += 1
            
            # Get input tensors
            input_tensors = [self.tensor_registry[input_id]["data"] for input_id in input_ids]
            
            # Dispatch to appropriate operation handler
            start_time = time.time()
            
            result = await self._dispatch_operation(operation, input_tensors, options)
            
            end_time = time.time()
            operation_time_ms = (end_time - start_time) * 1000
            
            # Update stats
            self.stats.total_time_ms += operation_time_ms
            
            # Store result in registry
            self.tensor_registry[output_id] = {
                "data": result,
                "shape": self._get_shape(result),
                "dtype": self._get_dtype(result),
                "created_at": time.time()
            }
            
            # Cache result
            if self.config.enable_optimization:
                self.tensor_cache[cache_key] = {
                    "data": self._copy_tensor(result),
                    "shape": self._get_shape(result),
                    "dtype": self._get_dtype(result)
                }
                
                # Limit cache size
                if len(self.tensor_cache) > self.config.cache_size:
                    # Remove oldest cache entry
                    oldest_key = next(iter(self.tensor_cache))
                    del self.tensor_cache[oldest_key]
            
        logger.debug(f"Applied operation {operation} on {input_ids}, output: {output_id} in {operation_time_ms:.2f}ms")
        return output_id
    
    async def get_tensor_info(self, tensor_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a tensor"""
        if tensor_id not in self.tensor_registry:
            return None
        
        entry = self.tensor_registry[tensor_id]
        
        # Create info without including actual data
        info = {
            "id": tensor_id,
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "created_at": entry["created_at"],
            "last_accessed": entry.get("last_accessed"),
            "from_cache": entry.get("from_cache", False),
            "memory_bytes": self._estimate_memory_usage(entry["data"])
        }
        
        return info
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        uptime = time.time() - self.stats.start_time
        
        # Calculate derived statistics
        flops_per_second = self.stats.total_flops / uptime if uptime > 0 else 0
        memory_usage_mb = self.stats.total_memory_bytes / (1024 * 1024)
        cache_hit_rate = (self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses)) if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0
        
        return {
            "backend": self.backend,
            "device": self.device,
            "precision": self.config.precision,
            "uptime_seconds": uptime,
            "total_operations": self.stats.total_operations,
            "total_flops": self.stats.total_flops,
            "flops_per_second": flops_per_second,
            "total_memory_bytes": self.stats.total_memory_bytes,
            "memory_usage_mb": memory_usage_mb,
            "total_time_ms": self.stats.total_time_ms,
            "cache_entries": len(self.tensor_cache),
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "operations_by_type": self.stats.operations_by_type,
            "registered_tensors": len(self.tensor_registry),
            "is_running": self.is_running,
            "worker_count": len(self.workers)
        }
    
    async def queue_operation(self, 
                          operation: str,
                          input_ids: List[str],
                          output_id: str = None,
                          options: Dict[str, Any] = None) -> str:
        """Queue an operation for asynchronous execution"""
        # Generate output ID if not provided
        if not output_id:
            output_id = f"tensor-{uuid.uuid4()}"
            
        # Add to operation queue
        await self.operation_queue.put({
            "operation": operation,
            "input_ids": input_ids,
            "output_id": output_id,
            "options": options or {},
            "queued_at": time.time()
        })
        
        logger.debug(f"Queued operation {operation} with output {output_id}")
        return output_id
    
    async def _worker(self, worker_id: int):
        """Worker task for processing queued operations"""
        logger.info(f"Tensor worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get next operation from queue with timeout
                try:
                    operation_spec = await asyncio.wait_for(
                        self.operation_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process operation
                try:
                    await self.apply_operation(
                        operation_spec["operation"],
                        operation_spec["input_ids"],
                        operation_spec["output_id"],
                        operation_spec["options"]
                    )
                except Exception as e:
                    logger.error(f"Worker {worker_id} error processing operation: {str(e)}")
                
                # Mark as done
                self.operation_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1.0)  # Avoid tight loop on error
        
        logger.info(f"Tensor worker {worker_id} stopped")
    
    async def _dispatch_operation(self, 
                             operation: str, 
                             input_tensors: List[Any],
                             options: Dict[str, Any]) -> Any:
        """Dispatch operation to appropriate handler"""
        # Common operations
        if operation == "matmul":
            return self._op_matmul(input_tensors[0], input_tensors[1])
        elif operation == "add":
            return self._op_add(input_tensors[0], input_tensors[1])
        elif operation == "sub":
            return self._op_sub(input_tensors[0], input_tensors[1])
        elif operation == "mul":
            return self._op_mul(input_tensors[0], input_tensors[1])
        elif operation == "div":
            return self._op_div(input_tensors[0], input_tensors[1])
        elif operation == "transpose":
            return self._op_transpose(input_tensors[0])
        elif operation == "reshape":
            return self._op_reshape(input_tensors[0], options.get("shape"))
        elif operation == "concat":
            return self._op_concat(input_tensors, options.get("axis", 0))
        elif operation == "slice":
            return self._op_slice(input_tensors[0], options.get("start"), options.get("end"))
        elif operation == "softmax":
            return self._op_softmax(input_tensors[0], options.get("axis", -1))
        elif operation == "relu":
            return self._op_relu(input_tensors[0])
        elif operation == "tanh":
            return self._op_tanh(input_tensors[0])
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _determine_backend(self) -> str:
        """Determine the optimal backend"""
        if self.config.backend != "auto":
            return self.config.backend
            
        # Check available backends
        if HAS_TORCH and torch.cuda.is_available():
            return "torch"
        elif HAS_TF and tf.config.list_physical_devices("GPU"):
            return "tensorflow"
        elif HAS_TORCH:
            return "torch"
        elif HAS_TF:
            return "tensorflow"
        else:
            return "numpy"
    
    def _determine_device(self) -> str:
        """Determine the optimal device"""
        if self.config.default_device != "auto":
            return self.config.default_device
            
        if self.backend == "torch":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        elif self.backend == "tensorflow":
            if tf.config.list_physical_devices("GPU"):
                return "gpu"
            else:
                return "cpu"
        else:
            return "cpu"
    
    def _convert_to_backend_tensor(self, data, dtype):
        """Convert data to the appropriate backend tensor type"""
        if self.backend == "numpy":
            if isinstance(data, np.ndarray):
                return data.astype(dtype)
            else:
                return np.array(data, dtype=dtype)
                
        elif self.backend == "torch":
            if isinstance(data, torch.Tensor):
                return data.to(dtype=self._torch_dtype(dtype), device=self._torch_device())
            elif isinstance(data, np.ndarray):
                return torch.tensor(data, dtype=self._torch_dtype(dtype), device=self._torch_device())
            else:
                return torch.tensor(data, dtype=self._torch_dtype(dtype), device=self._torch_device())
                
        elif self.backend == "tensorflow":
            if isinstance(data, tf.Tensor):
                return tf.cast(data, dtype=self._tf_dtype(dtype))
            elif isinstance(data, np.ndarray):
                return tf.convert_to_tensor(data, dtype=self._tf_dtype(dtype))
            else:
                return tf.convert_to_tensor(data, dtype=self._tf_dtype(dtype))
                
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _torch_dtype(self, dtype):
        """Convert dtype string to torch dtype"""
        if HAS_TORCH:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16,
                "int32": torch.int32,
                "int64": torch.int64
            }
            return dtype_map.get(dtype, torch.float32)
        return None
    
    def _torch_device(self):
        """Get torch device"""
        if HAS_TORCH:
            if self.device == "cuda":
                return torch.device("cuda")
            elif self.device == "mps":
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return None
    
    def _tf_dtype(self, dtype):
        """Convert dtype string to tensorflow dtype"""
        if HAS_TF:
            dtype_map = {
                "float32": tf.float32,
                "float16": tf.float16,
                "bfloat16": tf.bfloat16,
                "int32": tf.int32,
                "int64": tf.int64
            }
            return dtype_map.get(dtype, tf.float32)