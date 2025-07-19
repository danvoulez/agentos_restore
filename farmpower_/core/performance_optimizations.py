"""
Performance Optimizations for LogLineOS
Provides optimization utilities and enhanced data structures
Created: 2025-07-19 06:27:20 UTC
User: danvoulez
"""
import os
import time
import logging
import asyncio
import functools
import gc
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic, Set
import numpy as np
from collections import OrderedDict, deque
import weakref

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/performance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PerformanceOptimizations")

# Import optional libraries
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

# Memory-optimized LRU Cache
class OptimizedLRUCache:
    """
    Memory-optimized LRU cache implementation with optional TTL and size limits
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = None,
                weak_values: bool = False, on_evict: Callable = None):
        """
        Initialize optimized LRU cache
        
        Args:
            max_size: Maximum number of items to store
            ttl_seconds: Time-to-live in seconds (None for no expiry)
            weak_values: Use weak references for values to allow GC
            on_evict: Optional callback function when items are evicted
        """
        self._cache = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._timestamps = {}  # Stores timestamps for TTL
        self._weak_values = weak_values
        self._on_evict = on_evict
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()  # For thread safety in async code
    
    async def get(self, key: Any) -> Optional[Any]:
        """Get an item from the cache"""
        async with self._lock:
            # Check if key exists
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL if enabled
            if self._ttl_seconds is not None:
                timestamp = self._timestamps.get(key, 0)
                if time.time() - timestamp > self._ttl_seconds:
                    # Expired
                    self._remove_item(key)
                    self._misses += 1
                    return None
            
            # Move to end (most recently used)
            value = self._cache[key]
            self._cache.move_to_end(key)
            
            # Resolve weak reference if necessary
            if self._weak_values:
                value = value()
                if value is None:
                    # Reference was garbage collected
                    self._remove_item(key)
                    self._misses += 1
                    return None
            
            self._hits += 1
            return value
    
    async def set(self, key: Any, value: Any) -> None:
        """Set an item in the cache"""
        async with self._lock:
            # Store weak reference if enabled
            if self._weak_values:
                value = weakref.ref(value)
            
            # Add or update item
            self._cache[key] = value
            self._timestamps[key] = time.time()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Check size limit
            if len(self._cache) > self._max_size:
                self._evict_oldest()
    
    async def delete(self, key: Any) -> bool:
        """Remove an item from the cache"""
        async with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear the cache"""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
                "ttl_seconds": self._ttl_seconds,
                "weak_values": self._weak_values
            }
    
    def _remove_item(self, key: Any) -> None:
        """Remove an item from the cache"""
        if self._on_evict and key in self._cache:
            value = self._cache[key]
            if self._weak_values:
                value = value()  # Resolve weak reference
            
            if value is not None:
                self._on_evict(key, value)
        
        if key in self._cache:
            del self._cache[key]
        
        if key in self._timestamps:
            del self._timestamps[key]
    
    def _evict_oldest(self) -> None:
        """Evict the oldest item from the cache"""
        if not self._cache:
            return
        
        # Get the first item (oldest)
        oldest_key, oldest_value = next(iter(self._cache.items()))
        self._remove_item(oldest_key)

# Optimized tensor operations
class TensorOptimizer:
    """
    Provides optimized tensor operations for different backends
    """
    
    def __init__(self):
        """Initialize tensor optimizer"""
        self.backend = self._determine_backend()
        self.device = self._determine_device()
        logger.info(f"TensorOptimizer initialized with backend={self.backend}, device={self.device}")
    
    def optimize_memory_layout(self, tensor_data: Any) -> Any:
        """
        Optimize memory layout for tensor operations
        
        Args:
            tensor_data: Tensor data to optimize
            
        Returns:
            Optimized tensor data
        """
        if self.backend == "numpy":
            if isinstance(tensor_data, np.ndarray) and not tensor_data.flags.c_contiguous:
                # Convert to contiguous memory layout for better performance
                return np.ascontiguousarray(tensor_data)
        
        elif self.backend == "torch" and HAS_TORCH:
            if isinstance(tensor_data, torch.Tensor):
                # Ensure tensor is contiguous
                if not tensor_data.is_contiguous():
                    tensor_data = tensor_data.contiguous()
                
                # Move to correct device if needed
                if self.device.startswith("cuda") and tensor_data.device.type != "cuda":
                    tensor_data = tensor_data.cuda()
                elif self.device == "cpu" and tensor_data.device.type != "cpu":
                    tensor_data = tensor_data.cpu()
        
        elif self.backend == "tensorflow" and HAS_TF:
            # TensorFlow uses a different memory model, but we can ensure it's in the right device
            if isinstance(tensor_data, tf.Tensor):
                # For TF, we don't need to do much for memory layout
                pass
        
        return tensor_data
    
    def optimize_matmul(self, a: Any, b: Any) -> Any:
        """
        Optimized matrix multiplication
        
        Args:
            a: First matrix
            b: Second matrix
            
        Returns:
            Result of a @ b
        """
        # Optimize memory layout first
        a = self.optimize_memory_layout(a)
        b = self.optimize_memory_layout(b)
        
        if self.backend == "numpy":
            # For large matrices, use specialized BLAS implementation
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                a_size = a.size
                b_size = b.size
                
                if a_size * b_size > 1_000_000:
                    # For very large matrices, consider chunking
                    return self._chunked_matmul(a, b)
                else:
                    # Use numpy's optimized matmul
                    return np.matmul(a, b)
        
        elif self.backend == "torch" and HAS_TORCH:
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                # For very large tensors, consider using torch.bmm for batched matmul
                if len(a.shape) >= 3 and len(b.shape) >= 3:
                    return torch.bmm(a, b)  # Batched matmul
                else:
                    # Regular matmul with mixed precision if available
                    if hasattr(torch.cuda, "amp") and a.device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            return torch.matmul(a, b)
                    else:
                        return torch.matmul(a, b)
        
        elif self.backend == "tensorflow" and HAS_TF:
            if isinstance(a, tf.Tensor) and isinstance(b, tf.Tensor):
                # Use TF's matmul with XLA compilation if possible
                return tf.matmul(a, b)
        
        # Fallback
        if hasattr(a, "__matmul__"):
            return a @ b
        elif hasattr(a, "dot"):
            return a.dot(b)
        else:
            raise TypeError("Unsupported tensor types for matmul optimization")
    
    def optimize_batch_processing(self, data: List[Any], batch_size: int = 32, 
                                 process_func: Callable = None) -> List[Any]:
        """
        Optimize batch processing of data
        
        Args:
            data: List of data items to process
            batch_size: Size of each batch
            process_func: Function to apply to each batch
            
        Returns:
            Processed data
        """
        if not data:
            return []
        
        if process_func is None:
            return data
        
        # Process in batches
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Process batch
            if self.backend == "numpy":
                # Convert to numpy batch if possible
                try:
                    np_batch = np.array(batch)
                    batch_result = process_func(np_batch)
                    results.extend(batch_result)
                    continue
                except:
                    pass
            
            elif self.backend == "torch" and HAS_TORCH:
                # Convert to torch batch if possible
                try:
                    torch_batch = torch.stack([torch.tensor(item) for item in batch])
                    batch_result = process_func(torch_batch)
                    results.extend(batch_result)
                    continue
                except:
                    pass
            
            # Fallback to individual processing
            batch_result = [process_func(item) for item in batch]
            results.extend(batch_result)
        
        return results
    
    def _determine_backend(self) -> str:
        """Determine the optimal backend"""
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
        if self.backend == "torch":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
    
    def _chunked_matmul(self, a: np.ndarray, b: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
        """
        Perform matrix multiplication in chunks to reduce memory usage
        
        Args:
            a: First matrix (m x n)
            b: Second matrix (n x p)
            chunk_size: Size of chunks
            
        Returns:
            Result matrix (m x p)
        """
        m, n = a.shape
        n2, p = b.shape
        
        if n != n2:
            raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")
        
        # Initialize result matrix
        result = np.zeros((m, p), dtype=np.result_type(a.dtype, b.dtype))
        
        # Process in chunks
        for i in range(0, m, chunk_size):
            i_end = min(i + chunk_size, m)
            a_chunk = a[i:i_end]
            
            for j in range(0, p, chunk_size):
                j_end = min(j + chunk_size, p)
                b_chunk = b[:, j:j_end]
                
                # Compute chunk result
                result[i:i_end, j:j_end] = np.matmul(a_chunk, b_chunk)
        
        return result

# Memory optimization utilities
class MemoryOptimizer:
    """
    Utilities for optimizing memory usage
    """
    
    @staticmethod
    def monitor_memory_usage(interval_seconds: float = 60.0, threshold_mb: float = 1000.0,
                            callback: Callable = None):
        """
        Start monitoring memory usage with periodic checks
        
        Args:
            interval_seconds: Check interval in seconds
            threshold_mb: Memory threshold in MB to trigger GC
            callback: Optional callback when threshold is exceeded
        """
        async def _monitor():
            while True:
                try:
                    # Get current memory usage
                    memory_mb = MemoryOptimizer.get_memory_usage_mb()
                    
                    if memory_mb > threshold_mb:
                        logger.warning(f"Memory usage ({memory_mb:.2f} MB) exceeds threshold ({threshold_mb:.2f} MB)")
                        
                        # Run garbage collection
                        collected = gc.collect()
                        logger.info(f"Garbage collection freed {collected} objects")
                        
                        # Call callback if provided
                        if callback:
                            try:
                                callback(memory_mb)
                            except Exception as e:
                                logger.error(f"Error in memory callback: {str(e)}")
                        
                        # Check memory again after GC
                        new_memory_mb = MemoryOptimizer.get_memory_usage_mb()
                        logger.info(f"Memory usage after GC: {new_memory_mb:.2f} MB (freed {memory_mb - new_memory_mb:.2f} MB)")
                    
                except Exception as e:
                    logger.error(f"Error in memory monitor: {str(e)}")
                
                await asyncio.sleep(interval_seconds)
        
        # Start the monitoring task
        task = asyncio.create_task(_monitor())
        return task
    
    @staticmethod
    def get_memory_usage_mb() -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except ImportError:
            # Fallback if psutil is not available
            return 0.0
    
    @staticmethod
    def optimize_numpy_array(arr: np.ndarray) -> np.ndarray:
        """
        Optimize a numpy array for memory usage
        
        Args:
            arr: Numpy array to optimize
            
        Returns:
            Optimized array
        """
        if not isinstance(arr, np.ndarray):
            return arr
        
        # Check if we can use a more memory-efficient dtype
        current_dtype = arr.dtype
        
        # Find the actual range of values
        if np.issubdtype(current_dtype, np.integer):
            min_val = arr.min()
            max_val = arr.max()
            
            # Determine minimal integer type that can represent the range
            if min_val >= 0:
                if max_val <= 255:
                    new_dtype = np.uint8
                elif max_val <= 65535:
                    new_dtype = np.uint16
                elif max_val <= 4294967295:
                    new_dtype = np.uint32
                else:
                    new_dtype = np.uint64
            else:
                if min_val >= -128 and max_val <= 127:
                    new_dtype = np.int8
                elif min_val >= -32768 and max_val <= 32767:
                    new_dtype = np.int16
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    new_dtype = np.int32
                else:
                    new_dtype = np.int64
            
            # Only change if new dtype is more efficient
            if new_dtype != current_dtype and np.dtype(new_dtype).itemsize < np.dtype(current_dtype).itemsize:
                return arr.astype(new_dtype)
        
        elif np.issubdtype(current_dtype, np.floating):
            # Check if float32 is sufficient instead of float64
            if current_dtype == np.float64:
                # Check the precision actually needed
                float32_arr = arr.astype(np.float32)
                max_diff = np.abs(arr - float32_arr.astype(np.float64)).max()
                
                # If difference is small, use float32
                if max_diff < 1e-6:
                    return float32_arr
        
        # Make contiguous if not already
        if not arr.flags.c_contiguous:
            return np.ascontiguousarray(arr)
        
        return arr

# Optimized concurrent operations
class ConcurrentOptimizer:
    """
    Utilities for optimizing concurrent operations
    """
    
    def __init__(self, max_concurrency: int = None):
        """
        Initialize concurrent optimizer
        
        Args:
            max_concurrency: Maximum concurrent operations (None for auto)
        """
        # Auto-determine concurrency if not specified
        if max_concurrency is None:
            import multiprocessing
            max_concurrency = min(32, multiprocessing.cpu_count() * 2)
        
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        logger.info(f"ConcurrentOptimizer initialized with max_concurrency={max_concurrency}")
    
    async def run_concurrent(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """
        Run multiple tasks concurrently with controlled concurrency
        
        Args:
            tasks: List of callables to execute
            *args: Arguments to pass to each task
            **kwargs: Keyword arguments to pass to each task
            
        Returns:
            List of results in the same order as tasks
        """
        async def _wrapped_task(task):
            async with self.semaphore:
                if asyncio.iscoroutinefunction(task):
                    return await task(*args, **kwargs)
                else:
                    return await asyncio.to_thread(task, *args, **kwargs)
        
        # Create task futures
        futures = [_wrapped_task(task) for task in tasks]
        
        # Wait for all to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Check for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with error: {str(result)}")
        
        return results
    
    def optimize_data_partitioning(self, data: List[Any]) -> List[List[Any]]:
        """
        Optimally partition data for parallel processing
        
        Args:
            data: List of data items to partition
            
        Returns:
            List of data partitions
        """
        if not data:
            return []
        
        # Calculate optimal partition size
        total_items = len(data)
        
        # Don't create more partitions than we can process concurrently
        partition_count = min(self.max_concurrency, total_items)
        
        if partition_count <= 1:
            return [data]
        
        # Create partitions with approximately equal sizes
        base_size = total_items // partition_count
        remainder = total_items % partition_count
        
        partitions = []
        start_idx = 0
        
        for i in range(partition_count):
            # Add one extra item to some partitions to distribute remainder
            partition_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + partition_size
            
            partitions.append(data[start_idx:end_idx])
            start_idx = end_idx
        
        return partitions

# IO optimization utilities
class IOOptimizer:
    """
    Utilities for optimizing IO operations
    """
    
    @staticmethod
    async def buffered_file_read(filename: str, chunk_size: int = 64 * 1024) -> bytes:
        """
        Read a file in buffered chunks to optimize memory usage
        
        Args:
            filename: Path to the file
            chunk_size: Size of each read chunk in bytes
            
        Returns:
            File content as bytes
        """
        # Create a BytesIO buffer
        from io import BytesIO
        buffer = BytesIO()
        
        # Open file and read in chunks
        loop = asyncio.get_event_loop()
        
        def _read_file():
            with open(filename, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    buffer.write(chunk)
        
        # Run file reading in a thread
        await loop.run_in_executor(None, _read_file)
        
        # Return the buffer content
        return buffer.getvalue()
    
    @staticmethod
    async def parallel_file_read(filenames: List[str]) -> Dict[str, bytes]:
        """
        Read multiple files in parallel
        
        Args:
            filenames: List of file paths
            
        Returns:
            Dictionary mapping filenames to content
        """
        async def _read_file(filename):
            try:
                content = await IOOptimizer.buffered_file_read(filename)
                return filename, content
            except Exception as e:
                logger.error(f"Error reading {filename}: {str(e)}")
                return filename, None
        
        # Create tasks for each file
        tasks = [_read_file(filename) for filename in filenames]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        return {filename: content for filename, content in results if content is not None}
    
    @staticmethod
    async def optimized_json_read(filename: str) -> Any:
        """
        Optimized JSON file reading
        
        Args:
            filename: Path to JSON file
            
        Returns:
            Parsed JSON content
        """
        import json
        import orjson  # Faster JSON parser
        
        try:
            # Try using orjson for better performance
            loop = asyncio.get_event_loop()
            
            def _read_json():
                with open(filename, 'rb') as f:
                    return orjson.loads(f.read())
            
            return await loop.run_in_executor(None, _read_json)
        except ImportError:
            # Fall back to standard json
            loop = asyncio.get_event_loop()
            
            def _read_json_std():
                with open(filename, 'r') as f:
                    return json.load(f)
            
            return await loop.run_in_executor(None, _read_json_std)