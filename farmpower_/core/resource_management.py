"""
Resource Management System for LogLineOS
Provides tools for tracking and limiting resource usage
Created: 2025-07-19 06:37:55 UTC
User: danvoulez
"""
import os
import time
import logging
import asyncio
import threading
import uuid
import weakref
import gc
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager, asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/resource_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ResourceManagement")

# Try to import optional system monitoring libraries
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, system monitoring will be limited")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    logger.warning("GPUtil not available, GPU monitoring will be disabled")

@dataclass
class ResourceLimits:
    """Configuration for resource limits"""
    max_memory_mb: float = 1024.0  # Maximum memory usage in MB
    max_cpu_percent: float = 80.0  # Maximum CPU usage as percentage
    max_disk_gb: float = 10.0      # Maximum disk usage in GB
    max_gpu_memory_mb: float = 1024.0  # Maximum GPU memory in MB
    max_network_mbps: float = 100.0    # Maximum network bandwidth in Mbps
    max_file_handles: int = 1000    # Maximum number of open file handles
    max_threads: int = 100          # Maximum number of threads
    max_processes: int = 10         # Maximum number of child processes

class ResourceType(Enum):
    """Types of resources that can be managed"""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    GPU = "gpu"
    NETWORK = "network"
    FILE_HANDLES = "file_handles"
    THREADS = "threads"
    PROCESSES = "processes"

@dataclass
class ResourceUsage:
    """Current resource usage data"""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    disk_gb: float = 0.0
    gpu_memory_mb: float = 0.0
    network_mbps: float = 0.0
    file_handles: int = 0
    thread_count: int = 0
    process_count: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def is_exceeding_limits(self, limits: ResourceLimits) -> Dict[ResourceType, float]:
        """
        Check if usage exceeds any limits
        
        Args:
            limits: Resource limits to check against
            
        Returns:
            Dictionary of exceeded resources and their usage percentage
        """
        exceeded = {}
        
        if self.memory_mb > limits.max_memory_mb:
            exceeded[ResourceType.MEMORY] = self.memory_mb / limits.max_memory_mb
        
        if self.cpu_percent > limits.max_cpu_percent:
            exceeded[ResourceType.CPU] = self.cpu_percent / limits.max_cpu_percent
        
        if self.disk_gb > limits.max_disk_gb:
            exceeded[ResourceType.DISK] = self.disk_gb / limits.max_disk_gb
        
        if self.gpu_memory_mb > limits.max_gpu_memory_mb:
            exceeded[ResourceType.GPU] = self.gpu_memory_mb / limits.max_gpu_memory_mb
        
        if self.network_mbps > limits.max_network_mbps:
            exceeded[ResourceType.NETWORK] = self.network_mbps / limits.max_network_mbps
        
        if self.file_handles > limits.max_file_handles:
            exceeded[ResourceType.FILE_HANDLES] = self.file_handles / limits.max_file_handles
        
        if self.thread_count > limits.max_threads:
            exceeded[ResourceType.THREADS] = self.thread_count / limits.max_threads
        
        if self.process_count > limits.max_processes:
            exceeded[ResourceType.PROCESSES] = self.process_count / limits.max_processes
        
        return exceeded

class ResourceTracker:
    """
    Tracks resource usage over time
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize resource tracker
        
        Args:
            max_history: Maximum number of historical usage points to keep
        """
        self.max_history = max_history
        self.history: List[ResourceUsage] = []
        self.last_usage: Optional[ResourceUsage] = None
        
        # Network tracking needs to track previous measurements
        self._last_net_bytes_sent = 0
        self._last_net_bytes_recv = 0
        self._last_net_time = time.time()
        
        # Disk tracking needs to track previous measurements
        self._last_disk_read_bytes = 0
        self._last_disk_write_bytes = 0
        self._last_disk_time = time.time()
        
        logger.info("ResourceTracker initialized")
    
    def update(self):
        """
        Update resource usage data
        """
        # Create new usage data
        usage = ResourceUsage(timestamp=time.time())
        
        # Measure system resources
        self._update_memory_usage(usage)
        self._update_cpu_usage(usage)
        self._update_disk_usage(usage)
        self._update_gpu_usage(usage)
        self._update_network_usage(usage)
        self._update_process_usage(usage)
        
        # Save to history
        self.history.append(usage)
        self.last_usage = usage
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_current_usage(self) -> ResourceUsage:
        """
        Get current resource usage
        
        Returns:
            Current ResourceUsage data
        """
        if self.last_usage is None:
            self.update()
        
        return self.last_usage
    
    def get_usage_history(self) -> List[ResourceUsage]:
        """
        Get resource usage history
        
        Returns:
            List of historical ResourceUsage data
        """
        return self.history
    
    def get_peak_usage(self) -> ResourceUsage:
        """
        Get peak resource usage across all metrics
        
        Returns:
            ResourceUsage with peak values for each metric
        """
        if not self.history:
            return ResourceUsage()
        
        peak = ResourceUsage()
        
        # Find peak for each metric
        for usage in self.history:
            peak.memory_mb = max(peak.memory_mb, usage.memory_mb)
            peak.cpu_percent = max(peak.cpu_percent, usage.cpu_percent)
            peak.disk_gb = max(peak.disk_gb, usage.disk_gb)
            peak.gpu_memory_mb = max(peak.gpu_memory_mb, usage.gpu_memory_mb)
            peak.network_mbps = max(peak.network_mbps, usage.network_mbps)
            peak.file_handles = max(peak.file_handles, usage.file_handles)
            peak.thread_count = max(peak.thread_count, usage.thread_count)
            peak.process_count = max(peak.process_count, usage.process_count)
        
        return peak
    
    def _update_memory_usage(self, usage: ResourceUsage):
        """Update memory usage data"""
        try:
            if HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                usage.memory_mb = memory_info.rss / (1024 * 1024)
            else:
                # Fallback to less accurate method
                import resource
                usage.memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
    
    def _update_cpu_usage(self, usage: ResourceUsage):
        """Update CPU usage data"""
        try:
            if HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                usage.cpu_percent = process.cpu_percent(interval=0.1)
            else:
                # No good fallback for CPU usage
                pass
        except Exception as e:
            logger.error(f"Error getting CPU usage: {str(e)}")
    
    def _update_disk_usage(self, usage: ResourceUsage):
        """Update disk usage and I/O data"""
        try:
            if HAS_PSUTIL:
                # Disk space
                path = os.getcwd()  # Current working directory
                disk_usage = psutil.disk_usage(path)
                usage.disk_gb = disk_usage.used / (1024 * 1024 * 1024)
                
                # Disk I/O
                process = psutil.Process(os.getpid())
                io_counters = process.io_counters()
                
                # Calculate disk I/O rate
                current_time = time.time()
                time_diff = current_time - self._last_disk_time
                
                if time_diff > 0 and self._last_disk_read_bytes > 0:
                    read_bytes = io_counters.read_bytes - self._last_disk_read_bytes
                    write_bytes = io_counters.write_bytes - self._last_disk_write_bytes
                    
                    # Include disk I/O in the overall disk usage metric
                    io_mb = (read_bytes + write_bytes) / (1024 * 1024)
                    usage.disk_gb += io_mb / 1024
                
                # Update last values
                self._last_disk_read_bytes = io_counters.read_bytes
                self._last_disk_write_bytes = io_counters.write_bytes
                self._last_disk_time = current_time
            else:
                # Fallback to simpler disk space check
                path = os.getcwd()
                if hasattr(os, 'statvfs'):  # POSIX systems
                    statvfs = os.statvfs(path)
                    usage.disk_gb = (statvfs.f_blocks - statvfs.f_bfree) * statvfs.f_frsize / (1024 * 1024 * 1024)
        except Exception as e:
            logger.error(f"Error getting disk usage: {str(e)}")
    
    def _update_gpu_usage(self, usage: ResourceUsage):
        """Update GPU usage data"""
        try:
            if HAS_GPUTIL:
                # Get GPU memory usage
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Sum memory across all GPUs
                    total_memory_mb = 0
                    for gpu in gpus:
                        total_memory_mb += gpu.memoryUsed
                    
                    usage.gpu_memory_mb = total_memory_mb
        except Exception as e:
            logger.error(f"Error getting GPU usage: {str(e)}")
    
    def _update_network_usage(self, usage: ResourceUsage):
        """Update network usage data"""
        try:
            if HAS_PSUTIL:
                # Get network I/O
                current_time = time.time()
                net_io = psutil.net_io_counters()
                
                # Calculate network rate
                time_diff = current_time - self._last_net_time
                
                if time_diff > 0 and self._last_net_bytes_sent > 0:
                    bytes_sent = net_io.bytes_sent - self._last_net_bytes_sent
                    bytes_recv = net_io.bytes_recv - self._last_net_bytes_recv
                    
                    # Convert to Mbps
                    mbps = ((bytes_sent + bytes_recv) * 8) / (time_diff * 1_000_000)
                    usage.network_mbps = mbps
                
                # Update last values
                self._last_net_bytes_sent = net_io.bytes_sent
                self._last_net_bytes_recv = net_io.bytes_recv
                self._last_net_time = current_time
        except Exception as e:
            logger.error(f"Error getting network usage: {str(e)}")
    
    def _update_process_usage(self, usage: ResourceUsage):
        """Update process and thread counts and file handles"""
        try:
            if HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                
                # Thread count
                usage.thread_count = len(process.threads())
                
                # File handles
                usage.file_handles = process.num_fds()
                
                # Child processes
                usage.process_count = len(process.children())
            else:
                # Thread count fallback
                usage.thread_count = threading.active_count()
        except Exception as e:
            logger.error(f"Error getting process usage: {str(e)}")

class ResourceManager:
    """
    Resource management system that enforces limits
    """
    
    def __init__(self, limits: ResourceLimits = None):
        """
        Initialize resource manager
        
        Args:
            limits: Resource limits (defaults to reasonable values)
        """
        self.limits = limits or ResourceLimits()
        self.tracker = ResourceTracker()
        self.monitor_task = None
        self.running = False
        self.lock = asyncio.Lock()
        
        # Actions to take when limits are exceeded
        self.alert_callbacks = []
        self.throttle_callbacks = []
        
        # Resource reservations - track resources explicitly allocated
        self.reserved_memory_mb = 0.0
        self.reserved_disk_gb = 0.0
        self.reserved_handles = 0
        
        # Resource handle registry
        self.resource_handles = {}
        
        # Track failed allocations
        self.allocation_failures = {}
        
        logger.info(f"ResourceManager initialized with limits: "
                   f"memory={self.limits.max_memory_mb}MB, "
                   f"CPU={self.limits.max_cpu_percent}%, "
                   f"disk={self.limits.max_disk_gb}GB")
    
    async def start_monitoring(self, interval_seconds: float = 10.0):
        """
        Start resource usage monitoring
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_resources(interval_seconds))
        logger.info(f"Resource monitoring started with interval {interval_seconds}s")
    
    async def stop_monitoring(self):
        """Stop resource usage monitoring"""
        if not self.running:
            return
        
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        
        logger.info("Resource monitoring stopped")
    
    async def _monitor_resources(self, interval_seconds: float):
        """
        Monitor resource usage periodically
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        while self.running:
            try:
                # Update resource usage
                self.tracker.update()
                usage = self.tracker.last_usage
                
                # Check if any limits are exceeded
                exceeded = usage.is_exceeding_limits(self.limits)
                
                if exceeded:
                    # Log exceeded resources
                    resources_str = ", ".join([f"{r.value}: {p:.1f}x" for r, p in exceeded.items()])
                    logger.warning(f"Resource limits exceeded: {resources_str}")
                    
                    # Call alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(exceeded, usage)
                            else:
                                callback(exceeded, usage)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {str(e)}")
                    
                    # Call throttle callbacks if CPU or memory is exceeded
                    if ResourceType.CPU in exceeded or ResourceType.MEMORY in exceeded:
                        for callback in self.throttle_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(exceeded, usage)
                                else:
                                    callback(exceeded, usage)
                            except Exception as e:
                                logger.error(f"Error in throttle callback: {str(e)}")
                
                # Sleep until next check
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitor: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    def register_alert_callback(self, callback: Callable[[Dict[ResourceType, float], ResourceUsage], None]):
        """
        Register a callback for resource limit alerts
        
        Args:
            callback: Function to call when limits are exceeded
        """
        self.alert_callbacks.append(callback)
    
    def register_throttle_callback(self, callback: Callable[[Dict[ResourceType, float], ResourceUsage], None]):
        """
        Register a callback for resource throttling
        
        Args:
            callback: Function to call when throttling is needed
        """
        self.throttle_callbacks.append(callback)
    
    async def reserve_memory(self, size_mb: float, purpose: str) -> Optional[str]:
        """
        Reserve memory allocation
        
        Args:
            size_mb: Size in MB to reserve
            purpose: Purpose of the allocation
            
        Returns:
            Handle ID if successful, None if limit would be exceeded
        """
        async with self.lock:
            # Check current usage plus reservation
            usage = self.tracker.get_current_usage()
            total_memory = usage.memory_mb + self.reserved_memory_mb
            
            if total_memory + size_mb > self.limits.max_memory_mb:
                # Would exceed limit
                self._record_allocation_failure(ResourceType.MEMORY, size_mb, purpose)
                return None
            
            # Create reservation
            handle_id = str(uuid.uuid4())
            self.resource_handles[handle_id] = {
                "type": ResourceType.MEMORY,
                "size_mb": size_mb,
                "purpose": purpose,
                "timestamp": time.time()
            }
            
            self.reserved_memory_mb += size_mb
            logger.info(f"Memory reservation: {size_mb}MB for {purpose} (handle: {handle_id})")
            
            return handle_id
    
    async def reserve_disk(self, size_gb: float, purpose: str) -> Optional[str]:
        """
        Reserve disk space
        
        Args:
            size_gb: Size in GB to reserve
            purpose: Purpose of the allocation
            
        Returns:
            Handle ID if successful, None if limit would be exceeded
        """
        async with self.lock:
            # Check current usage plus reservation
            usage = self.tracker.get_current_usage()
            total_disk = usage.disk_gb + self.reserved_disk_gb
            
            if total_disk + size_gb > self.limits.max_disk_gb:
                # Would exceed limit
                self._record_allocation_failure(ResourceType.DISK, size_gb, purpose)
                return None
            
            # Create reservation
            handle_id = str(uuid.uuid4())
            self.resource_handles[handle_id] = {
                "type": ResourceType.DISK,
                "size_gb": size_gb,
                "purpose": purpose,
                "timestamp": time.time()
            }
            
            self.reserved_disk_gb += size_gb
            logger.info(f"Disk reservation: {size_gb}GB for {purpose} (handle: {handle_id})")
            
            return handle_id
    
    async def reserve_file_handles(self, count: int, purpose: str) -> Optional[str]:
        """
        Reserve file handles
        
        Args:
            count: Number of file handles to reserve
            purpose: Purpose of the allocation
            
        Returns:
            Handle ID if successful, None if limit would be exceeded
        """
        async with self.lock:
            # Check current usage plus reservation
            usage = self.tracker.get_current_usage()
            total_handles = usage.file_handles + self.reserved_handles
            
            if total_handles + count > self.limits.max_file_handles:
                # Would exceed limit
                self._record_allocation_failure(ResourceType.FILE_HANDLES, count, purpose)
                return None
            
            # Create reservation
            handle_id = str(uuid.uuid4())
            self.resource_handles[handle_id] = {
                "type": ResourceType.FILE_HANDLES,
                "count": count,
                "purpose": purpose,
                "timestamp": time.time()
            }
            
            self.reserved_handles += count
            logger.info(f"File handle reservation: {count} for {purpose} (handle: {handle_id})")
            
            return handle_id
    
    async def release_reservation(self, handle_id: str) -> bool:
        """
        Release a resource reservation
        
        Args:
            handle_id: Handle ID to release
            
        Returns:
            True if released, False if not found
        """
        async with self.lock:
            if handle_id not in self.resource_handles:
                return False
            
            reservation = self.resource_handles[handle_id]
            resource_type = reservation["type"]
            
            # Update reserved amounts
            if resource_type == ResourceType.MEMORY:
                self.reserved_memory_mb -= reservation["size_mb"]
            elif resource_type == ResourceType.DISK:
                self.reserved_disk_gb -= reservation["size_gb"]
            elif resource_type == ResourceType.FILE_HANDLES:
                self.reserved_handles -= reservation["count"]
            
            # Remove from handles
            del self.resource_handles[handle_id]
            
            logger.info(f"Released {resource_type.value} reservation (handle: {handle_id})")
            return True
    
    @asynccontextmanager
    async def memory_allocation(self, size_mb: float, purpose: str):
        """
        Context manager for temporary memory allocation
        
        Args:
            size_mb: Size in MB to allocate
            purpose: Purpose of the allocation
            
        Yields:
            None if successful
            
        Raises:
            ResourceError: If allocation failed
        """
        handle_id = await self.reserve_memory(size_mb, purpose)
        if handle_id is None:
            raise ResourceError(f"Memory allocation of {size_mb}MB failed for {purpose}")
        
        try:
            yield
        finally:
            await self.release_reservation(handle_id)
    
    @asynccontextmanager
    async def disk_allocation(self, size_gb: float, purpose: str):
        """
        Context manager for temporary disk allocation
        
        Args:
            size_gb: Size in GB to allocate
            purpose: Purpose of the allocation
            
        Yields:
            None if successful
            
        Raises:
            ResourceError: If allocation failed
        """
        handle_id = await self.reserve_disk(size_gb, purpose)
        if handle_id is None:
            raise ResourceError(f"Disk allocation of {size_gb}GB failed for {purpose}")
        
        try:
            yield
        finally:
            await self.release_reservation(handle_id)
    
    @asynccontextmanager
    async def file_handle_allocation(self, count: int, purpose: str):
        """
        Context manager for temporary file handle allocation
        
        Args:
            count: Number of file handles to allocate
            purpose: Purpose of the allocation
            
        Yields:
            None if successful
            
        Raises:
            ResourceError: If allocation failed
        """
        handle_id = await self.reserve_file_handles(count, purpose)
        if handle_id is None:
            raise ResourceError(f"File handle allocation of {count} failed for {purpose}")
        
        try:
            yield
        finally:
            await self.release_reservation(handle_id)
    
    def _record_allocation_failure(self, resource_type: ResourceType, amount: float, purpose: str):
        """Record a failed allocation"""
        if resource_type not in self.allocation_failures:
            self.allocation_failures[resource_type] = []
        
        self.allocation_failures[resource_type].append({
            "amount": amount,
            "purpose": purpose,
            "timestamp": time.time()
        })
        
        logger.warning(f"Failed to allocate {amount} of {resource_type.value} for {purpose}")
    
    async def force_garbage_collection(self) -> Tuple[int, float]:
        """
        Force garbage collection to free memory
        
        Returns:
            Tuple of (objects_collected, before_memory_mb - after_memory_mb)
        """
        # Get memory before
        self.tracker.update()
        before_memory_mb = self.tracker.last_usage.memory_mb
        
        # Run garbage collection
        collected = gc.collect(2)  # Full collection
        
        # Get memory after
        self.tracker.update()
        after_memory_mb = self.tracker.last_usage.memory_mb
        
        memory_freed = before_memory_mb - after_memory_mb
        
        logger.info(f"Forced garbage collection: {collected} objects collected, {memory_freed:.2f}MB freed")
        return collected, memory_freed
    
    def get_resource_limits(self) -> ResourceLimits:
        """
        Get current resource limits
        
        Returns:
            ResourceLimits object
        """
        return self.limits
    
    def set_resource_limits(self, limits: ResourceLimits):
        """
        Set new resource limits
        
        Args:
            limits: New resource limits
        """
        self.limits = limits
        logger.info(f"Resource limits updated: "
                  f"memory={self.limits.max_memory_mb}MB, "
                  f"CPU={self.limits.max_cpu_percent}%, "
                  f"disk={self.limits.max_disk_gb}GB")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get resource manager statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            usage = self.tracker.get_current_usage()
            peak = self.tracker.get_peak_usage()
            
            return {
                "current_usage": {
                    "memory_mb": usage.memory_mb,
                    "cpu_percent": usage.cpu_percent,
                    "disk_gb": usage.disk_gb,
                    "gpu_memory_mb": usage.gpu_memory_mb,
                    "network_mbps": usage.network_mbps,
                    "file_handles": usage.file_handles,
                    "thread_count": usage.thread_count,
                    "process_count": usage.process_count
                },
                "peak_usage": {
                    "memory_mb": peak.memory_mb,
                    "cpu_percent": peak.cpu_percent,
                    "disk_gb": peak.disk_gb,
                    "gpu_memory_mb": peak.gpu_memory_mb,
                    "network_mbps": peak.network_mbps,
                    "file_handles": peak.file_handles,
                    "thread_count": peak.thread_count,
                    "process_count": peak.process_count
                },
                "limits": {
                    "memory_mb": self.limits.max_memory_mb,
                    "cpu_percent": self.limits.max_cpu_percent,
                    "disk_gb": self.limits.max_disk_gb,
                    "gpu_memory_mb": self.limits.max_gpu_memory_mb,
                    "network_mbps": self.limits.max_network_mbps,
                    "file_handles": self.limits.max_file_handles,
                    "thread_count": self.limits.max_threads,
                    "process_count": self.limits.max_processes
                },
                "reservations": {
                    "memory_mb": self.reserved_memory_mb,
                    "disk_gb": self.reserved_disk_gb,
                    "file_handles": self.reserved_handles,
                    "active_reservations": len(self.resource_handles)
                },
                "allocation_failures": {
                    resource_type.value: len(failures)
                    for resource_type, failures in self.allocation_failures.items()
                }
            }

class ResourceError(Exception):
    """Exception raised for resource allocation failures"""
    pass

# Smart resource-aware cache
class ResourceAwareCache:
    """
    Cache that automatically adjusts its size based on memory pressure
    """
    
    def __init__(self, resource_manager: ResourceManager, max_size: int = 1000, 
                max_memory_percent: float = 20.0):
        """
        Initialize resource-aware cache
        
        Args:
            resource_manager: ResourceManager instance
            max_size: Maximum number of items to store
            max_memory_percent: Maximum percentage of total memory to use
        """
        self.resource_manager = resource_manager
        self.max_size = max_size
        self.max_memory_percent = max_memory_percent
        self.items = {}
        self.access_times = {}
        self.item_sizes = {}
        self.lock = asyncio.Lock()
        
        # Register for memory pressure alerts
        self.resource_manager.register_alert_callback(self._handle_resource_alert)
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_pressure_shrinks = 0
        
        logger.info(f"ResourceAwareCache initialized with max_size={max_size}, "
                  f"max_memory_percent={max_memory_percent}%")
    
    async def get(self, key: Any) -> Optional[Any]:
        """
        Get an item from the cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        async with self.lock:
            if key not in self.items:
                self.misses += 1
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            self.hits += 1
            
            return self.items[key]
    
    async def set(self, key: Any, value: Any, size_estimate: Optional[float] = None) -> bool:
        """
        Set an item in the cache
        
        Args:
            key: Cache key
            value: Item to cache
            size_estimate: Optional size estimate in bytes
            
        Returns:
            True if stored, False if rejected due to size
        """
        # Check if we need to estimate size
        if size_estimate is None:
            import sys
            size_estimate = self._estimate_size(value)
        
        # Skip items that are too large (>10% of cache)
        limits = self.resource_manager.get_resource_limits()
        max_item_bytes = limits.max_memory_mb * 1024 * 1024 * 0.1
        if size_estimate > max_item_bytes:
            logger.warning(f"Item too large for cache: {size_estimate} bytes")
            return False
        
        async with self.lock:
            # If key exists, update it
            if key in self.items:
                old_size = self.item_sizes.get(key, 0)
                self.items[key] = value
                self.access_times[key] = time.time()
                self.item_sizes[key] = size_estimate
                
                # No need to evict if replacing with smaller item
                if size_estimate <= old_size:
                    return True
            else:
                # Add new item
                self.items[key] = value
                self.access_times[key] = time.time()
                self.item_sizes[key] = size_estimate
            
            # Check if we need to evict items
            await self._evict_if_needed()
            
            return True
    
    async def delete(self, key: Any) -> bool:
        """
        Delete an item from the cache
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        async with self.lock:
            if key not in self.items:
                return False
            
            del self.items[key]
            del self.access_times[key]
            
            if key in self.item_sizes:
                del self.item_sizes[key]
            
            return True
    
    async def clear(self):
        """Clear the cache"""
        async with self.lock:
            self.items.clear()
            self.access_times.clear()
            self.item_sizes.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            # Calculate total size
            total_size_bytes = sum(self.item_sizes.values())
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            # Calculate hit rate
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "items": len(self.items),
                "max_size": self.max_size,
                "total_size_bytes": total_size_bytes,
                "total_size_mb": total_size_mb,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "memory_pressure_shrinks": self.memory_pressure_shrinks
            }
    
    async def _evict_if_needed(self):
        """Evict items if cache is too large"""
        # Check size limit
        if len(self.items) > self.max_size:
            await self._evict_items(len(self.items) - self.max_size)
        
        # Check memory usage
        limits = self.resource_manager.get_resource_limits()
        max_cache_bytes = limits.max_memory_mb * 1024 * 1024 * (self.max_memory_percent / 100)
        
        total_size_bytes = sum(self.item_sizes.values())
        if total_size_bytes > max_cache_bytes:
            # Calculate how many bytes to free
            bytes_to_free = total_size_bytes - max_cache_bytes
            await self._evict_by_bytes(bytes_to_free)
    
    async def _evict_items(self, count: int):
        """
        Evict a specific number of items
        
        Args:
            count: Number of items to evict
        """
        if count <= 0:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Evict oldest items
        for i in range(min(count, len(sorted_keys))):
            key = sorted_keys[i]
            del self.items[key]
            del self.access_times[key]
            
            if key in self.item_sizes:
                del self.item_sizes[key]
            
            self.evictions += 1
    
    async def _evict_by_bytes(self, bytes_to_free: float):
        """
        Evict items to free a specific amount of memory
        
        Args:
            bytes_to_free: Number of bytes to free
        """
        if bytes_to_free <= 0:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Evict oldest items until we've freed enough memory
        freed_bytes = 0
        for key in sorted_keys:
            size = self.item_sizes.get(key, 0)
            
            del self.items[key]
            del self.access_times[key]
            
            if key in self.item_sizes:
                del self.item_sizes[key]
            
            self.evictions += 1
            freed_bytes += size
            
            if freed_bytes >= bytes_to_free:
                break
    
    def _handle_resource_alert(self, exceeded: Dict[ResourceType, float], usage: ResourceUsage):
        """
        Handle resource alert
        
        Args:
            exceeded: Dictionary of exceeded resources and percentages
            usage: Current resource usage
        """
        # React to memory pressure
        if ResourceType.MEMORY in exceeded:
            # Schedule cache reduction
            asyncio.create_task(self._reduce_cache_size())
    
    async def _reduce_cache_size(self):
        """Reduce cache size when under memory pressure"""
        async with self.lock:
            # Cut cache size in half
            target_size = max(10, len(self.items) // 2)
            items_to_evict = len(self.items) - target_size
            
            if items_to_evict > 0:
                logger.info(f"Reducing cache size from {len(self.items)} to {target_size} due to memory pressure")
                await self._evict_items(items_to_evict)
                self.memory_pressure_shrinks += 1
    
    def _estimate_size(self, obj: Any) -> float:
        """
        Estimate the memory size of an object
        
        Args:
            obj: Object to measure
            
        Returns:
            Estimated size in bytes
        """
        # Import sys for size estimation
        import sys
        
        # Special handling for common types
        if isinstance(obj, (str, bytes, bytearray)):
            return len(obj)
        elif isinstance(obj, (int, float, bool, complex)):
            return sys.getsizeof(obj)
        elif isinstance(obj, dict):
            # Estimate size of keys and values
            return sys.getsizeof(obj) + sum(
                self._estimate_size(k) + self._estimate_size(v)
                for k, v in obj.items()
            )
        elif isinstance(obj, (list, tuple, set, frozenset)):
            # Estimate size of container and elements
            return sys.getsizeof(obj) + sum(
                self._estimate_size(item)
                for item in obj
            )
        else:
            # Fallback for other objects
            return sys.getsizeof(obj)

# File handle manager to prevent resource leaks
class FileHandleManager:
    """
    Manages file handles to prevent leaks
    """
    
    def __init__(self, resource_manager: ResourceManager = None):
        """
        Initialize file handle manager
        
        Args:
            resource_manager: Optional ResourceManager instance
        """
        self.resource_manager = resource_manager
        self.open_files = {}
        self.lock = asyncio.Lock()
        
        # Stats
        self.files_opened = 0
        self.files_closed = 0
        self.leak_warnings = 0
        
        # Track directories where files are opened
        self.directories = {}
        
        logger.info("FileHandleManager initialized")
    
    @contextmanager
    def open_file(self, filename: str, mode: str = "r", **kwargs) -> Any:
        """
        Context manager for safely opening files
        
        Args:
            filename: File to open
            mode: File mode
            **kwargs: Additional arguments for open()
            
        Yields:
            File object
            
        Raises:
            ResourceError: If file handle limit would be exceeded
        """
        file_id = None
        file_obj = None
        
        try:
            # Check with resource manager if available
            if self.resource_manager:
                # See if we would exceed limits
                usage = self.resource_manager.tracker.get_current_usage()
                limits = self.resource_manager.get_resource_limits()
                
                if usage.file_handles + 1 > limits.max_file_handles:
                    raise ResourceError("File handle limit would be exceeded")
            
            # Generate a unique ID for this file handle
            file_id = str(uuid.uuid4())
            
            # Open the file
            file_obj = open(filename, mode, **kwargs)
            
            # Register the file
            self._register_file(file_id, filename, file_obj, mode)
            
            # Yield the file object
            yield file_obj
            
        finally:
            # Make sure file is closed
            if file_obj is not None:
                try:
                    file_obj.close()
                except:
                    pass
            
            # Unregister file
            if file_id is not None:
                self._unregister_file(file_id)
    
    async def open_file_async(self, filename: str, mode: str = "r", **kwargs) -> Any:
        """
        Async context manager for safely opening files
        
        Args:
            filename: File to open
            mode: File mode
            **kwargs: Additional arguments for open()
            
        Returns:
            Context manager for file object
            
        Raises:
            ResourceError: If file handle limit would be exceeded
        """
        @asynccontextmanager
        async def _async_open_file():
            with self.open_file(filename, mode, **kwargs) as f:
                yield f
        
        return _async_open_file()
    
    def _register_file(self, file_id: str, filename: str, file_obj: Any, mode: str):
        """Register an opened file"""
        # Get directory
        directory = os.path.dirname(os.path.abspath(filename))
        
        with self.lock:
            self.open_files[file_id] = {
                "filename": filename,
                "file": file_obj,
                "mode": mode,
                "opened_at": time.time(),
                "thread": threading.current_thread().name,
                "directory": directory
            }
            
            self.files_opened += 1
            
            # Update directory stats
            if directory not in self.directories:
                self.directories[directory] = 0
            self.directories[directory] += 1
    
    def _unregister_file(self, file_id: str):
        """Unregister a file when closed"""
        with self.lock:
            if file_id in self.open_files:
                file_info = self.open_files[file_id]
                
                # Update directory stats
                directory = file_info["directory"]
                if directory in self.directories:
                    self.directories[directory] -= 1
                    if self.directories[directory] <= 0:
                        del self.directories[directory]
                
                del self.open_files[file_id]
                self.files_closed += 1
    
    async def check_for_leaks(self, max_age_seconds: float = 300.0) -> List[Dict[str, Any]]:
        """
        Check for potential file handle leaks
        
        Args:
            max_age_seconds: Maximum age before considering a file leaked
            
        Returns:
            List of potential leaks
        """
        now = time.time()
        leaks = []
        
        async with self.lock:
            for file_id, file_info in self.open_files.items():
                age = now - file_info["opened_at"]
                if age > max_age_seconds:
                    leak_info = {
                        "file_id": file_id,
                        "filename": file_info["filename"],
                        "mode": file_info["mode"],
                        "age_seconds": age,
                        "thread": file_info["thread"]
                    }
                    leaks.append(leak_info)
                    self.leak_warnings += 1
        
        return leaks
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get file handle statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            # Group files by directory
            by_directory = {}
            for directory, count in self.directories.items():
                by_directory[directory] = count
            
            return {
                "open_files": len(self.open_files),
                "files_opened": self.files_opened,
                "files_closed": self.files_closed,
                "leak_warnings": self.leak_warnings,
                "by_directory": by_directory
            }
    
    async def force_close_all(self) -> int:
        """
        Force close all open files
        
        Returns:
            Number of files closed
        """
        closed_count = 0
        
        async with self.lock:
            # Create a copy of the open_files dict to avoid modification during iteration
            open_files = dict(self.open_files)
            
            for file_id, file_info in open_files.items():
                file_obj = file_info["file"]
                try:
                    file_obj.close()
                    self._unregister_file(file_id)
                    closed_count += 1
                except Exception as e:
                    logger.error(f"Error closing file {file_info['filename']}: {str(e)}")
        
        return closed_count

# Memory usage watcher for detecting leaks
class MemoryLeakDetector:
    """
    Detects potential memory leaks by tracking memory usage patterns
    """
    
    def __init__(self, resource_manager: ResourceManager, 
                growth_threshold_mb: float = 10.0,
                window_size: int = 10):
        """
        Initialize memory leak detector
        
        Args:
            resource_manager: ResourceManager instance
            growth_threshold_mb: Memory growth threshold for leak detection
            window_size: Number of samples to use for trend analysis
        """
        self.resource_manager = resource_manager
        self.growth_threshold_mb = growth_threshold_mb
        self.window_size = window_size
        
        # Memory usage history
        self.memory_history = []
        self.timestamp_history = []
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Stats
        self.leak_alerts = 0
        self.last_alert_time = 0
        
        logger.info(f"MemoryLeakDetector initialized with growth_threshold={growth_threshold_mb}MB")
    
    async def sample_memory(self):
        """Take a memory usage sample"""
        # Get current usage
        usage = self.resource_manager.tracker.get_current_usage()
        memory_mb = usage.memory_mb
        timestamp = time.time()
        
        # Add to history
        self.memory_history.append(memory_mb)
        self.timestamp_history.append(timestamp)
        
        # Keep history at window size
        if len(self.memory_history) > self.window_size:
            self.memory_history.pop(0)
            self.timestamp_history.pop(0)
        
        # Check for leaks if we have enough history
        if len(self.memory_history) == self.window_size:
            await self._check_for_leak()
    
    async def _check_for_leak(self):
        """Check for memory leaks"""
        # Need at least 2 samples
        if len(self.memory_history) < 2:
            return
        
        # Calculate memory growth rate (MB/s)
        memory_diff = self.memory_history[-1] - self.memory_history[0]
        time_diff = self.timestamp_history[-1] - self.timestamp_history[0]
        
        if time_diff <= 0:
            return
        
        growth_rate = memory_diff / time_diff  # MB/s
        
        # Check if growth rate is concerning
        if memory_diff > self.growth_threshold_mb and growth_rate > 0:
            # Only alert once per minute
            now = time.time()
            if now - self.last_alert_time > 60:
                self.last_alert_time = now
                self.leak_alerts += 1
                
                leak_info = {
                    "memory_diff_mb": memory_diff,
                    "time_period_seconds": time_diff,
                    "growth_rate_mb_per_second": growth_rate,
                    "current_memory_mb": self.memory_history[-1],
                    "timestamp": now
                }
                
                logger.warning(f"Potential memory leak detected: {memory_diff:.2f}MB over "
                             f"{time_diff:.1f}s ({growth_rate:.2f}MB/s)")
                
                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(leak_info)
                        else:
                            callback(leak_info)
                    except Exception as e:
                        logger.error(f"Error in leak alert callback: {str(e)}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback for leak alerts
        
        Args:
            callback: Function to call when a leak is detected
        """
        self.alert_callbacks.append(callback)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get leak detector statistics
        
        Returns:
            Dictionary of statistics
        """
        # Calculate recent growth if we have enough history
        growth_rate = 0.0
        if len(self.memory_history) >= 2:
            memory_diff = self.memory_history[-1] - self.memory_history[0]
            time_diff = self.timestamp_history[-1] - self.timestamp_history[0]
            if time_diff > 0:
                growth_rate = memory_diff / time_diff  # MB/s
        
        return {
            "current_memory_mb": self.memory_history[-1] if self.memory_history else 0,
            "history_points": len(self.memory_history),
            "window_size": self.window_size,
            "growth_threshold_mb": self.growth_threshold_mb,
            "recent_growth_rate_mb_per_second": growth_rate,
            "leak_alerts": self.leak_alerts,
            "last_alert_time": self.last_alert_time
        }

# Singleton resource manager
_resource_manager = None

def get_resource_manager() -> ResourceManager:
    """
    Get the singleton resource manager instance
    
    Returns:
        ResourceManager instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    
    return _resource_manager