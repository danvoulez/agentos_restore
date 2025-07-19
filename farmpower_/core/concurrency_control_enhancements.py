"""
Enhanced Concurrency Control for LogLineOS
Provides advanced concurrency mechanisms and patterns
Created: 2025-07-19 06:43:11 UTC
User: danvoulez
"""
import os
import time
import logging
import asyncio
import threading
import uuid
import inspect
import weakref
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from contextlib import asynccontextmanager, contextmanager
import random
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/concurrency_enhancements.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConcurrencyControlEnhancements")

# Import from core if available
try:
    from core.concurrency_control import KeyLock, Actor, ActorSystem, OptimisticLock, ReadWriteLock
except ImportError:
    logger.warning("Could not import from core.concurrency_control, some features may be limited")

class DistributedLock:
    """
    Distributed lock implementation using Redis or other backend
    Enables locking across multiple processes or machines
    """
    
    def __init__(self, resource_name: str, 
                ttl_ms: int = 30000, 
                retry_interval_ms: int = 100,
                retry_timeout_ms: int = 10000,
                redis_client = None,
                local_lock: Optional[asyncio.Lock] = None):
        """
        Initialize distributed lock
        
        Args:
            resource_name: Name of resource to lock
            ttl_ms: Time-to-live in milliseconds
            retry_interval_ms: Retry interval in milliseconds
            retry_timeout_ms: Retry timeout in milliseconds
            redis_client: Redis client instance (if None, falls back to local)
            local_lock: Local lock for single-process fallback
        """
        self.resource_name = resource_name
        self.ttl_ms = ttl_ms
        self.retry_interval_ms = retry_interval_ms
        self.retry_timeout_ms = retry_timeout_ms
        self.redis_client = redis_client
        self.local_lock = local_lock or asyncio.Lock()
        self.owner = None
        self.acquired = False
        self.refresh_task = None
        
        # Generate a unique token for this lock instance
        self.lock_token = f"{os.getpid()}-{uuid.uuid4()}"
        
        # Stats
        self.acquire_attempts = 0
        self.successful_acquires = 0
        self.failed_acquires = 0
        self.refreshes = 0
        
        logger.info(f"Distributed lock created for {resource_name} with token {self.lock_token}")
    
    @asynccontextmanager
    async def acquire(self, timeout_ms: Optional[int] = None):
        """
        Acquire the distributed lock
        
        Args:
            timeout_ms: Optional timeout in milliseconds (None for retry_timeout_ms)
            
        Yields:
            True if lock acquired
            
        Raises:
            TimeoutError: If lock couldn't be acquired within timeout
        """
        acquired = False
        timeout = timeout_ms if timeout_ms is not None else self.retry_timeout_ms
        
        # Try to acquire the lock
        try:
            acquired = await self._acquire_lock(timeout)
            if not acquired:
                raise TimeoutError(f"Failed to acquire distributed lock for {self.resource_name}")
            
            # Start refresh task to maintain the lock
            self.refresh_task = asyncio.create_task(self._refresh_lock())
            
            # Yield control with lock acquired
            yield True
            
        finally:
            # Release the lock when done
            if acquired:
                await self._release_lock()
                if self.refresh_task:
                    self.refresh_task.cancel()
                    try:
                        await self.refresh_task
                    except asyncio.CancelledError:
                        pass
                    self.refresh_task = None
    
    async def _acquire_lock(self, timeout_ms: int) -> bool:
        """
        Acquire the lock with retries
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if acquired, False if timeout
        """
        self.acquire_attempts += 1
        start_time = time.time()
        deadline = start_time + (timeout_ms / 1000)
        
        # Try to acquire until timeout
        while time.time() < deadline:
            # Try Redis if available
            if self.redis_client:
                success = await self._acquire_with_redis()
            else:
                # Fall back to local lock
                success = await self._acquire_local()
            
            if success:
                self.acquired = True
                self.successful_acquires += 1
                return True
            
            # Wait before retry
            await asyncio.sleep(self.retry_interval_ms / 1000)
        
        # Timeout
        self.failed_acquires += 1
        return False
    
    async def _acquire_with_redis(self) -> bool:
        """
        Acquire lock using Redis
        
        Returns:
            True if acquired
        """
        try:
            # Use Redis SET NX with TTL
            result = await self.redis_client.set(
                f"lock:{self.resource_name}",
                self.lock_token,
                px=self.ttl_ms,
                nx=True
            )
            
            if result:
                self.owner = self.lock_token
                logger.debug(f"Acquired Redis lock for {self.resource_name}")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Redis lock error: {str(e)}")
            return False
    
    async def _acquire_local(self) -> bool:
        """
        Acquire local lock
        
        Returns:
            True if acquired
        """
        # For local lock, just try to acquire without blocking
        acquired = self.local_lock.locked()
        if not acquired:
            acquired = self.local_lock.acquire()
            if acquired:
                self.owner = self.lock_token
                logger.debug(f"Acquired local lock for {self.resource_name}")
                return True
        return False
    
    async def _refresh_lock(self):
        """Periodically refresh the lock to maintain ownership"""
        try:
            # Refresh at 1/3 of the TTL interval
            refresh_interval = self.ttl_ms / 3000  # Convert to seconds
            
            while True:
                await asyncio.sleep(refresh_interval)
                
                if self.redis_client:
                    await self._refresh_with_redis()
                # Local locks don't need refreshing
                
                self.refreshes += 1
        except asyncio.CancelledError:
            # Normal cancellation when lock is released
            pass
        except Exception as e:
            logger.error(f"Error in lock refresh: {str(e)}")
    
    async def _refresh_with_redis(self) -> bool:
        """
        Refresh lock using Redis
        
        Returns:
            True if refreshed
        """
        try:
            # Use Lua script to check owner and extend
            script = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('pexpire', KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            
            result = await self.redis_client.eval(
                script,
                keys=[f"lock:{self.resource_name}"],
                args=[self.lock_token, self.ttl_ms]
            )
            
            return bool(result)
        except Exception as e:
            logger.error(f"Redis refresh error: {str(e)}")
            return False
    
    async def _release_lock(self) -> bool:
        """
        Release the lock
        
        Returns:
            True if released
        """
        if not self.acquired:
            return False
        
        try:
            if self.redis_client:
                await self._release_with_redis()
            else:
                self._release_local()
            
            self.acquired = False
            self.owner = None
            return True
        except Exception as e:
            logger.error(f"Error releasing lock: {str(e)}")
            return False
    
    async def _release_with_redis(self) -> bool:
        """
        Release Redis lock
        
        Returns:
            True if released
        """
        try:
            # Use Lua script to check owner before deleting
            script = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            else
                return 0
            end
            """
            
            result = await self.redis_client.eval(
                script,
                keys=[f"lock:{self.resource_name}"],
                args=[self.lock_token]
            )
            
            success = bool(result)
            if success:
                logger.debug(f"Released Redis lock for {self.resource_name}")
            return success
        except Exception as e:
            logger.error(f"Redis release error: {str(e)}")
            return False
    
    def _release_local(self) -> bool:
        """
        Release local lock
        
        Returns:
            True if released
        """
        try:
            if self.local_lock.locked():
                self.local_lock.release()
                logger.debug(f"Released local lock for {self.resource_name}")
                return True
            return False
        except RuntimeError:
            # Not holding the lock
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get lock statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            "resource_name": self.resource_name,
            "acquired": self.acquired,
            "owner": self.owner,
            "acquire_attempts": self.acquire_attempts,
            "successful_acquires": self.successful_acquires,
            "failed_acquires": self.failed_acquires,
            "refreshes": self.refreshes,
            "backend": "redis" if self.redis_client else "local"
        }

class AsyncExecutorPool:
    """
    Pool of worker threads for CPU-bound tasks
    Prevents blocking the event loop with intensive operations
    """
    
    def __init__(self, max_workers: int = None, thread_name_prefix: str = "AsyncWorker"):
        """
        Initialize async executor pool
        
        Args:
            max_workers: Maximum worker threads (None for CPU count * 5)
            thread_name_prefix: Prefix for thread names
        """
        # Default to CPU count * 5 if not specified
        if max_workers is None:
            import multiprocessing
            max_workers = multiprocessing.cpu_count() * 5
        
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        # Stats
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.active_tasks = 0
        self._lock = threading.RLock()
        
        logger.info(f"AsyncExecutorPool initialized with {max_workers} workers")
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Submit a task to the executor
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
        """
        loop = asyncio.get_running_loop()
        
        with self._lock:
            self.tasks_submitted += 1
            self.active_tasks += 1
        
        try:
            # Submit task to executor
            result = await loop.run_in_executor(
                self.executor,
                lambda: func(*args, **kwargs)
            )
            
            with self._lock:
                self.tasks_completed += 1
                self.active_tasks -= 1
            
            return result
        except Exception as e:
            with self._lock:
                self.tasks_failed += 1
                self.active_tasks -= 1
            
            # Re-raise the exception
            raise
    
    async def map(self, func: Callable, items: List[Any], timeout: Optional[float] = None) -> List[Any]:
        """
        Map a function over items in parallel
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            timeout: Optional timeout in seconds
            
        Returns:
            List of results
        """
        # Create tasks
        tasks = [self.submit(func, item) for item in items]
        
        # Wait for all tasks to complete
        if timeout is not None:
            results = await asyncio.gather(*tasks, return_exceptions=True, timeout=timeout)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with error: {str(result)}")
        
        return results
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor
        
        Args:
            wait: Whether to wait for pending tasks
        """
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get executor statistics
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "tasks_submitted": self.tasks_submitted,
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
                "active_tasks": self.active_tasks,
                "completion_rate": self.tasks_completed / max(1, self.tasks_submitted),
                "failure_rate": self.tasks_failed / max(1, self.tasks_submitted)
            }

class Semaphore:
    """
    Enhanced semaphore with timeout and statistics
    """
    
    def __init__(self, value: int = 1, name: str = None):
        """
        Initialize semaphore
        
        Args:
            value: Initial value
            name: Optional name for the semaphore
        """
        self.semaphore = asyncio.Semaphore(value)
        self.name = name or f"sem-{uuid.uuid4()}"
        self.initial_value = value
        
        # Stats
        self.acquire_count = 0
        self.release_count = 0
        self.timeout_count = 0
        self.contention_count = 0
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire the semaphore
        
        Args:
            timeout: Timeout in seconds
            
        Yields:
            None if acquired
            
        Raises:
            TimeoutError: If semaphore couldn't be acquired within timeout
        """
        start_time = time.time()
        
        # Try to acquire with timeout
        if timeout is not None:
            try:
                acquired = False
                try:
                    acquired = await asyncio.wait_for(self.semaphore.acquire(), timeout)
                except asyncio.TimeoutError:
                    async with self._lock:
                        self.timeout_count += 1
                        self.contention_count += 1
                    raise TimeoutError(f"Timeout acquiring semaphore {self.name}")
                
                if not acquired:
                    async with self._lock:
                        self.contention_count += 1
                    raise TimeoutError(f"Failed to acquire semaphore {self.name}")
            except Exception as e:
                if not isinstance(e, TimeoutError):
                    # Wrap other exceptions
                    raise TimeoutError(f"Error acquiring semaphore {self.name}: {str(e)}")
                raise
        else:
            # No timeout, just wait
            await self.semaphore.acquire()
        
        # Update stats
        async with self._lock:
            self.acquire_count += 1
            
            # If acquisition took significant time, count as contention
            if time.time() - start_time > 0.1:  # 100ms
                self.contention_count += 1
        
        try:
            # Yield control with semaphore acquired
            yield
        finally:
            # Release the semaphore
            self.semaphore.release()
            
            # Update stats
            async with self._lock:
                self.release_count += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get semaphore statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self._lock:
            return {
                "name": self.name,
                "initial_value": self.initial_value,
                "acquire_count": self.acquire_count,
                "release_count": self.release_count,
                "timeout_count": self.timeout_count,
                "contention_count": self.contention_count,
                "contention_rate": self.contention_count / max(1, self.acquire_count)
            }

class TaskGroup:
    """
    Group of related tasks with management capabilities
    Similar to Python 3.11+ TaskGroup but with more features
    """
    
    def __init__(self, name: str = None, limit: int = 0, timeout: float = None):
        """
        Initialize task group
        
        Args:
            name: Optional name for the group
            limit: Max concurrent tasks (0 for unlimited)
            timeout: Default timeout for tasks
        """
        self.name = name or f"taskgroup-{uuid.uuid4()}"
        self.limit = limit
        self.default_timeout = timeout
        
        # Task management
        self.tasks = {}
        self.semaphore = asyncio.Semaphore(limit) if limit > 0 else None
        self.active = True
        
        # Stats
        self.tasks_created = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.tasks_cancelled = 0
        
        logger.info(f"TaskGroup {self.name} initialized with limit={limit}")
    
    async def create_task(self, coro, *, name: str = None, timeout: float = None) -> asyncio.Task:
        """
        Create a managed task
        
        Args:
            coro: Coroutine to run
            name: Optional task name
            timeout: Optional timeout (overrides default)
            
        Returns:
            asyncio.Task instance
            
        Raises:
            RuntimeError: If group is not active
        """
        if not self.active:
            raise RuntimeError(f"TaskGroup {self.name} is not active")
        
        # Apply timeout if specified
        effective_timeout = timeout or self.default_timeout
        if effective_timeout is not None:
            coro = self._with_timeout(coro, effective_timeout)
        
        # Apply concurrency limit if specified
        if self.semaphore is not None:
            coro = self._with_semaphore(coro)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        task_name = name or f"task-{task_id}"
        
        # Create task
        task = asyncio.create_task(coro, name=task_name)
        task._task_id = task_id
        
        # Add task callback
        task.add_done_callback(self._task_done_callback)
        
        # Register task
        self.tasks[task_id] = {
            "task": task,
            "name": task_name,
            "created_at": time.time()
        }
        
        self.tasks_created += 1
        return task
    
    async def _with_timeout(self, coro, timeout: float):
        """Add timeout to a coroutine"""
        try:
            return await asyncio.wait_for(coro, timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Task in group {self.name} timed out after {timeout}s")
            raise
    
    async def _with_semaphore(self, coro):
        """Add semaphore to limit concurrency"""
        async with self.semaphore:
            return await coro
    
    def _task_done_callback(self, task):
        """Callback when a task completes"""
        task_id = getattr(task, '_task_id', None)
        if task_id is None or task_id not in self.tasks:
            return
        
        # Update stats based on outcome
        try:
            exception = task.exception()
            if exception is None:
                self.tasks_completed += 1
            else:
                self.tasks_failed += 1
                # Log the exception
                logger.error(f"Task {self.tasks[task_id]['name']} in group {self.name} failed: {str(exception)}")
        except asyncio.CancelledError:
            self.tasks_cancelled += 1
        
        # Remove from active tasks
        if self.active:
            del self.tasks[task_id]
    
    async def cancel_all(self) -> int:
        """
        Cancel all active tasks
        
        Returns:
            Number of tasks cancelled
        """
        cancelled = 0
        for task_info in list(self.tasks.values()):
            task = task_info["task"]
            if not task.done():
                task.cancel()
                cancelled += 1
        
        return cancelled
    
    async def wait_all(self, timeout: float = None) -> Tuple[List[asyncio.Task], List[asyncio.Task]]:
        """
        Wait for all tasks to complete
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Tuple of (done_tasks, pending_tasks)
        """
        tasks = [info["task"] for info in self.tasks.values()]
        
        if not tasks:
            return [], []
        
        if timeout is not None:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            return list(done), list(pending)
        else:
            done = await asyncio.gather(*tasks, return_exceptions=True)
            return tasks, []
    
    async def shutdown(self, cancel: bool = False, timeout: float = None) -> int:
        """
        Shutdown the task group
        
        Args:
            cancel: Whether to cancel active tasks
            timeout: Optional timeout for waiting
            
        Returns:
            Number of tasks affected
        """
        self.active = False
        affected = 0
        
        if cancel:
            affected = await self.cancel_all()
        
        if timeout is not None:
            await self.wait_all(timeout)
        
        return affected
    
    @asynccontextmanager
    async def as_context(self):
        """
        Use task group as a context manager
        
        Yields:
            TaskGroup instance
        """
        try:
            yield self
        finally:
            await self.shutdown(cancel=True)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get task group statistics
        
        Returns:
            Dictionary of statistics
        """
        active_tasks = len(self.tasks)
        
        return {
            "name": self.name,
            "active": self.active,
            "limit": self.limit,
            "active_tasks": active_tasks,
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_cancelled": self.tasks_cancelled,
            "completion_rate": self.tasks_completed / max(1, self.tasks_created),
            "failure_rate": self.tasks_failed / max(1, self.tasks_created)
        }

class BatchProcessor:
    """
    Process items in batches for efficiency
    """
    
    def __init__(self, batch_size: int = 100, max_delay_ms: int = 50):
        """
        Initialize batch processor
        
        Args:
            batch_size: Maximum items per batch
            max_delay_ms: Maximum delay before processing incomplete batch
        """
        self.batch_size = batch_size
        self.max_delay_ms = max_delay_ms
        
        # Batch state
        self.current_batch = []
        self.batch_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.processing = False
        self.timer_task = None
        
        # Stats
        self.items_added = 0
        self.batches_processed = 0
        self.items_processed = 0
        self.timer_triggered = 0
        self.size_triggered = 0
        
        logger.info(f"BatchProcessor initialized with batch_size={batch_size}, max_delay={max_delay_ms}ms")
    
    async def add_item(self, item: Any) -> None:
        """
        Add an item to the current batch
        
        Args:
            item: Item to add
        """
        async with self.lock:
            self.current_batch.append(item)
            self.items_added += 1
            
            # Start timer if this is the first item
            if len(self.current_batch) == 1:
                self._start_timer()
            
            # Check if batch is full
            if len(self.current_batch) >= self.batch_size:
                self.batch_event.set()
                self.size_triggered += 1
    
    async def add_items(self, items: List[Any]) -> None:
        """
        Add multiple items to the current batch
        
        Args:
            items: Items to add
        """
        if not items:
            return
        
        async with self.lock:
            self.current_batch.extend(items)
            self.items_added += len(items)
            
            # Start timer if these are the first items
            if len(self.current_batch) == len(items):
                self._start_timer()
            
            # Check if batch is full
            if len(self.current_batch) >= self.batch_size:
                self.batch_event.set()
                self.size_triggered += 1
    
    def _start_timer(self) -> None:
        """Start timer for batch processing"""
        if self.timer_task is not None:
            self.timer_task.cancel()
        
        self.timer_task = asyncio.create_task(self._timer())
    
    async def _timer(self) -> None:
        """Timer for delayed batch processing"""
        try:
            await asyncio.sleep(self.max_delay_ms / 1000)
            # Trigger batch processing
            self.batch_event.set()
            self.timer_triggered += 1
        except asyncio.CancelledError:
            # Timer cancelled (likely due to batch being full)
            pass
    
    async def process_batches(self, processor_func: Callable[[List[Any]], Any]) -> None:
        """
        Process batches as they become available
        
        Args:
            processor_func: Function to process each batch
        """
        self.processing = True
        
        try:
            while self.processing:
                # Wait for batch
                await self.batch_event.wait()
                
                # Get the current batch and reset
                async with self.lock:
                    batch = self.current_batch
                    self.current_batch = []
                    self.batch_event.clear()
                    
                    # Cancel timer if it's running
                    if self.timer_task:
                        self.timer_task.cancel()
                        self.timer_task = None
                
                if batch:
                    # Process the batch
                    try:
                        if asyncio.iscoroutinefunction(processor_func):
                            await processor_func(batch)
                        else:
                            processor_func(batch)
                        
                        # Update stats
                        async with self.lock:
                            self.batches_processed += 1
                            self.items_processed += len(batch)
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
        finally:
            self.processing = False
    
    async def stop(self) -> None:
        """Stop batch processing"""
        self.processing = False
        
        # Process any remaining items
        if self.timer_task:
            self.timer_task.cancel()
            self.timer_task = None
        
        # Signal any waiting processors
        self.batch_event.set()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get batch processor statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            current_batch_size = len(self.current_batch)
            
            return {
                "batch_size": self.batch_size,
                "max_delay_ms": self.max_delay_ms,
                "current_batch_size": current_batch_size,
                "items_added": self.items_added,
                "batches_processed": self.batches_processed,
                "items_processed": self.items_processed,
                "average_batch_size": self.items_processed / max(1, self.batches_processed),
                "timer_triggered": self.timer_triggered,
                "size_triggered": self.size_triggered
            }

class AsyncPool:
    """
    Pool of reusable async resources
    """
    
    def __init__(self, 
                factory: Callable[[], Any], 
                max_size: int = 10, 
                min_size: int = 2,
                max_idle_time_seconds: float = 60.0,
                name: str = None):
        """
        Initialize async resource pool
        
        Args:
            factory: Function to create new resources
            max_size: Maximum pool size
            min_size: Minimum pool size
            max_idle_time_seconds: Maximum idle time before closing excess resources
            name: Optional pool name
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time_seconds
        self.name = name or f"pool-{uuid.uuid4()}"
        
        # Pool state
        self.resources = []  # Free resources
        self.in_use = {}     # Resources currently in use
        self.creation_times = {}  # When resources were created
        self.last_used = {}  # When resources were last used
        self.lock = asyncio.Lock()
        
        # Maintenance task
        self.maintenance_task = None
        self.shutdown_event = asyncio.Event()
        
        # Stats
        self.created = 0
        self.acquired = 0
        self.released = 0
        self.closed = 0
        self.wait_time_total = 0.0
        
        logger.info(f"AsyncPool {self.name} initialized with max_size={max_size}, min_size={min_size}")
    
    async def start(self):
        """Start the resource pool and maintenance task"""
        # Pre-create minimum resources
        async with self.lock:
            for _ in range(self.min_size):
                await self._create_resource()
        
        # Start maintenance task
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
    
    async def stop(self):
        """Stop the resource pool"""
        # Signal maintenance task to stop
        self.shutdown_event.set()
        
        # Wait for maintenance task to finish
        if self.maintenance_task:
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
            self.maintenance_task = None
        
        # Close all resources
        async with self.lock:
            # Close free resources
            for resource in self.resources:
                await self._close_resource(resource)
            self.resources = []
            
            # Close in-use resources
            for resource in list(self.in_use.keys()):
                await self._close_resource(resource)
            self.in_use = {}
            
            # Clear tracking
            self.creation_times = {}
            self.last_used = {}
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire a resource from the pool
        
        Args:
            timeout: Optional timeout in seconds
            
        Yields:
            Resource from the pool
            
        Raises:
            TimeoutError: If resource couldn't be acquired within timeout
        """
        start_time = time.time()
        resource = await self._acquire_resource(timeout)
        
        try:
            yield resource
        finally:
            # Return resource to the pool
            await self._release_resource(resource)
            
            # Update wait time stats
            wait_time = time.time() - start_time
            self.wait_time_total += wait_time
    
    async def _acquire_resource(self, timeout: Optional[float] = None) -> Any:
        """
        Acquire a resource with timeout
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Resource
            
        Raises:
            TimeoutError: If resource couldn't be acquired within timeout
        """
        deadline = time.time() + timeout if timeout is not None else None
        
        while True:
            # Check if we've exceeded timeout
            if deadline and time.time() > deadline:
                raise TimeoutError(f"Timeout acquiring resource from pool {self.name}")
            
            # Try to get a free resource
            async with self.lock:
                if self.resources:
                    # Get a resource from the free list
                    resource = self.resources.pop()
                    
                    # Mark as in use
                    self.in_use[resource] = time.time()
                    
                    # Update stats
                    self.acquired += 1
                    
                    return resource
                
                # No free resources, can we create a new one?
                total_resources = len(self.resources) + len(self.in_use)
                if total_resources < self.max_size:
                    # Create a new resource
                    resource = await self._create_resource()
                    
                    # Mark as in use
                    self.in_use[resource] = time.time()
                    
                    # Update stats
                    self.acquired += 1
                    
                    return resource
            
            # All resources in use and at max size, wait a bit
            remaining = deadline - time.time() if deadline else None
            try:
                await asyncio.sleep(min(0.1, remaining) if remaining is not None else 0.1)
            except:
                pass
    
    async def _release_resource(self, resource: Any) -> None:
        """
        Release a resource back to the pool
        
        Args:
            resource: Resource to release
        """
        async with self.lock:
            if resource not in self.in_use:
                # Resource not tracked as in use
                logger.warning(f"Untracked resource returned to pool {self.name}")
                return
            
            # Remove from in_use
            del self.in_use[resource]
            
            # Update last used time
            self.last_used[resource] = time.time()
            
            # Put back in the free list
            self.resources.append(resource)
            
            # Update stats
            self.released += 1
    
    async def _create_resource(self) -> Any:
        """
        Create a new resource
        
        Returns:
            New resource
        """
        # Call factory function
        if asyncio.iscoroutinefunction(self.factory):
            resource = await self.factory()
        else:
            resource = self.factory()
        
        # Track resource
        now = time.time()
        self.creation_times[resource] = now
        self.last_used[resource] = now
        
        # Update stats
        self.created += 1
        
        logger.debug(f"Created new resource in pool {self.name} (total: {self.created})")
        return resource
    
    async def _close_resource(self, resource: Any) -> None:
        """
        Close a resource
        
        Args:
            resource: Resource to close
        """
        try:
            # Try to close the resource
            if hasattr(resource, "close"):
                if asyncio.iscoroutinefunction(resource.close):
                    await resource.close()
                else:
                    resource.close()
            elif hasattr(resource, "disconnect"):
                if asyncio.iscoroutinefunction(resource.disconnect):
                    await resource.disconnect()
                else:
                    resource.disconnect()
        except Exception as e:
            logger.error(f"Error closing resource in pool {self.name}: {str(e)}")
        
        # Clean up tracking
        if resource in self.creation_times:
            del self.creation_times[resource]
        
        if resource in self.last_used:
            del self.last_used[resource]
        
        # Update stats
        self.closed += 1
    
    async def _maintenance_loop(self) -> None:
        """Maintenance loop to manage pool resources"""
        try:
            while not self.shutdown_event.is_set():
                # Wait a bit
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=10.0)
                    if self.shutdown_event.is_set():
                        break
                except asyncio.TimeoutError:
                    pass
                
                # Perform maintenance
                await self._clean_idle_resources()
                
        except asyncio.CancelledError:
            # Normal cancellation during shutdown
            pass
        except Exception as e:
            logger.error(f"Error in pool {self.name} maintenance loop: {str(e)}")
    
    async def _clean_idle_resources(self) -> None:
        """Clean up idle resources beyond the minimum size"""
        async with self.lock:
            now = time.time()
            
            # Only clean if we have more than min_size resources
            total_resources = len(self.resources) + len(self.in_use)
            if total_resources <= self.min_size:
                return
            
            # Find idle resources to close
            resources_to_close = []
            
            for resource in self.resources:
                last_used_time = self.last_used.get(resource, 0)
                idle_time = now - last_used_time
                
                # Close if idle for too long and above min_size
                if idle_time > self.max_idle_time and (total_resources - len(resources_to_close)) > self.min_size:
                    resources_to_close.append(resource)
            
            # Close idle resources
            for resource in resources_to_close:
                self.resources.remove(resource)
                await self._close_resource(resource)
                logger.debug(f"Closed idle resource in pool {self.name}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            total_resources = len(self.resources) + len(self.in_use)
            avg_wait_time = self.wait_time_total / max(1, self.acquired)
            
            return {
                "name": self.name,
                "free_resources": len(self.resources),
                "in_use_resources": len(self.in_use),
                "total_resources": total_resources,
                "max_size": self.max_size,
                "min_size": self.min_size,
                "created": self.created,
                "acquired": self.acquired,
                "released": self.released,
                "closed": self.closed,
                "utilization": len(self.in_use) / max(1, total_resources),
                "average_wait_time": avg_wait_time
            }