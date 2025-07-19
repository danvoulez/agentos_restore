"""
Concurrency Control System for LogLineOS
Provides advanced concurrency patterns and utilities
Created: 2025-07-19 06:33:24 UTC
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
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from contextlib import asynccontextmanager, contextmanager
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/concurrency.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConcurrencyControl")

# Fine-grained locking system
class KeyLock:
    """
    Fine-grained locking system that provides separate locks for different keys
    Avoids lock contention when different operations target different resources
    """
    
    def __init__(self, lock_type: str = "async"):
        """
        Initialize key lock system
        
        Args:
            lock_type: Type of locks to use ("async" or "thread")
        """
        self.lock_type = lock_type
        self.locks = {}
        self.master_lock = asyncio.Lock() if lock_type == "async" else threading.RLock()
        
        # Stats
        self.lock_acquisitions = 0
        self.lock_contentions = 0
        self.keys_created = 0
        self.lock_wait_time_total = 0.0
        
        logger.info(f"KeyLock initialized with lock_type={lock_type}")
    
    async def get_lock_async(self, key: Any) -> asyncio.Lock:
        """
        Get or create an async lock for a specific key
        
        Args:
            key: The key to lock on
            
        Returns:
            An asyncio.Lock instance
        """
        if self.lock_type != "async":
            raise ValueError("Cannot get async lock with non-async lock type")
        
        # Using async context manager to safely access the locks dict
        async with self.master_lock:
            if key not in self.locks:
                self.locks[key] = asyncio.Lock()
                self.keys_created += 1
            return self.locks[key]
    
    def get_lock_sync(self, key: Any) -> threading.Lock:
        """
        Get or create a threading lock for a specific key
        
        Args:
            key: The key to lock on
            
        Returns:
            A threading.Lock instance
        """
        if self.lock_type != "thread":
            raise ValueError("Cannot get thread lock with non-thread lock type")
        
        # Using context manager to safely access the locks dict
        with self.master_lock:
            if key not in self.locks:
                self.locks[key] = threading.RLock()
                self.keys_created += 1
            return self.locks[key]
    
    @asynccontextmanager
    async def acquire_async(self, key: Any, timeout: Optional[float] = None):
        """
        Acquire an async lock for a specific key
        
        Args:
            key: The key to lock on
            timeout: Optional timeout in seconds
            
        Yields:
            None when lock is acquired
            
        Raises:
            TimeoutError: If lock couldn't be acquired within timeout
        """
        lock = await self.get_lock_async(key)
        
        start_time = time.time()
        
        try:
            # Try to acquire the lock with timeout
            if timeout is not None:
                acquired = False
                try:
                    acquired = await asyncio.wait_for(lock.acquire(), timeout)
                except asyncio.TimeoutError:
                    self.lock_contentions += 1
                    raise TimeoutError(f"Timeout acquiring lock for key {key}")
                
                if not acquired:
                    self.lock_contentions += 1
                    raise TimeoutError(f"Failed to acquire lock for key {key}")
            else:
                # No timeout, just wait
                await lock.acquire()
            
            # Update stats
            wait_time = time.time() - start_time
            self.lock_acquisitions += 1
            self.lock_wait_time_total += wait_time
            
            # If wait time was significant, log it
            if wait_time > 0.1:  # 100ms
                logger.warning(f"Lock acquisition for key {key} took {wait_time:.3f}s")
            
            # Yield control with lock acquired
            yield
        finally:
            # Release the lock
            lock.release()
    
    @contextmanager
    def acquire_sync(self, key: Any, timeout: Optional[float] = None):
        """
        Acquire a threading lock for a specific key
        
        Args:
            key: The key to lock on
            timeout: Optional timeout in seconds
            
        Yields:
            None when lock is acquired
            
        Raises:
            TimeoutError: If lock couldn't be acquired within timeout
        """
        lock = self.get_lock_sync(key)
        
        start_time = time.time()
        
        try:
            # Try to acquire the lock with timeout
            acquired = lock.acquire(timeout=timeout if timeout is not None else -1)
            
            if not acquired:
                self.lock_contentions += 1
                raise TimeoutError(f"Timeout acquiring lock for key {key}")
            
            # Update stats
            wait_time = time.time() - start_time
            self.lock_acquisitions += 1
            self.lock_wait_time_total += wait_time
            
            # If wait time was significant, log it
            if wait_time > 0.1:  # 100ms
                logger.warning(f"Lock acquisition for key {key} took {wait_time:.3f}s")
            
            # Yield control with lock acquired
            yield
        finally:
            # Release the lock
            try:
                lock.release()
            except RuntimeError:
                # Might not be holding the lock if an exception occurred
                pass
    
    async def cleanup_unused(self, max_age: float = 60.0):
        """
        Clean up unused locks to prevent memory leaks
        
        Args:
            max_age: Maximum age of unused locks in seconds
        """
        # Only makes sense for async locks that track last use
        if self.lock_type != "async":
            return
        
        # Implementation would require tracking last use of each lock
        # This is a placeholder for the idea
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get lock statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            "lock_type": self.lock_type,
            "active_keys": len(self.locks),
            "keys_created": self.keys_created,
            "lock_acquisitions": self.lock_acquisitions,
            "lock_contentions": self.lock_contentions,
            "avg_wait_time": (self.lock_wait_time_total / self.lock_acquisitions) if self.lock_acquisitions > 0 else 0,
            "contention_rate": (self.lock_contentions / self.lock_acquisitions) if self.lock_acquisitions > 0 else 0
        }

# Actor model implementation for better concurrency
class Actor:
    """
    Actor model implementation for isolated state and concurrency
    Each actor has its own state and processes messages one at a time
    """
    
    def __init__(self, name: str, processor: Callable = None):
        """
        Initialize an actor
        
        Args:
            name: Name of the actor
            processor: Optional message processor function
        """
        self.name = name
        self.processor = processor
        self.mailbox = asyncio.Queue()
        self.running = False
        self.task = None
        self.state = {}
        
        # Stats
        self.messages_processed = 0
        self.errors = 0
        self.processing_time = 0.0
        self.last_active = time.time()
        
        logger.info(f"Actor {self.name} created")
    
    async def start(self):
        """Start the actor's message processing loop"""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._process_messages())
        logger.info(f"Actor {self.name} started")
    
    async def stop(self):
        """Stop the actor's message processing loop"""
        if not self.running:
            return
        
        self.running = False
        
        if self.task:
            # Wait for current message processing to complete
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            
            self.task = None
        
        logger.info(f"Actor {self.name} stopped")
    
    async def tell(self, message: Any):
        """
        Send a message to the actor (non-blocking)
        
        Args:
            message: The message to send
        """
        await self.mailbox.put(message)
    
    async def ask(self, message: Any, timeout: float = None) -> Any:
        """
        Send a message and wait for a response
        
        Args:
            message: The message to send
            timeout: Optional timeout in seconds
            
        Returns:
            The response from the actor
            
        Raises:
            TimeoutError: If response wasn't received within timeout
        """
        # Create a future to receive the response
        response_future = asyncio.Future()
        
        # Create a message with the future
        ask_message = {
            "payload": message,
            "response_future": response_future
        }
        
        # Send the message
        await self.mailbox.put(ask_message)
        
        # Wait for response with optional timeout
        if timeout is not None:
            try:
                return await asyncio.wait_for(response_future, timeout)
            except asyncio.TimeoutError:
                # Mark the future as done to prevent later response
                if not response_future.done():
                    response_future.set_exception(TimeoutError(f"No response from actor {self.name} within {timeout}s"))
                raise TimeoutError(f"No response from actor {self.name} within {timeout}s")
        else:
            return await response_future
    
    async def _process_messages(self):
        """Process messages from the mailbox"""
        while self.running:
            try:
                # Get the next message (with timeout to allow stopping)
                try:
                    message = await asyncio.wait_for(self.mailbox.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                self.last_active = time.time()
                
                # Process the message
                start_time = time.time()
                
                try:
                    # Check if this is an ask message with response future
                    if isinstance(message, dict) and "response_future" in message:
                        response_future = message["response_future"]
                        actual_message = message["payload"]
                        
                        # Process the message
                        if self.processor:
                            result = await self._call_processor(actual_message)
                        else:
                            result = await self.receive(actual_message)
                        
                        # Set the result in the future
                        if not response_future.done():
                            response_future.set_result(result)
                    else:
                        # Regular message without response
                        if self.processor:
                            await self._call_processor(message)
                        else:
                            await self.receive(message)
                    
                    self.messages_processed += 1
                except Exception as e:
                    self.errors += 1
                    logger.error(f"Actor {self.name} error processing message: {str(e)}", exc_info=True)
                    
                    # Set exception in future if this was an ask
                    if isinstance(message, dict) and "response_future" in message:
                        response_future = message["response_future"]
                        if not response_future.done():
                            response_future.set_exception(e)
                
                # Update processing time
                self.processing_time += time.time() - start_time
                
                # Mark the message as done
                self.mailbox.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.errors += 1
                logger.error(f"Actor {self.name} unexpected error: {str(e)}", exc_info=True)
    
    async def _call_processor(self, message: Any) -> Any:
        """Call the message processor with actor state"""
        if asyncio.iscoroutinefunction(self.processor):
            return await self.processor(message, self.state)
        else:
            return self.processor(message, self.state)
    
    async def receive(self, message: Any) -> Any:
        """
        Default message handler method
        Override this in subclasses
        
        Args:
            message: The message to process
            
        Returns:
            Optional response
        """
        logger.warning(f"Actor {self.name} has no custom receive method and no processor")
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get actor statistics
        
        Returns:
            Dictionary of statistics
        """
        avg_processing_time = self.processing_time / max(1, self.messages_processed)
        
        return {
            "name": self.name,
            "running": self.running,
            "queue_size": self.mailbox.qsize(),
            "messages_processed": self.messages_processed,
            "errors": self.errors,
            "avg_processing_time": avg_processing_time,
            "last_active": self.last_active,
            "idle_time": time.time() - self.last_active
        }

# Actor system to manage multiple actors
class ActorSystem:
    """
    System to manage multiple actors
    Provides actor lifecycle management and supervision
    """
    
    def __init__(self, name: str = "actor-system"):
        """
        Initialize actor system
        
        Args:
            name: Name of the actor system
        """
        self.name = name
        self.actors = {}
        self.running = False
        self.supervisor_task = None
        
        logger.info(f"ActorSystem {self.name} created")
    
    async def start(self):
        """Start the actor system and all actors"""
        if self.running:
            return
        
        self.running = True
        
        # Start all actors
        for actor in self.actors.values():
            await actor.start()
        
        # Start supervisor
        self.supervisor_task = asyncio.create_task(self._supervisor())
        
        logger.info(f"ActorSystem {self.name} started with {len(self.actors)} actors")
    
    async def stop(self):
        """Stop the actor system and all actors"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop supervisor
        if self.supervisor_task:
            self.supervisor_task.cancel()
            try:
                await self.supervisor_task
            except asyncio.CancelledError:
                pass
            self.supervisor_task = None
        
        # Stop all actors
        for actor in self.actors.values():
            await actor.stop()
        
        logger.info(f"ActorSystem {self.name} stopped")
    
    def create_actor(self, name: str, processor: Callable = None) -> Actor:
        """
        Create and register a new actor
        
        Args:
            name: Name of the actor
            processor: Optional message processor function
            
        Returns:
            The created actor
        """
        if name in self.actors:
            raise ValueError(f"Actor {name} already exists")
        
        actor = Actor(name, processor)
        self.actors[name] = actor
        
        # Start actor if system is running
        if self.running:
            asyncio.create_task(actor.start())
        
        return actor
    
    def register_actor(self, actor: Actor):
        """
        Register an existing actor
        
        Args:
            actor: The actor to register
        """
        if actor.name in self.actors:
            raise ValueError(f"Actor {actor.name} already exists")
        
        self.actors[actor.name] = actor
        
        # Start actor if system is running
        if self.running:
            asyncio.create_task(actor.start())
    
    def get_actor(self, name: str) -> Optional[Actor]:
        """
        Get an actor by name
        
        Args:
            name: Name of the actor
            
        Returns:
            The actor or None if not found
        """
        return self.actors.get(name)
    
    async def _supervisor(self):
        """Supervisor task to monitor and manage actors"""
        while self.running:
            try:
                # Check all actors
                for name, actor in list(self.actors.items()):
                    # Check for errors or stuck actors
                    stats = await actor.get_stats()
                    
                    # Example supervision strategy: restart actors with too many errors
                    if stats["errors"] > 10:
                        logger.warning(f"Actor {name} has too many errors, restarting")
                        await actor.stop()
                        await actor.start()
                    
                    # Example: check for stuck actors
                    if stats["idle_time"] > 3600 and stats["queue_size"] > 0:
                        logger.warning(f"Actor {name} appears stuck, restarting")
                        await actor.stop()
                        await actor.start()
                
                # Sleep before next check
                await asyncio.sleep(10.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Actor system supervisor error: {str(e)}", exc_info=True)
                await asyncio.sleep(10.0)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get actor system statistics
        
        Returns:
            Dictionary of statistics
        """
        actor_stats = {}
        total_messages = 0
        total_errors = 0
        
        for name, actor in self.actors.items():
            stats = await actor.get_stats()
            actor_stats[name] = stats
            total_messages += stats["messages_processed"]
            total_errors += stats["errors"]
        
        return {
            "name": self.name,
            "running": self.running,
            "actor_count": len(self.actors),
            "total_messages": total_messages,
            "total_errors": total_errors,
            "error_rate": total_errors / max(1, total_messages),
            "actors": actor_stats
        }

# Optimistic concurrency control
class OptimisticLock:
    """
    Optimistic concurrency control using version numbers
    Allows concurrent reads and detects conflicts on writes
    """
    
    def __init__(self, initial_version: int = 0):
        """
        Initialize optimistic lock
        
        Args:
            initial_version: Initial version number
        """
        self.version = initial_version
        self.lock = asyncio.Lock()
        
        # Stats
        self.read_count = 0
        self.successful_writes = 0
        self.write_conflicts = 0
    
    async def read(self) -> int:
        """
        Read the current version
        
        Returns:
            Current version number
        """
        async with self.lock:
            self.read_count += 1
            return self.version
    
    async def write(self, expected_version: int) -> bool:
        """
        Attempt to write with expected version
        
        Args:
            expected_version: Expected current version
            
        Returns:
            True if write succeeded, False if version conflict
        """
        async with self.lock:
            if self.version == expected_version:
                # No conflict, update version
                self.version += 1
                self.successful_writes += 1
                return True
            else:
                # Version conflict
                self.write_conflicts += 1
                return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get lock statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            return {
                "current_version": self.version,
                "read_count": self.read_count,
                "successful_writes": self.successful_writes,
                "write_conflicts": self.write_conflicts,
                "conflict_rate": self.write_conflicts / max(1, self.write_conflicts + self.successful_writes)
            }

# Optimistic concurrency control for objects
class OptimisticConcurrencyControl:
    """
    Optimistic concurrency control for object modifications
    """
    
    def __init__(self):
        """Initialize optimistic concurrency control"""
        self.objects = {}
        self.lock = asyncio.Lock()
        
        # Stats
        self.read_count = 0
        self.successful_updates = 0
        self.update_conflicts = 0
    
    @dataclass
    class VersionedObject:
        """Object with version tracking"""
        data: Any
        version: int
        created_at: float
        last_updated_at: float
    
    async def get(self, object_id: str) -> Tuple[Any, int]:
        """
        Get an object with its version
        
        Args:
            object_id: ID of the object
            
        Returns:
            Tuple of (object_data, version)
            
        Raises:
            KeyError: If object doesn't exist
        """
        async with self.lock:
            if object_id not in self.objects:
                raise KeyError(f"Object {object_id} not found")
            
            versioned_obj = self.objects[object_id]
            self.read_count += 1
            
            # Return a copy of the data to prevent modification of the original
            data_copy = self._copy_data(versioned_obj.data)
            return data_copy, versioned_obj.version
    
    async def create(self, object_id: str, data: Any) -> int:
        """
        Create a new object
        
        Args:
            object_id: ID for the new object
            data: Object data
            
        Returns:
            Initial version number (0)
            
        Raises:
            ValueError: If object already exists
        """
        current_time = time.time()
        
        async with self.lock:
            if object_id in self.objects:
                raise ValueError(f"Object {object_id} already exists")
            
            # Store a copy of the data
            data_copy = self._copy_data(data)
            
            self.objects[object_id] = self.VersionedObject(
                data=data_copy,
                version=0,
                created_at=current_time,
                last_updated_at=current_time
            )
            
            self.successful_updates += 1
            return 0
    
    async def update(self, object_id: str, data: Any, expected_version: int) -> int:
        """
        Update an object with version check
        
        Args:
            object_id: ID of the object
            data: New object data
            expected_version: Expected current version
            
        Returns:
            New version number
            
        Raises:
            KeyError: If object doesn't exist
            ValueError: If version conflict
        """
        async with self.lock:
            if object_id not in self.objects:
                raise KeyError(f"Object {object_id} not found")
            
            versioned_obj = self.objects[object_id]
            
            if versioned_obj.version != expected_version:
                self.update_conflicts += 1
                raise ValueError(
                    f"Version conflict for object {object_id}: "
                    f"expected {expected_version}, actual {versioned_obj.version}"
                )
            
            # Store a copy of the data
            data_copy = self._copy_data(data)
            
            # Update object with new version
            self.objects[object_id] = self.VersionedObject(
                data=data_copy,
                version=versioned_obj.version + 1,
                created_at=versioned_obj.created_at,
                last_updated_at=time.time()
            )
            
            self.successful_updates += 1
            return versioned_obj.version + 1
    
    async def delete(self, object_id: str, expected_version: int) -> bool:
        """
        Delete an object with version check
        
        Args:
            object_id: ID of the object
            expected_version: Expected current version
            
        Returns:
            True if deleted successfully
            
        Raises:
            KeyError: If object doesn't exist
            ValueError: If version conflict
        """
        async with self.lock:
            if object_id not in self.objects:
                raise KeyError(f"Object {object_id} not found")
            
            versioned_obj = self.objects[object_id]
            
            if versioned_obj.version != expected_version:
                self.update_conflicts += 1
                raise ValueError(
                    f"Version conflict for object {object_id}: "
                    f"expected {expected_version}, actual {versioned_obj.version}"
                )
            
            # Delete object
            del self.objects[object_id]
            self.successful_updates += 1
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            return {
                "object_count": len(self.objects),
                "read_count": self.read_count,
                "successful_updates": self.successful_updates,
                "update_conflicts": self.update_conflicts,
                "conflict_rate": self.update_conflicts / max(1, self.update_conflicts + self.successful_updates)
            }
    
    def _copy_data(self, data: Any) -> Any:
        """
        Create a deep copy of data
        
        Args:
            data: Data to copy
            
        Returns:
            Copy of the data
        """
        # Simple implementation for common types
        if isinstance(data, dict):
            return {k: self._copy_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._copy_data(item) for item in data]
        elif isinstance(data, set):
            return {self._copy_data(item) for item in data}
        else:
            # Assume immutable type
            return data

# ReadWriteLock for reader/writer pattern
class ReadWriteLock:
    """
    Read-write lock that allows concurrent reads but exclusive writes
    """
    
    def __init__(self):
        """Initialize read-write lock"""
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._reader_count = 0
        
        # Stats
        self.read_acquisitions = 0
        self.write_acquisitions = 0
        self.read_wait_time = 0.0
        self.write_wait_time = 0.0
    
    @asynccontextmanager
    async def read_lock(self):
        """
        Acquire a read lock
        Multiple read locks can be held simultaneously
        
        Yields:
            None when lock is acquired
        """
        start_time = time.time()
        
        # Increment reader count atomically
        async with self._read_lock:
            self._reader_count += 1
            # Acquire write lock on first reader
            if self._reader_count == 1:
                await self._write_lock.acquire()
        
        try:
            # Update stats
            self.read_acquisitions += 1
            self.read_wait_time += time.time() - start_time
            
            # Yield with read lock held
            yield
        finally:
            # Decrement reader count atomically
            async with self._read_lock:
                self._reader_count -= 1
                # Release write lock on last reader
                if self._reader_count == 0:
                    self._write_lock.release()
    
    @asynccontextmanager
    async def write_lock(self):
        """
        Acquire a write lock
        Only one write lock can be held at a time, and no read locks can be held
        
        Yields:
            None when lock is acquired
        """
        start_time = time.time()
        
        # Acquire write lock
        await self._write_lock.acquire()
        
        try:
            # Update stats
            self.write_acquisitions += 1
            self.write_wait_time += time.time() - start_time
            
            # Yield with write lock held
            yield
        finally:
            # Release write lock
            self._write_lock.release()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get lock statistics
        
        Returns:
            Dictionary of statistics
        """
        avg_read_wait = self.read_wait_time / max(1, self.read_acquisitions)
        avg_write_wait = self.write_wait_time / max(1, self.write_acquisitions)
        
        return {
            "reader_count": self._reader_count,
            "read_acquisitions": self.read_acquisitions,
            "write_acquisitions": self.write_acquisitions,
            "avg_read_wait": avg_read_wait,
            "avg_write_wait": avg_write_wait,
            "write_to_read_ratio": self.write_acquisitions / max(1, self.read_acquisitions)
        }

# Shared lock manager for distributed locking
class SharedLockManager:
    """
    Manager for shared locks across processes or services
    Uses an external lock service for coordination
    """
    
    def __init__(self, lock_service_url: str = None, client_id: str = None):
        """
        Initialize shared lock manager
        
        Args:
            lock_service_url: URL of the lock service
            client_id: Unique client ID
        """
        self.lock_service_url = lock_service_url or "http://localhost:8080/locks"
        self.client_id = client_id or f"client-{uuid.uuid4()}"
        self.acquired_locks = set()
        self.lock = asyncio.Lock()
        
        # Stats
        self.acquisitions = 0
        self.releases = 0
        self.acquisition_failures = 0
        
        logger.info(f"SharedLockManager initialized with client_id={self.client_id}")
    
    async def acquire(self, resource_id: str, timeout_ms: float = 30000.0) -> bool:
        """
        Acquire a shared lock
        
        Args:
            resource_id: ID of resource to lock
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        
        # Calculate deadline
        deadline = start_time + (timeout_ms / 1000.0)
        
        while time.time() < deadline:
            try:
                # Try to acquire lock
                # This would be a call to an external lock service in a real implementation
                acquired = await self._call_lock_service(
                    "acquire",
                    resource_id=resource_id,
                    client_id=self.client_id
                )
                
                if acquired:
                    # Track acquired lock
                    async with self.lock:
                        self.acquired_locks.add(resource_id)
                        self.acquisitions += 1
                    
                    logger.info(f"Lock acquired for {resource_id} by {self.client_id}")
                    return True
                
                # Wait before retry
                jitter = random.uniform(0.1, 0.3)  # 100-300ms jitter
                await asyncio.sleep(jitter)
                
            except Exception as e:
                logger.error(f"Error acquiring lock for {resource_id}: {str(e)}")
                # Brief delay before retry
                await asyncio.sleep(0.5)
        
        # Timeout
        self.acquisition_failures += 1
        logger.warning(f"Timeout acquiring lock for {resource_id}")
        return False
    
    async def release(self, resource_id: str) -> bool:
        """
        Release a shared lock
        
        Args:
            resource_id: ID of resource to unlock
            
        Returns:
            True if released, False if not held
        """
        # Check if we hold this lock
        async with self.lock:
            if resource_id not in self.acquired_locks:
                logger.warning(f"Attempted to release lock for {resource_id} that wasn't acquired")
                return False
        
        try:
            # Release lock
            # This would be a call to an external lock service in a real implementation
            released = await self._call_lock_service(
                "release",
                resource_id=resource_id,
                client_id=self.client_id
            )
            
            if released:
                # Remove from tracking
                async with self.lock:
                    self.acquired_locks.remove(resource_id)
                    self.releases += 1
                
                logger.info(f"Lock released for {resource_id} by {self.client_id}")
                return True
            
            logger.warning(f"Failed to release lock for {resource_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error releasing lock for {resource_id}: {str(e)}")
            return False
    
    @asynccontextmanager
    async def with_lock(self, resource_id: str, timeout_ms: float = 30000.0):
        """
        Context manager for acquiring and releasing a lock
        
        Args:
            resource_id: ID of resource to lock
            timeout_ms: Timeout in milliseconds
            
        Yields:
            True if lock acquired
            
        Raises:
            TimeoutError: If lock couldn't be acquired within timeout
        """
        acquired = await self.acquire(resource_id, timeout_ms)
        
        if not acquired:
            raise TimeoutError(f"Timeout acquiring lock for {resource_id}")
        
        try:
            # Yield with lock held
            yield True
        finally:
            # Release lock
            await self.release(resource_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get lock manager statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            return {
                "client_id": self.client_id,
                "acquired_locks": len(self.acquired_locks),
                "acquisitions": self.acquisitions,
                "releases": self.releases,
                "acquisition_failures": self.acquisition_failures,
                "failure_rate": self.acquisition_failures / max(1, self.acquisitions + self.acquisition_failures)
            }
    
    async def _call_lock_service(self, action: str, resource_id: str, client_id: str) -> bool:
        """
        Call the external lock service
        
        Args:
            action: "acquire" or "release"
            resource_id: ID of resource to lock/unlock
            client_id: Client ID
            
        Returns:
            True if successful, False otherwise
        """
        # This is a mock implementation
        # In a real system, this would make HTTP/gRPC calls to an external lock service
        
        # Simulate successful acquisition/release with 90% probability
        return random.random() < 0.9

# Event bus for decoupling components
class EventBus:
    """
    Event bus for publishing and subscribing to events
    Allows decoupling of components through event-based communication
    """
    
    def __init__(self, name: str = "event-bus"):
        """
        Initialize event bus
        
        Args:
            name: Name of the event bus
        """
        self.name = name
        self.subscribers = {}  # event_type -> list of callbacks
        self.lock = asyncio.Lock()
        
        # Stats
        self.events_published = 0
        self.events_delivered = 0
        self.delivery_failures = 0
        
        logger.info(f"EventBus {self.name} created")
    
    async def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to an event type
        
        Args:
            event_type: Type of events to subscribe to
            callback: Callback function for events
        """
        async with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            self.subscribers[event_type].append(callback)
            
        logger.debug(f"Subscribed to event type: {event_type}")
    
    async def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type
        
        Args:
            event_type: Type of events to unsubscribe from
            callback: Callback function to remove
            
        Returns:
            True if unsubscribed, False if not found
        """
        async with self.lock:
            if event_type not in self.subscribers:
                return False
            
            callbacks = self.subscribers[event_type]
            if callback not in callbacks:
                return False
            
            callbacks.remove(callback)
            
            # Clean up empty lists
            if not callbacks:
                del self.subscribers[event_type]
            
        logger.debug(f"Unsubscribed from event type: {event_type}")
        return True
    
    async def publish(self, event_type: str, event_data: Any, wait_for_delivery: bool = False):
        """
        Publish an event
        
        Args:
            event_type: Type of event
            event_data: Event data
            wait_for_delivery: If True, wait for all subscribers to process the event
        """
        # Get subscribers
        callbacks = []
        async with self.lock:
            self.events_published += 1
            if event_type in self.subscribers:
                callbacks = list(self.subscribers[event_type])
        
        if not callbacks:
            logger.debug(f"No subscribers for event type: {event_type}")
            return
        
        # Deliver to subscribers
        logger.debug(f"Delivering event {event_type} to {len(callbacks)} subscribers")
        
        tasks = []
        for callback in callbacks:
            if wait_for_delivery:
                # Create tasks for synchronous delivery
                task = self._deliver_event(callback, event_type, event_data)
                tasks.append(task)
            else:
                # Fire and forget
                asyncio.create_task(self._deliver_event(callback, event_type, event_data))
        
        if wait_for_delivery and tasks:
            # Wait for all deliveries to complete
            await asyncio.gather(*tasks)
    
    async def _deliver_event(self, callback: Callable, event_type: str, event_data: Any):
        """
        Deliver an event to a subscriber
        
        Args:
            callback: Subscriber callback
            event_type: Type of event
            event_data: Event data
        """
        try:
            # Call the callback
            if asyncio.iscoroutinefunction(callback):
                await callback(event_type, event_data)
            else:
                callback(event_type, event_data)
            
            async with self.lock:
                self.events_delivered += 1
                
        except Exception as e:
            logger.error(f"Error delivering event {event_type}: {str(e)}", exc_info=True)
            
            async with self.lock:
                self.delivery_failures += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics
        
        Returns:
            Dictionary of statistics
        """
        async with self.lock:
            event_types = list(self.subscribers.keys())
            subscriber_counts = {event_type: len(callbacks) for event_type, callbacks in self.subscribers.items()}
            
            return {
                "name": self.name,
                "event_types": event_types,
                "subscriber_counts": subscriber_counts,
                "total_subscribers": sum(len(callbacks) for callbacks in self.subscribers.values()),
                "events_published": self.events_published,
                "events_delivered": self.events_delivered,
                "delivery_failures": self.delivery_failures,
                "delivery_rate": self.events_delivered / max(1, self.events_published)
            }