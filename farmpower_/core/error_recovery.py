"""
Error Recovery System for LogLineOS
Provides robust error handling and recovery mechanisms
Created: 2025-07-19 06:27:20 UTC
User: danvoulez
"""
import os
import time
import logging
import asyncio
import traceback
import random
import uuid
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

# Import from architecture standards if available
try:
    from core.architecture_standards import Result, ErrorInfo, ErrorCategory
except ImportError:
    # Define minimal versions if not available
    class ErrorCategory(Enum):
        VALIDATION = "validation"
        PERMISSION = "permission"
        NOT_FOUND = "not_found"
        CONFLICT = "conflict"
        EXTERNAL = "external"
        TIMEOUT = "timeout"
        RESOURCE = "resource"
        INTERNAL = "internal"
        UNKNOWN = "unknown"

    @dataclass
    class ErrorInfo:
        message: str
        category: ErrorCategory
        code: str
        details: Dict[str, Any] = None
        source: str = None
        timestamp: float = None
        trace_id: str = None
        
        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = time.time()
            if self.details is None:
                self.details = {}
    
    T = TypeVar('T')
    E = TypeVar('E')
    
    class Result(Generic[T, E]):
        def __init__(self, value: Optional[T] = None, error: Optional[E] = None):
            self._value = value
            self._error = error
            self._is_ok = error is None
        
        @staticmethod
        def ok(value: T) -> 'Result[T, E]':
            return Result(value=value)
        
        @staticmethod
        def err(error: E) -> 'Result[T, E]':
            return Result(error=error)
        
        def is_ok(self) -> bool:
            return self._is_ok
        
        def is_err(self) -> bool:
            return not self._is_ok
        
        def unwrap(self) -> T:
            if self._is_ok:
                return self._value
            raise ValueError(f"Tried to unwrap an error result: {self._error}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/error_recovery.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ErrorRecovery")

# Retry strategy enum
class RetryStrategy(Enum):
    """Different retry strategies"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    RANDOM_EXPONENTIAL = "random_exponential"

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    initial_delay_ms: float = 100.0
    max_delay_ms: float = 30000.0  # 30 seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter_factor: float = 0.1  # 10% jitter
    retry_codes: List[str] = None  # Error codes to retry on
    retry_categories: List[ErrorCategory] = None  # Error categories to retry on
    
    def __post_init__(self):
        if self.retry_codes is None:
            self.retry_codes = []
        if self.retry_categories is None:
            self.retry_categories = [
                ErrorCategory.TIMEOUT, 
                ErrorCategory.EXTERNAL,
                ErrorCategory.RESOURCE
            ]

# Advanced retry decorator with different strategies
def with_retry(config: RetryConfig = None):
    """
    Advanced retry decorator with multiple strategies
    
    Args:
        config: RetryConfig with retry settings
    
    Returns:
        Decorated function
    """
    # Use default config if none provided
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            attempt = 0
            last_error = None
            delay_ms = config.initial_delay_ms
            
            while attempt <= config.max_retries:
                try:
                    result = await func(*args, **kwargs)
                    
                    # If using Result type, check for retriable errors
                    if isinstance(result, Result) and result.is_err():
                        error = result.unwrap_err()
                        if isinstance(error, ErrorInfo):
                            # Check if we should retry this error
                            should_retry = (
                                error.code in config.retry_codes or
                                error.category in config.retry_categories
                            )
                            
                            if not should_retry:
                                return result  # Don't retry this type of error
                            
                            last_error = error
                        else:
                            # Unknown error format, assume success
                            return result
                    else:
                        # Success or non-Result type
                        return result
                    
                except Exception as e:
                    last_error = e
                
                # Increment attempt counter
                attempt += 1
                
                # If max retries reached, raise the last error or return error Result
                if attempt > config.max_retries:
                    if isinstance(last_error, Exception):
                        raise last_error
                    else:
                        # Assume ErrorInfo or similar
                        return Result.err(last_error)
                
                # Calculate delay for next retry
                delay_ms = _calculate_delay(delay_ms, attempt, config)
                
                # Log retry
                if isinstance(last_error, Exception):
                    error_msg = str(last_error)
                else:
                    error_msg = getattr(last_error, "message", str(last_error))
                
                logger.info(f"Retry {attempt}/{config.max_retries} for {func.__name__} "
                           f"after {delay_ms:.2f}ms delay. Error: {error_msg}")
                
                # Wait before retrying
                await asyncio.sleep(delay_ms / 1000.0)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            attempt = 0
            last_error = None
            delay_ms = config.initial_delay_ms
            
            while attempt <= config.max_retries:
                try:
                    result = func(*args, **kwargs)
                    
                    # If using Result type, check for retriable errors
                    if isinstance(result, Result) and result.is_err():
                        error = result.unwrap_err()
                        if isinstance(error, ErrorInfo):
                            # Check if we should retry this error
                            should_retry = (
                                error.code in config.retry_codes or
                                error.category in config.retry_categories
                            )
                            
                            if not should_retry:
                                return result  # Don't retry this type of error
                            
                            last_error = error
                        else:
                            # Unknown error format, assume success
                            return result
                    else:
                        # Success or non-Result type
                        return result
                    
                except Exception as e:
                    last_error = e
                
                # Increment attempt counter
                attempt += 1
                
                # If max retries reached, raise the last error or return error Result
                if attempt > config.max_retries:
                    if isinstance(last_error, Exception):
                        raise last_error
                    else:
                        # Assume ErrorInfo or similar
                        return Result.err(last_error)
                
                # Calculate delay for next retry
                delay_ms = _calculate_delay(delay_ms, attempt, config)
                
                # Log retry
                if isinstance(last_error, Exception):
                    error_msg = str(last_error)
                else:
                    error_msg = getattr(last_error, "message", str(last_error))
                
                logger.info(f"Retry {attempt}/{config.max_retries} for {func.__name__} "
                           f"after {delay_ms:.2f}ms delay. Error: {error_msg}")
                
                # Wait before retrying
                time.sleep(delay_ms / 1000.0)
        
        # Use the appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

def _calculate_delay(current_delay: float, attempt: int, config: RetryConfig) -> float:
    """
    Calculate the next retry delay based on strategy
    
    Args:
        current_delay: Current delay in milliseconds
        attempt: Current attempt number (1-based)
        config: RetryConfig with retry settings
        
    Returns:
        Next delay in milliseconds
    """
    if config.strategy == RetryStrategy.FIXED_DELAY:
        delay = config.initial_delay_ms
    
    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.initial_delay_ms * attempt
    
    elif config.strategy == RetryStrategy.RANDOM_EXPONENTIAL:
        # Exponential base with randomization
        factor = 2 ** (attempt - 1)
        base_delay = config.initial_delay_ms * factor
        max_rand = int(base_delay * config.jitter_factor * 2)
        rand_offset = random.randint(0, max_rand) - (max_rand // 2)
        delay = base_delay + rand_offset
    
    else:  # Default: EXPONENTIAL_BACKOFF
        delay = config.initial_delay_ms * (2 ** (attempt - 1))
    
    # Add jitter to avoid thundering herd
    if config.jitter_factor > 0:
        jitter = config.jitter_factor * delay
        delay += random.uniform(-jitter, jitter)
    
    # Cap at max delay
    return min(delay, config.max_delay_ms)

# Advanced circuit breaker
class CircuitBreaker:
    """
    Circuit breaker pattern with advanced features
    """
    
    # Circuit states
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, no calls allowed
    HALF_OPEN = "half_open"  # Testing recovery
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                recovery_timeout_ms: float = 30000.0, half_open_max_calls: int = 1,
                error_codes: List[str] = None, error_categories: List[ErrorCategory] = None):
        """
        Initialize circuit breaker
        
        Args:
            name: Name for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout_ms: Milliseconds to wait before testing recovery
            half_open_max_calls: Max calls allowed in half-open state
            error_codes: Specific error codes to consider as failures
            error_categories: Error categories to consider as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.half_open_max_calls = half_open_max_calls
        self.error_codes = error_codes or []
        self.error_categories = error_categories or [
            ErrorCategory.EXTERNAL,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RESOURCE
        ]
        
        # State tracking
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.state_transitions = []
        
        logger.info(f"Circuit breaker {self.name} initialized in CLOSED state")
    
    async def execute(self, func, *args, **kwargs):
        """
        Execute a function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Function result or error
        """
        async with self._lock:
            self.total_calls += 1
            
            # Check if circuit is open
            if self.state == self.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() * 1000 - self.last_failure_time > self.recovery_timeout_ms:
                    logger.info(f"Circuit {self.name} transitioning from OPEN to HALF_OPEN")
                    self._transition_state(self.HALF_OPEN)
                    self.half_open_calls = 0
                else:
                    # Circuit is open and timeout not elapsed - reject the call
                    self.rejected_calls += 1
                    logger.warning(f"Circuit {self.name} is OPEN, call rejected")
                    return Result.err(ErrorInfo(
                        message=f"Circuit breaker {self.name} is open",
                        category=ErrorCategory.EXTERNAL,
                        code="CIRCUIT_OPEN"
                    ))
            
            # Check if we're in half-open state and already at max calls
            if self.state == self.HALF_OPEN and self.half_open_calls >= self.half_open_max_calls:
                self.rejected_calls += 1
                logger.warning(f"Circuit {self.name} is HALF_OPEN and at max calls, rejecting")
                return Result.err(ErrorInfo(
                    message=f"Circuit breaker {self.name} is half-open and at max calls",
                    category=ErrorCategory.EXTERNAL,
                    code="CIRCUIT_HALF_OPEN"
                ))
            
            # Increment half-open calls counter if applicable
            if self.state == self.HALF_OPEN:
                self.half_open_calls += 1
        
        # Execute the function outside of the lock to allow concurrency
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Handle Result type
            if isinstance(result, Result):
                if result.is_err():
                    error = result.unwrap_err()
                    # Check if this error should trigger the circuit breaker
                    should_count = False
                    
                    if isinstance(error, ErrorInfo):
                        should_count = (
                            error.code in self.error_codes or
                            error.category in self.error_categories
                        )
                    
                    if should_count:
                        await self._record_failure(error)
                    return result
                else:
                    await self._record_success()
                    return result
            else:
                # Success case
                await self._record_success()
                return result
                
        except Exception as e:
            # Record failure
            await self._record_failure(e)
            
            # Wrap in Result if desired
            # return Result.err(...)
            
            # Or re-raise
            raise
    
    async def _record_success(self):
        """Record a successful call"""
        async with self._lock:
            self.successful_calls += 1
            
            if self.state == self.HALF_OPEN:
                logger.info(f"Circuit {self.name} recovered, transitioning to CLOSED")
                self._transition_state(self.CLOSED)
                
            # Reset failure count on success
            self.failure_count = 0
    
    async def _record_failure(self, error):
        """Record a failed call"""
        async with self._lock:
            self.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time() * 1000
            
            if self.state == self.CLOSED and self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit {self.name} reached failure threshold "
                             f"({self.failure_count}/{self.failure_threshold}), transitioning to OPEN")
                self._transition_state(self.OPEN)
                
            elif self.state == self.HALF_OPEN:
                logger.warning(f"Circuit {self.name} failed during recovery, transitioning back to OPEN")
                self._transition_state(self.OPEN)
    
    def _transition_state(self, new_state):
        """Transition circuit state with logging"""
        old_state = self.state
        self.state = new_state
        timestamp = time.time()
        
        # Record state transition
        self.state_transitions.append({
            "from": old_state,
            "to": new_state,
            "timestamp": timestamp,
            "failure_count": self.failure_count
        })
        
        logger.info(f"Circuit breaker {self.name} state changed: {old_state} -> {new_state}")
    
    def get_stats(self):
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "success_rate": (self.successful_calls / self.total_calls) if self.total_calls > 0 else 0,
            "state_transitions": self.state_transitions[-10:],  # Last 10 transitions
            "current_timestamp": time.time() * 1000
        }

# Fault tolerance with bulkhead pattern
class Bulkhead:
    """
    Bulkhead pattern for fault isolation
    Limits concurrent executions to prevent cascading failures
    """
    
    def __init__(self, name: str, max_concurrent: int = 10, max_queue_size: int = 100):
        """
        Initialize bulkhead
        
        Args:
            name: Name for this bulkhead
            max_concurrent: Maximum concurrent executions
            max_queue_size: Maximum queue size for pending executions
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        # State tracking
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.current_executions = 0
        self.queue_size = 0
        self._lock = asyncio.Lock()
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.rejected_executions = 0
        
        logger.info(f"Bulkhead {self.name} initialized with max_concurrent={max_concurrent}, "
                  f"max_queue_size={max_queue_size}")
    
    async def execute(self, func, *args, **kwargs):
        """
        Execute a function with bulkhead protection
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Function result or error
        """
        # Check if queue is full
        async with self._lock:
            if self.queue_size >= self.max_queue_size:
                self.rejected_executions += 1
                logger.warning(f"Bulkhead {self.name} queue is full, execution rejected")
                return Result.err(ErrorInfo(
                    message=f"Bulkhead {self.name} queue is full",
                    category=ErrorCategory.RESOURCE,
                    code="BULKHEAD_QUEUE_FULL"
                ))
            
            self.queue_size += 1
        
        try:
            # Try to acquire semaphore
            async with self.semaphore:
                # Execution started
                async with self._lock:
                    self.queue_size -= 1
                    self.current_executions += 1
                    self.total_executions += 1
                
                # Execute the function
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Handle Result type
                    if isinstance(result, Result):
                        if result.is_err():
                            async with self._lock:
                                self.failed_executions += 1
                        else:
                            async with self._lock:
                                self.successful_executions += 1
                    else:
                        # Assume success
                        async with self._lock:
                            self.successful_executions += 1
                