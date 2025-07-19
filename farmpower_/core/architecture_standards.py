"""
Architecture Standards for LogLineOS
Provides core patterns, utilities and guidelines for architectural consistency
Created: 2025-07-19 06:27:20 UTC
User: danvoulez
"""
import asyncio
import functools
import inspect
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/architecture.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ArchitectureStandards")

# Result type for consistent error handling
T = TypeVar('T')
E = TypeVar('E')

class Result(Generic[T, E]):
    """
    Result type for consistent error handling across LogLineOS
    Inspired by Rust's Result type
    """
    
    def __init__(self, value: Optional[T] = None, error: Optional[E] = None):
        self._value = value
        self._error = error
        self._is_ok = error is None
    
    @staticmethod
    def ok(value: T) -> 'Result[T, E]':
        """Create a success result"""
        return Result(value=value)
    
    @staticmethod
    def err(error: E) -> 'Result[T, E]':
        """Create an error result"""
        return Result(error=error)
    
    def is_ok(self) -> bool:
        """Check if result is successful"""
        return self._is_ok
    
    def is_err(self) -> bool:
        """Check if result is an error"""
        return not self._is_ok
    
    def unwrap(self) -> T:
        """Get the value or raise an exception if error"""
        if self._is_ok:
            return self._value
        raise ValueError(f"Tried to unwrap an error result: {self._error}")
    
    def unwrap_or(self, default: T) -> T:
        """Get the value or a default if error"""
        if self._is_ok:
            return self._value
        return default
    
    def unwrap_err(self) -> E:
        """Get the error or raise an exception if ok"""
        if not self._is_ok:
            return self._error
        raise ValueError("Tried to unwrap_err on a success result")
    
    def map(self, func: Callable[[T], Any]) -> 'Result':
        """Map the success value"""
        if self._is_ok:
            return Result.ok(func(self._value))
        return Result.err(self._error)
    
    def map_err(self, func: Callable[[E], Any]) -> 'Result':
        """Map the error value"""
        if self._is_ok:
            return Result.ok(self._value)
        return Result.err(func(self._error))
    
    def __str__(self) -> str:
        if self._is_ok:
            return f"Ok({self._value})"
        return f"Err({self._error})"

# Standard error types
class ErrorCategory(Enum):
    """Standard error categories for consistent error classification"""
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
    """Standard error information structure"""
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

# Async utilities
class AsyncUtilities:
    """Utilities for consistent async programming"""
    
    @staticmethod
    async def with_timeout(coro, timeout_seconds: float, timeout_message: str = None):
        """Run a coroutine with a timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            msg = timeout_message or f"Operation timed out after {timeout_seconds} seconds"
            return Result.err(ErrorInfo(
                message=msg,
                category=ErrorCategory.TIMEOUT,
                code="OPERATION_TIMEOUT"
            ))
    
    @staticmethod
    def run_in_executor(func, *args, **kwargs):
        """Run a blocking function in the default executor"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            None, 
            lambda: func(*args, **kwargs)
        )
    
    @staticmethod
    async def gather_with_concurrency(n: int, *coros):
        """Run coroutines with limited concurrency"""
        semaphore = asyncio.Semaphore(n)
        
        async def sem_coro(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(*(sem_coro(c) for c in coros))
    
    @staticmethod
    def periodic(interval_seconds: float):
        """Decorator to make a function run periodically"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapped(*args, **kwargs):
                while True:
                    await func(*args, **kwargs)
                    await asyncio.sleep(interval_seconds)
            return wrapped
        return decorator

# Retry decorator with exponential backoff
def retry(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0,
          retry_on: List[Exception] = None):
    """
    Retry decorator with exponential backoff
    """
    retry_on = retry_on or [Exception]
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                retry_count = 0
                delay = base_delay
                
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except tuple(retry_on) as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.warning(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                            raise
                        
                        # Calculate delay with exponential backoff and jitter
                        jitter = 0.1 * delay * (2 * (0.5 - random.random()))
                        current_delay = min(max_delay, delay + jitter)
                        
                        logger.info(f"Retry {retry_count}/{max_retries} for {func.__name__} after {current_delay:.2f}s")
                        await asyncio.sleep(current_delay)
                        delay *= 2  # Exponential backoff
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retry_count = 0
                delay = base_delay
                
                while True:
                    try:
                        return func(*args, **kwargs)
                    except tuple(retry_on) as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.warning(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                            raise
                        
                        # Calculate delay with exponential backoff and jitter
                        jitter = 0.1 * delay * (2 * (0.5 - random.random()))
                        current_delay = min(max_delay, delay + jitter)
                        
                        logger.info(f"Retry {retry_count}/{max_retries} for {func.__name__} after {current_delay:.2f}s")
                        time.sleep(current_delay)
                        delay *= 2  # Exponential backoff
            
            return wrapper
    
    return decorator

# Circuit breaker for resilient external calls
class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external service calls
    """
    
    # Circuit states
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, no calls allowed
    HALF_OPEN = "half_open"  # Testing recovery
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                recovery_timeout: float = 30.0, half_open_max_calls: int = 1):
        """
        Initialize circuit breaker
        
        Args:
            name: Name for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # State tracking
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def __call__(self, func, *args, **kwargs):
        """Use as decorator or context manager"""
        async with self._lock:
            # Check if circuit is open
            if self.state == self.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.info(f"Circuit {self.name} transitioning from OPEN to HALF_OPEN")
                    self.state = self.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    logger.warning(f"Circuit {self.name} is OPEN, call rejected")
                    return Result.err(ErrorInfo(
                        message=f"Circuit {self.name} is open",
                        category=ErrorCategory.EXTERNAL,
                        code="CIRCUIT_OPEN"
                    ))
            
            # Check if we're in half-open state and already at max calls
            if self.state == self.HALF_OPEN and self.half_open_calls >= self.half_open_max_calls:
                logger.warning(f"Circuit {self.name} is HALF_OPEN and at max calls, rejecting")
                return Result.err(ErrorInfo(
                    message=f"Circuit {self.name} is half-open and at max calls",
                    category=ErrorCategory.EXTERNAL,
                    code="CIRCUIT_HALF_OPEN"
                ))
            
            # Increment half-open calls counter if applicable
            if self.state == self.HALF_OPEN:
                self.half_open_calls += 1
        
        # Execute the function
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Handle success
            async with self._lock:
                if self.state == self.HALF_OPEN:
                    logger.info(f"Circuit {self.name} recovered, transitioning to CLOSED")
                    self.state = self.CLOSED
                    
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            # Handle failure
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.state == self.CLOSED and self.failure_count >= self.failure_threshold:
                    logger.warning(f"Circuit {self.name} transitioning from CLOSED to OPEN after {self.failure_count} failures")
                    self.state = self.OPEN
                elif self.state == self.HALF_OPEN:
                    logger.warning(f"Circuit {self.name} transitioning from HALF_OPEN to OPEN after failure during recovery")
                    self.state = self.OPEN
            
            # Re-raise the exception
            raise

# Service discovery and dependency injection
class ServiceRegistry:
    """
    Service registry for dependency injection and service discovery
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            cls._instance._services = {}
        return cls._instance
    
    def register(self, service_name: str, service_instance: Any):
        """Register a service"""
        self._services[service_name] = service_instance
    
    def get(self, service_name: str) -> Any:
        """Get a service by name"""
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not registered")
        return self._services[service_name]
    
    def has(self, service_name: str) -> bool:
        """Check if a service is registered"""
        return service_name in self._services
    
    def create_and_register(self, service_name: str, service_class: type, *args, **kwargs) -> Any:
        """Create a service instance and register it"""
        service = service_class(*args, **kwargs)
        self.register(service_name, service)
        return service

# Example usage:
# registry = ServiceRegistry()
# registry.register("llm_service", LLMService())
# llm = registry.get("llm_service")

# Instrumentation decorator for logging, timing, and metrics
def instrumented(name: str = None, log_level: str = "info", 
                include_args: bool = False, include_result: bool = False):
    """
    Decorator for instrumenting functions with logging, timing, and metrics
    """
    def decorator(func):
        # Get function name if not provided
        nonlocal name
        if name is None:
            name = func.__name__
        
        log_method = getattr(logger, log_level)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Build argument string if requested
            args_str = ""
            if include_args:
                # Extract only serializable arguments
                safe_args = str(args)[:100] if args else ""
                safe_kwargs = str(kwargs)[:100] if kwargs else ""
                args_str = f"(args: {safe_args}, kwargs: {safe_kwargs})"
            
            log_method(f"Starting {name} {args_str}")
            
            try:
                result = await func(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Log completion with result if requested
                result_str = ""
                if include_result:
                    # Safely serialize result
                    if hasattr(result, "__str__"):
                        result_str = f": {str(result)[:100]}"
                
                log_method(f"Completed {name} in {duration_ms:.2f}ms{result_str}")
                
                # Update metrics (assuming a metrics collector exists)
                try:
                    metrics.record_timing(f"{name}_duration_ms", duration_ms)
                    metrics.increment(f"{name}_success_count")
                except:
                    pass  # Metrics collection is optional
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Failed {name} after {duration_ms:.2f}ms: {str(e)}")
                
                # Update error metrics
                try:
                    metrics.record_timing(f"{name}_duration_ms", duration_ms)
                    metrics.increment(f"{name}_error_count")
                except:
                    pass  # Metrics collection is optional
                
                # Re-raise
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Build argument string if requested
            args_str = ""
            if include_args:
                # Extract only serializable arguments
                safe_args = str(args)[:100] if args else ""
                safe_kwargs = str(kwargs)[:100] if kwargs else ""
                args_str = f"(args: {safe_args}, kwargs: {safe_kwargs})"
            
            log_method(f"Starting {name} {args_str}")
            
            try:
                result = func(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Log completion with result if requested
                result_str = ""
                if include_result:
                    # Safely serialize result
                    if hasattr(result, "__str__"):
                        result_str = f": {str(result)[:100]}"
                
                log_method(f"Completed {name} in {duration_ms:.2f}ms{result_str}")
                
                # Update metrics (assuming a metrics collector exists)
                try:
                    metrics.record_timing(f"{name}_duration_ms", duration_ms)
                    metrics.increment(f"{name}_success_count")
                except:
                    pass  # Metrics collection is optional
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Failed {name} after {duration_ms:.2f}ms: {str(e)}")
                
                # Update error metrics
                try:
                    metrics.record_timing(f"{name}_duration_ms", duration_ms)
                    metrics.increment(f"{name}_error_count")
                except:
                    pass  # Metrics collection is optional
                
                # Re-raise
                raise
        
        # Use the appropriate wrapper based on whether func is async or not
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator