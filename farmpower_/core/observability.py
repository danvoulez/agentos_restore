"""
Observability Framework for LogLineOS
Provides advanced monitoring, logging, and metrics
Created: 2025-07-19 06:43:11 UTC
User: danvoulez
"""
import os
import time
import logging
import asyncio
import threading
import uuid
import json
import inspect
import traceback
import socket
import platform
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from functools import wraps
from contextlib import contextmanager
import weakref
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/observability.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Observability")

# Try to import optional libraries
try:
    import prometheus_client
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning("prometheus_client not available, Prometheus metrics will be disabled")

try:
    import opentelemetry.trace as otel_trace
    import opentelemetry.metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    logger.warning("opentelemetry not available, distributed tracing will be disabled")

class LogLevel(Enum):
    """Log levels with consistent naming"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class LogContext:
    """Context data to include in logs"""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user: Optional[str] = None
    component: Optional[str] = None
    request_id: Optional[str] = None
    additional: Dict[str, Any] = field(default_factory=dict)

class StructuredLogger:
    """
    Logger that outputs structured logs in JSON format
    """
    
    def __init__(self, name: str, default_context: Optional[LogContext] = None):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            default_context: Default context for all logs
        """
        self.name = name
        self.default_context = default_context or LogContext()
        self.logger = logging.getLogger(name)
        
        # Setup JSON formatter for the logger if not already done
        self._setup_json_formatter()
        
        logger.info(f"StructuredLogger {name} initialized")
    
    def _setup_json_formatter(self):
        """Set up JSON formatter for logs"""
        # Check if there's already a file handler with a JSON formatter
        has_json_formatter = False
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) and hasattr(handler, 'formatter') and getattr(handler.formatter, '_is_json_formatter', False):
                has_json_formatter = True
                break
        
        if not has_json_formatter:
            # Create a file handler with JSON formatter
            json_handler = logging.FileHandler(f"logs/{self.name}_structured.log")
            json_formatter = self._create_json_formatter()
            json_formatter._is_json_formatter = True
            json_handler.setFormatter(json_formatter)
            self.logger.addHandler(json_handler)
    
    def _create_json_formatter(self):
        """Create a JSON formatter for logs"""
        return logging.Formatter(fmt="%(message)s")
    
    def log(self, level: LogLevel, message: str, context: Optional[LogContext] = None, **kwargs):
        """
        Log a message with structured context
        
        Args:
            level: Log level
            message: Log message
            context: Optional additional context
            **kwargs: Additional fields to include in the log
        """
        # Merge contexts
        merged_context = self._merge_contexts(context)
        
        # Create log data
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "logger": self.name,
            "message": message,
            **kwargs
        }
        
        # Add context fields
        if merged_context.trace_id:
            log_data["trace_id"] = merged_context.trace_id
        
        if merged_context.span_id:
            log_data["span_id"] = merged_context.span_id
        
        if merged_context.user:
            log_data["user"] = merged_context.user
        
        if merged_context.component:
            log_data["component"] = merged_context.component
        
        if merged_context.request_id:
            log_data["request_id"] = merged_context.request_id
        
        # Add additional context
        for key, value in merged_context.additional.items():
            log_data[key] = value
        
        # Convert log data to JSON
        log_message = json.dumps(log_data)
        
        # Log at the appropriate level
        log_method = getattr(self.logger, level.value)
        log_method(log_message)
    
    def _merge_contexts(self, context: Optional[LogContext]) -> LogContext:
        """Merge provided context with default context"""
        if context is None:
            return self.default_context
        
        merged = LogContext(
            trace_id=context.trace_id or self.default_context.trace_id,
            span_id=context.span_id or self.default_context.span_id,
            user=context.user or self.default_context.user,
            component=context.component or self.default_context.component,
            request_id=context.request_id or self.default_context.request_id,
            additional={**self.default_context.additional, **context.additional}
        )
        
        return merged
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log at DEBUG level"""
        self.log(LogLevel.DEBUG, message, context, **kwargs)
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log at INFO level"""
        self.log(LogLevel.INFO, message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log at WARNING level"""
        self.log(LogLevel.WARNING, message, context, **kwargs)
    
    def error(self, message: str, context: Optional[LogContext] = None, exception: Optional[Exception] = None, **kwargs):
        """
        Log at ERROR level with optional exception details
        
        Args:
            message: Log message
            context: Optional log context
            exception: Optional exception to include
            **kwargs: Additional fields
        """
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
            kwargs["stacktrace"] = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        
        self.log(LogLevel.ERROR, message, context, **kwargs)
    
    def critical(self, message: str, context: Optional[LogContext] = None, exception: Optional[Exception] = None, **kwargs):
        """Log at CRITICAL level with optional exception details"""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
            kwargs["stacktrace"] = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        
        self.log(LogLevel.CRITICAL, message, context, **kwargs)

class MetricsCollector:
    """
    Metrics collection and reporting
    """
    
    def __init__(self, service_name: str, enable_prometheus: bool = True):
        """
        Initialize metrics collector
        
        Args:
            service_name: Name of the service
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.service_name = service_name
        self.enable_prometheus = enable_prometheus and HAS_PROMETHEUS
        
        # Metrics storage (for non-Prometheus mode)
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
        self.meters = {}
        
        # Prometheus registry
        self.prometheus_registry = None
        if self.enable_prometheus:
            self.prometheus_registry = prometheus_client.CollectorRegistry()
            
            # Start Prometheus HTTP server
            self.start_prometheus_server()
        
        # OpenTelemetry integration
        self.otel_meter = None
        if HAS_OPENTELEMETRY:
            meter_provider = MeterProvider()
            self.otel_meter = meter_provider.get_meter(self.service_name)
        
        logger.info(f"MetricsCollector initialized for {service_name}, prometheus={self.enable_prometheus}")
    
    def start_prometheus_server(self, port: int = 9090):
        """
        Start Prometheus metrics HTTP server
        
        Args:
            port: HTTP port to listen on
        """
        if self.enable_prometheus:
            prometheus_client.start_http_server(port, registry=self.prometheus_registry)
            logger.info(f"Prometheus metrics server started on port {port}")
    
    def counter(self, name: str, description: str, labels: List[str] = None):
        """
        Create or get a counter metric
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional label names
            
        Returns:
            Counter metric
        """
        full_name = f"{self.service_name}_{name}"
        labels = labels or []
        
        if self.enable_prometheus:
            # Create Prometheus counter
            if full_name not in self.counters:
                self.counters[full_name] = prometheus_client.Counter(
                    full_name, 
                    description,
                    labels,
                    registry=self.prometheus_registry
                )
            
            return self.counters[full_name]
        else:
            # Create simple counter
            if full_name not in self.counters:
                self.counters[full_name] = _SimpleCounter(full_name, description, labels)
            
            return self.counters[full_name]
    
    def gauge(self, name: str, description: str, labels: List[str] = None):
        """
        Create or get a gauge metric
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional label names
            
        Returns:
            Gauge metric
        """
        full_name = f"{self.service_name}_{name}"
        labels = labels or []
        
        if self.enable_prometheus:
            # Create Prometheus gauge
            if full_name not in self.gauges:
                self.gauges[full_name] = prometheus_client.Gauge(
                    full_name, 
                    description,
                    labels,
                    registry=self.prometheus_registry
                )
            
            return self.gauges[full_name]
        else:
            # Create simple gauge
            if full_name not in self.gauges:
                self.gauges[full_name] = _SimpleGauge(full_name, description, labels)
            
            return self.gauges[full_name]
    
    def histogram(self, name: str, description: str, labels: List[str] = None, buckets: List[float] = None):
        """
        Create or get a histogram metric
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional label names
            buckets: Optional histogram buckets
            
        Returns:
            Histogram metric
        """
        full_name = f"{self.service_name}_{name}"
        labels = labels or []
        buckets = buckets or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        if self.enable_prometheus:
            # Create Prometheus histogram
            if full_name not in self.histograms:
                self.histograms[full_name] = prometheus_client.Histogram(
                    full_name, 
                    description,
                    labels,
                    buckets=buckets,
                    registry=self.prometheus_registry
                )
            
            return self.histograms[full_name]
        else:
            # Create simple histogram
            if full_name not in self.histograms:
                self.histograms[full_name] = _SimpleHistogram(full_name, description, labels, buckets)
            
            return self.histograms[full_name]
    
    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """
        Increment a counter
        
        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels
        """
        if name.startswith(f"{self.service_name}_"):
            metric_name = name
        else:
            metric_name = f"{self.service_name}_{name}"
        
        # Default labels
        labels = labels or {}
        
        if metric_name in self.counters:
            counter = self.counters[metric_name]
            if self.enable_prometheus:
                if labels:
                    counter.labels(**labels).inc(value)
                else:
                    counter.inc(value)
            else:
                counter.inc(value, labels)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """
        Set a gauge value
        
        Args:
            name: Metric name
            value: Value to set
            labels: Optional labels
        """
        if name.startswith(f"{self.service_name}_"):
            metric_name = name
        else:
            metric_name = f"{self.service_name}_{name}"
        
        # Default labels
        labels = labels or {}
        
        if metric_name in self.gauges:
            gauge = self.gauges[metric_name]
            if self.enable_prometheus:
                if labels:
                    gauge.labels(**labels).set(value)
                else:
                    gauge.set(value)
            else:
                gauge.set(value, labels)
    
    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        """
        Observe a histogram value
        
        Args:
            name: Metric name
            value: Value to observe
            labels: Optional labels
        """
        if name.startswith(f"{self.service_name}_"):
            metric_name = name
        else:
            metric_name = f"{self.service_name}_{name}"
        
        # Default labels
        labels = labels or {}
        
        if metric_name in self.histograms:
            histogram = self.histograms[metric_name]
            if self.enable_prometheus:
                if labels:
                    histogram.labels(**labels).observe(value)
                else:
                    histogram.observe(value)
            else:
                histogram.observe(value, labels)
    
    def timing(self, name: str, func):
        """
        Decorator to time a function and record to histogram
        
        Args:
            name: Metric name
            func: Function to time
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                self.observe(name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.observe(name, duration, {"status": "error"})
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.observe(name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.observe(name, duration, {"status": "error"})
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @contextmanager
    def timed_block(self, name: str, labels: Dict[str, str] = None):
        """
        Context manager to time a block of code
        
        Args:
            name: Metric name
            labels: Optional labels
            
        Yields:
            None
        """
        start_time = time.time()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            if labels is None:
                labels = {}
            if not success:
                labels["status"] = "error"
            self.observe(name, duration, labels)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all current metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Counters
        for name, counter in self.counters.items():
            if self.enable_prometheus:
                metrics[name] = prometheus_client.generate_latest(counter).decode('utf-8')
            else:
                metrics[name] = counter.get_value()
        
        # Gauges
        for name, gauge in self.gauges.items():
            if self.enable_prometheus:
                metrics[name] = prometheus_client.generate_latest(gauge).decode('utf-8')
            else:
                metrics[name] = gauge.get_value()
        
        # Histograms
        for name, histogram in self.histograms.items():
            if self.enable_prometheus:
                metrics[name] = prometheus_client.generate_latest(histogram).decode('utf-8')
            else:
                metrics[name] = histogram.get_values()
        
        return metrics

# Simple metric classes for non-Prometheus mode
class _SimpleCounter:
    """Simple counter implementation"""
    
    def __init__(self, name, description, labels):
        self.name = name
        self.description = description
        self.label_names = labels
        self.values = {}
        self.default_value = 0
    
    def inc(self, value=1, labels=None):
        """Increment counter"""
        key = self._get_key(labels)
        if key not in self.values:
            self.values[key] = 0
        self.values[key] += value
    
    def get_value(self, labels=None):
        """Get counter value"""
        key = self._get_key(labels)
        return self.values.get(key, 0)
    
    def _get_key(self, labels):
        """Convert labels to a hashable key"""
        if not labels:
            return "default"
        
        # Sort labels by key for consistent hashing
        return tuple(sorted(labels.items()))

class _SimpleGauge:
    """Simple gauge implementation"""
    
    def __init__(self, name, description, labels):
        self.name = name
        self.description = description
        self.label_names = labels
        self.values = {}
    
    def set(self, value, labels=None):
        """Set gauge value"""
        key = self._get_key(labels)
        self.values[key] = value
    
    def inc(self, value=1, labels=None):
        """Increment gauge"""
        key = self._get_key(labels)
        if key not in self.values:
            self.values[key] = 0
        self.values[key] += value
    
    def dec(self, value=1, labels=None):
        """Decrement gauge"""
        key = self._get_key(labels)
        if key not in self.values:
            self.values[key] = 0
        self.values[key] -= value
    
    def get_value(self, labels=None):
        """Get gauge value"""
        key = self._get_key(labels)
        return self.values.get(key, 0)
    
    def _get_key(self, labels):
        """Convert labels to a hashable key"""
        if not labels:
            return "default"
        
        # Sort labels by key for consistent hashing
        return tuple(sorted(labels.items()))

class _SimpleHistogram:
    """Simple histogram implementation"""
    
    def __init__(self, name, description, labels, buckets):
        self.name = name
        self.description = description
        self.label_names = labels
        self.buckets = sorted(buckets)
        self.values = {}
        self.counts = {}
        self.sums = {}
    
    def observe(self, value, labels=None):
        """Observe a value"""
        key = self._get_key(labels)
        
        # Initialize if first observation
        if key not in self.values:
            self.values[key] = {bucket: 0 for bucket in self.buckets}
            self.values[key]["inf"] = 0
            self.counts[key] = 0
            self.sums[key] = 0
        
        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self.values[key][bucket] += 1
        
        # Always update the inf bucket
        self.values[key]["inf"] += 1
        
        # Update count and sum
        self.counts[key] += 1
        self.sums[key] += value
    
    def get_values(self, labels=None):
        """Get histogram values"""
        key = self._get_key(labels)
        
        if key not in self.values:
            return {
                "buckets": {bucket: 0 for bucket in self.buckets},
                "count": 0,
                "sum": 0
            }
        
        return {
            "buckets": self.values[key],
            "count": self.counts[key],
            "sum": self.sums[key]
        }
    
    def _get_key(self, labels):
        """Convert labels to a hashable key"""
        if not labels:
            return "default"
        
        # Sort labels by key for consistent hashing
        return tuple(sorted(labels.items()))

class Tracer:
    """
    Distributed tracing functionality
    """
    
    def __init__(self, service_name: str, use_opentelemetry: bool = True):
        """
        Initialize tracer
        
        Args:
            service_name: Name of the service
            use_opentelemetry: Whether to use OpenTelemetry
        """
        self.service_name = service_name
        self.use_opentelemetry = use_opentelemetry and HAS_OPENTELEMETRY
        
        # Local trace context
        self.active_spans = {}
        
        # OpenTelemetry setup
        self.otel_tracer = None
        if self.use_opentelemetry:
            tracer_provider = TracerProvider()
            otel_trace.set_tracer_provider(tracer_provider)
            self.otel_tracer = otel_trace.get_tracer(service_name)
        
        # Generate a hostname identifier
        self.hostname = socket.gethostname()
        
        logger.info(f"Tracer initialized for {service_name}, opentelemetry={self.use_opentelemetry}")
    
    def start_span(self, name: