# agentos-pessoas/logs/logger.py
import sys
import logging
from loguru import logger
import uuid
import contextvars

# Import settings to get log level dynamically
try:
    from schemas.base import settings
    LOG_LEVEL = settings.LOG_LEVEL
except ImportError:
    LOG_LEVEL = "INFO"  # Fallback if settings cannot be imported early

# Context variable for trace ID
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)

class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward Loguru sinks."""
    def emit(self, record: logging.LogRecord):
        try:
        )

def setup_logging(service_name: str = "agentos-pessoas"):
    """Configure Loguru logger."""
    logger.remove()
    log_level = LOG_LEVEL.upper()  # Use level from settings/default

    logger.add(
        sys.stderr,
        diagnose=(log_level == "DEBUG")
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logger.info(f"Logging configured for service '{service_name}' at level {log_level}")
