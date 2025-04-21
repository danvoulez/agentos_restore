from loguru import logger
import sys, os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, enqueue=True, backtrace=True, diagnose=False)