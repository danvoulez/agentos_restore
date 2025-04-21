import redis.asyncio as redis
from app.core.config import get_settings

_settings = get_settings()

def get_redis_client() -> redis.Redis:
    return redis.from_url(_settings.SALES_CELERY_BROKER, decode_responses=True)