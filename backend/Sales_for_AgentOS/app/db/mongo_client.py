from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import get_settings
from app.core.logging_config import logger

_client: AsyncIOMotorClient | None = None

async def connect_to_mongo() -> None:
    global _client
    if not _client:
        uri = get_settings().SALES_MONGODB_URI
        _client = AsyncIOMotorClient(uri)
        logger.info("Mongo connected")

async def close_mongo_connection() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("Mongo connection closed")

def get_database() -> AsyncIOMotorDatabase:
    if _client is None:
        raise RuntimeError("Call connect_to_mongo() first")
    return _client[get_settings().SALES_MONGO_DB_NAME]