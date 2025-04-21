# agentos-pessoas/db/mongo_client.py
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from schemas.base import settings  # Use the corrected settings
from loguru import logger

_mongo_client: AsyncIOMotorClient | None = None
_mongo_db: AsyncIOMotorDatabase | None = None

async def connect_to_mongo():
    global _mongo_client, _mongo_db
    if _mongo_client and _mongo_db:
        logger.debug("MongoDB connection already established.")
        return
    try:
        mongo_uri = settings.MONGODB_URI
        logger.success(f"Connected to MongoDB database '{db_name}' successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to connect to MongoDB: {e}")
        raise RuntimeError(f"Failed to connect to MongoDB: {e}") from e

async def close_mongo_connection():
    global _mongo_client, _mongo_db
    if _mongo_client:
        logger.info("Closing MongoDB connection...")

def get_database() -> AsyncIOMotorDatabase:
    if _mongo_db is None:
        logger.error("Database instance is not available. Ensure connect_to_mongo() was called.")
        raise RuntimeError("Database not connected")
    return _mongo_db
