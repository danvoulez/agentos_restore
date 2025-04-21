from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api import api_router
from app.core.logging_config import logger
from app.db.mongo_client import connect_to_mongo, close_mongo_connection

app = FastAPI(title="AgentOS Sales")

@app.on_event("startup")
async def on_startup():
    await connect_to_mongo()
    Instrumentator().instrument(app).expose(app)
    logger.info("API started")

@app.on_event("shutdown")
async def on_shutdown():
    await close_mongo_connection()
    logger.info("API stopped")

app.include_router(api_router, prefix="/api/v1")