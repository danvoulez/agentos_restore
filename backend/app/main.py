# promptos_backend/app/main.py
from fastapi import FastAPI
from app.api.v1.api import router as api_router
from loguru import logger

app = FastAPI(title="PromptOS Backend")

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up PromptOS Backend service")
    # Add startup logic here

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down PromptOS Backend service")
    # Add shutdown logic here
