# agentos-pessoas/main.py
from fastapi import FastAPI
from routes import profiles, roles, integrations
from loguru import logger

app = FastAPI(title="AgentOS Pessoas")

# Include routers
app.include_router(profiles.router, prefix="/profiles", tags=["Profiles"])
app.include_router(roles.router, prefix="/roles", tags=["Roles"])
app.include_router(integrations.router, prefix="/integrations", tags=["Integrations"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up AgentOS Pessoas service")
    # Add startup logic here

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AgentOS Pessoas service")
    # Add shutdown logic here
