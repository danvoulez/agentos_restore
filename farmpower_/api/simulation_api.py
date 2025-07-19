"""
Simulation API for LogLineOS/DiamondSpan
Provides HTTP endpoints for simulation capabilities
Created: 2025-07-19 05:37:29 UTC
User: danvoulez
"""
from fastapi import FastAPI, HTTPException, Depends, Header, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import json
import asyncio
import logging
from datetime import datetime

from core.simulation_engine import SimulationEngine, SimulationConfig, SimulationResult
try:
    from core.logline_vm import LogLineVM
    HAS_VM = True
except ImportError:
    HAS_VM = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Simulation_API")

# Initialize components
app = FastAPI(title="LogLineOS Simulation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation engine
sim_config = SimulationConfig(
    max_depth=5,
    max_branches=10,
    max_duration_seconds=60,
    resolution_ms=100,
    tension_threshold=17.3,
    randomize_outcomes=True
)

vm = LogLineVM() if HAS_VM else None
sim_engine = SimulationEngine(config=sim_config, vm=vm)

# Request models
class SimulateRequest(BaseModel):
    span: Dict[str, Any]
    config_override: Optional[Dict[str, Any]] = None

class PredictorRequest(BaseModel):
    kind: str
    module_path: str
    function_name: str

# Authentication middleware
async def verify_api_key(api_key: str = Header(None, alias="X-API-Key")):
    # In production, you would validate against a database of API keys
    # For now, we'll accept any key for demonstration
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required")
    return api_key

# Routes
@app.get("/")
async def root():
    return {
        "name": "LogLineOS Simulation API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/simulate")
async def simulate_span(request: SimulateRequest, api_key: str = Depends(verify_api_key)):
    """Simulate a span execution"""
    try:
        result = await sim_engine.simulate_span(request.span, request.config_override)
        return {
            "simulation_id": result.simulation_id,
            "success": result.success,
            "outcome_count": len(result.outcome_spans),
            "metrics": result.metrics,
            "duration_seconds": result.duration_seconds
        }
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulations")
async def list_simulations(api_key: str = Depends(verify_api_key)):
    """List all simulations"""
    try:
        simulations = await sim_engine.get_simulation_history()
        return {"simulations": simulations}
    except Exception as e:
        logger.error(f"Error listing simulations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulations/{simulation_id}")
async def get_simulation(simulation_id: str, api_key: str = Depends(verify_api_key)):
    """Get a specific simulation"""
    try:
        simulation = await sim_engine.get_simulation_history(simulation_id)
        if "error" in simulation:
            raise HTTPException(status_code=404, detail=simulation["error"])
        return simulation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting simulation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulations/{simulation_id}/outcomes")
async def get_simulation_outcomes(simulation_id: str, api_key: str = Depends(verify_api_key)):
    """Get outcomes for a specific simulation"""
    try:
        simulation = await sim_engine.get_simulation_history(simulation_id)
        if "error" in simulation:
            raise HTTPException(status_code=404, detail=simulation["error"])
        return {"outcomes": simulation.get("outcome_spans", [])}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting outcomes: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get simulation statistics"""
    try:
        stats = await sim_engine.get_simulation_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictors")
async def register_predictor(request: PredictorRequest, api_key: str = Depends(verify_api_key)):
    """Register a custom predictor"""
    try:
        # Dynamic import of predictor function
        import importlib
        module = importlib.import_module(request.module_path)
        function = getattr(module, request.function_name)
        
        # Register with the simulation engine
        sim_engine.register_predictor(request.kind, function)
        
        return {
            "status": "success",
            "message": f"Registered predictor for kind: {request.kind}"
        }
    except Exception as e:
        logger.error(f"Error registering predictor: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("simulation_api:app", host="0.0.0.0", port=8084, reload=True)