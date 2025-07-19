"""
Miner API for LogLineOS/DiamondSpan
Provides HTTP endpoints for diamond mining operations
Created: 2025-07-19 05:42:25 UTC
User: danvoulez
"""
from fastapi import FastAPI, HTTPException, Depends, Header, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import uvicorn
import os
import json
import asyncio
import logging
import time
from datetime import datetime

from farm.miner import DiamondMiner, MinerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Miner_API")

# Initialize components
app = FastAPI(title="LogLineOS Diamond Miner API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global miner configuration
miner_config = MinerConfig(
    threads=int(os.environ.get("MINER_THREADS", "4")),
    target_difficulty=float(os.environ.get("MINER_DIFFICULTY", "1.0")),
    energy_factor=float(os.environ.get("ENERGY_FACTOR", "1.0")),
    reward_schema=os.environ.get("REWARD_SCHEMA", "logarithmic"),
    god_key=os.environ.get("GOD_KEY", "user_special_key_never_share")
)

# Initialize miner
diamond_miner = DiamondMiner(config=miner_config)

# Request models
class MineRequest(BaseModel):
    content: Union[str, Dict[str, Any]]
    parent_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    priority: Optional[float] = 1.0

class ConfigUpdateRequest(BaseModel):
    threads: Optional[int] = None
    target_difficulty: Optional[float] = None
    energy_factor: Optional[float] = None
    reward_schema: Optional[str] = None
    adaptive_difficulty: Optional[bool] = None
    min_quality_threshold: Optional[float] = None
    mutation_rate: Optional[float] = None

# Authentication middleware
async def verify_api_key(api_key: str = Header(None, alias="X-API-Key")):
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required")
    return api_key

async def verify_god_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    scheme, token = authorization.split()
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    
    if token != miner_config.god_key:
        raise HTTPException(status_code=403, detail="Invalid god key")
    
    return True

# Routes
@app.get("/")
async def root():
    return {
        "name": "LogLineOS Diamond Miner API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/start")
async def start_mining(api_key: str = Depends(verify_api_key)):
    """Start mining operations"""
    try:
        result = diamond_miner.start_mining()
        return result
    except Exception as e:
        logger.error(f"Error starting mining: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_mining(api_key: str = Depends(verify_api_key)):
    """Stop mining operations"""
    try:
        result = diamond_miner.stop_mining()
        return result
    except Exception as e:
        logger.error(f"Error stopping mining: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mine")
async def mine_span(request: MineRequest, api_key: str = Depends(verify_api_key)):
    """Submit content for mining"""
    try:
        result = diamond_miner.submit_for_mining(
            content=request.content,
            parent_ids=request.parent_ids,
            metadata=request.metadata,
            priority=request.priority
        )
        return result
    except Exception as e:
        logger.error(f"Error submitting for mining: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get mining statistics"""
    try:
        stats = diamond_miner.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def update_config(request: ConfigUpdateRequest, is_god: bool = Depends(verify_god_key)):
    """Update miner configuration (requires god key)"""
    try:
        # Update configuration values that are provided
        updates = {}
        
        if request.threads is not None:
            diamond_miner.config.threads = max(1, min(32, request.threads))
            updates["threads"] = diamond_miner.config.threads
            
        if request.target_difficulty is not None:
            diamond_miner.config.target_difficulty = max(0.1, min(100.0, request.target_difficulty))
            updates["target_difficulty"] = diamond_miner.config.target_difficulty
            
        if request.energy_factor is not None:
            diamond_miner.config.energy_factor = max(0.1, min(10.0, request.energy_factor))
            updates["energy_factor"] = diamond_miner.config.energy_factor
            
        if request.reward_schema is not None:
            valid_schemas = ["logarithmic", "linear", "quadratic"]
            if request.reward_schema not in valid_schemas:
                raise HTTPException(status_code=400, detail=f"Invalid reward schema. Must be one of: {valid_schemas}")
            diamond_miner.config.reward_schema = request.reward_schema
            updates["reward_schema"] = diamond_miner.config.reward_schema
            
        if request.adaptive_difficulty is not None:
            diamond_miner.config.adaptive_difficulty = request.adaptive_difficulty
            updates["adaptive_difficulty"] = diamond_miner.config.adaptive_difficulty
            
        if request.min_quality_threshold is not None:
            diamond_miner.config.min_quality_threshold = max(0.0, min(10.0, request.min_quality_threshold))
            updates["min_quality_threshold"] = diamond_miner.config.min_quality_threshold
            
        if request.mutation_rate is not None:
            diamond_miner.config.mutation_rate = max(0.0, min(0.5, request.mutation_rate))
            updates["mutation_rate"] = diamond_miner.config.mutation_rate
        
        return {
            "status": "success",
            "updates": updates,
            "config": {
                "threads": diamond_miner.config.threads,
                "target_difficulty": diamond_miner.config.target_difficulty,
                "energy_factor": diamond_miner.config.energy_factor,
                "reward_schema": diamond_miner.config.reward_schema,
                "adaptive_difficulty": diamond_miner.config.adaptive_difficulty,
                "min_quality_threshold": diamond_miner.config.min_quality_threshold,
                "mutation_rate": diamond_miner.config.mutation_rate
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mined")
async def get_mined_spans(
    limit: int = Query(10, ge=1, le=100),
    api_key: str = Depends(verify_api_key)
):
    """Get recently mined spans"""
    try:
        mined_spans = []
        span_count = min(limit, diamond_miner.mined_spans.qsize())
        
        # Create a copy of mined spans to avoid modifying the queue
        spans_copy = []
        for _ in range(span_count):
            try:
                span = diamond_miner.mined_spans.get_nowait()
                spans_copy.append(span)
                diamond_miner.mined_spans.put(span)  # Put it back
            except:
                break
        
        return {
            "spans": spans_copy,
            "total": diamond_miner.mined_spans.qsize()
        }
    except Exception as e:
        logger.error(f"Error getting mined spans: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pending")
async def get_pending_count(api_key: str = Depends(verify_api_key)):
    """Get number of pending mining jobs"""
    return {
        "pending": diamond_miner.pending_spans.qsize(),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("miner_api:app", host="0.0.0.0", port=8085, reload=True)