"""
API for deploying and managing comprehensive span collections
Created: 2025-07-19 05:09:54 UTC
"""
from fastapi import FastAPI, HTTPException, Depends, Request, Header
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import uvicorn
import os
import json
import asyncio
from datetime import datetime

from core.diamond_span import DiamondSpan
from core.scarcity_engine import ScarcityEngine
from farm.diamond_farm import DiamondFarm
from farm.span_collection import SpanCollection

# Initialize components
app = FastAPI(title="Diamond Span Deployment API", version="1.0.0")
GOD_KEY = os.getenv("GOD_KEY", "user_special_key_never_share")

# Create farm instance
scarcity_engine = ScarcityEngine()
farm = DiamondFarm(scarcity_engine, god_key=GOD_KEY)

# Request models
class SpanRequest(BaseModel):
    id: Optional[str] = None
    kind: str
    verb: str
    actor: str
    object: str
    payload: Dict[str, Any]
    parent_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class DeployRequest(BaseModel):
    collection_path: str
    is_god_mode: bool = False

class ServiceRequest(BaseModel):
    service: str
    workers: int = 4

# Authentication dependency
async def verify_god_key(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    scheme, token = authorization.split()
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    
    if token != GOD_KEY:
        raise HTTPException(status_code=403, detail="Invalid credentials")
    
    return True

@app.on_event("startup")
async def startup_event():
    # Start the farm
    await farm.start_mining(num_miners=4)

@app.post("/spans/register")
async def register_span(span: SpanRequest, is_god: bool = Depends(verify_god_key)):
    """Register a new span"""
    # Convert to DiamondSpan
    diamond_span = DiamondSpan(
        id=span.id,
        parent_ids=span.parent_ids or [],
        content={
            "kind": span.kind,
            "verb": span.verb,
            "actor": span.actor,
            "object": span.object,
            "payload": span.payload
        },
        metadata=span.metadata or {}
    )
    
    # Register span
    try:
        span_id = await farm.register_span(diamond_span)
        return {"status": "success", "id": span_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/spans/deploy")
async def deploy_collection(request: DeployRequest, is_god: bool = Depends(verify_god_key)):
    """Deploy a full span collection"""
    try:
        # Load collection from file
        with open(request.collection_path, 'r') as f:
            collection = json.load(f)
        
        # Register spans in dependency order
        registered = []
        remaining = list(collection.keys())
        
        while remaining and len(registered) < len(collection):
            progress_made = False
            
            for span_id in list(remaining):
                span = collection[span_id]
                
                # Check if parents are registered
                parents_ready = True
                for parent_id in span.get("parent_ids", []):
                    if parent_id not in registered and parent_id in collection:
                        parents_ready = False
                        break
                
                if parents_ready:
                    # Register span
                    diamond_span = DiamondSpan(
                        id=span_id,
                        parent_ids=span.get("parent_ids", []),
                        content=span.get("content", {}),
                        metadata=span.get("metadata", {})
                    )
                    
                    await farm.register_span(diamond_span)
                    registered.append(span_id)
                    remaining.remove(span_id)
                    progress_made = True
            
            if not progress_made:
                # Circular dependency or other issue
                break
        
        return {
            "status": "success", 
            "registered": len(registered),
            "failed": len(remaining),
            "total": len(collection)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/services/start")
async def start_service(request: ServiceRequest, is_god: bool = Depends(verify_god_key)):
    """Start a service"""
    if request.service == "mining":
        result = await farm.start_mining(num_miners=request.workers)
        return {"status": "success", "service": request.service, "result": result}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown service: {request.service}")

@app.get("/market/stats")
async def get_market_stats():
    """Get market statistics"""
    return await farm.get_market_stats()

@app.post("/generate")
async def generate_spans(is_god: bool = Depends(verify_god_key)):
    """Generate a new comprehensive span collection"""
    try:
        collector = SpanCollection(god_key=GOD_KEY)
        result = collector.generate_comprehensive_set()
        
        # Export collection
        output_path = f"data/spans_generated_{int(datetime.now().timestamp())}.json"
        collector.export_collection(output_path)
        
        return {
            "status": "success",
            "total_spans": result["total_spans"],
            "by_kind": result["by_kind"],
            "output_path": output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("deploy_spans_api:app", host="0.0.0.0", port=8080, reload=True)