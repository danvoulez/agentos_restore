"""
API for interacting with the Diamond Span Farm
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import json
import time
from core.diamond_span import DiamondSpan
from core.scarcity_engine import ScarcityEngine
from farm.diamond_farm import DiamondFarm

# Initialize components
scarcity_engine = ScarcityEngine()
farm = DiamondFarm(scarcity_engine, god_key=os.getenv("GOD_KEY", "user_special_key_never_share"))

# Start mining threads
farm.start_mining(num_miners=4)

# Create FastAPI app
app = FastAPI(title="Diamond Span API", version="1.0.0")

# Request models
class SpanRequest(BaseModel):
    content: Dict[str, Any]
    parent_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class MineRequest(BaseModel):
    prompt: str
    parent_ids: Optional[List[str]] = None
    is_god_mode: Optional[bool] = False

# Response models
class SpanResponse(BaseModel):
    id: str
    content: Dict[str, Any]
    energy: float
    signature: str
    created_at: str
    value: float

# API Authentication middleware
async def verify_god_key(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header:
        scheme, token = auth_header.split()
        if scheme.lower() == "bearer" and token == os.getenv("GOD_KEY", "user_special_key_never_share"):
            return True
    return False

# Routes
@app.get("/")
async def root():
    return {"message": "Diamond Span API", "version": "1.0.0"}

@app.post("/spans/register", response_model=dict)
async def register_span(span_data: SpanRequest):
    """Register a new Diamond Span"""
    span = DiamondSpan(
        content=span_data.content,
        parent_ids=span_data.parent_ids,
        metadata=span_data.metadata
    )
    
    span_id = farm.register_span(span)
    return {"id": span_id, "message": "Span registered"}

@app.post("/spans/mine", response_model=dict)
async def mine_span(mine_request: MineRequest):
    """Mine a new Diamond Span from content"""
    content = {"text": mine_request.prompt}
    metadata = {}
    
    if mine_request.is_god_mode:
        is_god = await verify_god_key(Request.headers)
        if is_god:
            metadata["creator"] = os.getenv("GOD_KEY")
        else:
            raise HTTPException(status_code=403, detail="Unauthorized for god mode")
    
    result = farm.submit_for_mining(
        content=content,
        parent_ids=mine_request.parent_ids,
        metadata=metadata
    )
    
    return {"status": "submitted", "message": "Span submitted for mining"}

@app.get("/spans/{span_id}", response_model=SpanResponse)
async def get_span(span_id: str):
    """Get a specific Diamond Span by ID"""
    span = farm.get_span(span_id)
    if not span:
        raise HTTPException(status_code=404, detail="Span not found")
    
    value = farm.calculate_span_value(span)
    
    return {
        "id": span.id,
        "content": span.content,
        "energy": span.energy,
        "signature": span.signature,
        "created_at": span.created_at.isoformat(),
        "value": value
    }

@app.get("/spans", response_model=List[SpanResponse])
async def list_spans():
    """List all spans in the farm"""
    spans = farm.get_all_spans()
    
    result = []
    for span in spans:
        value = farm.calculate_span_value(span)
        result.append({
            "id": span.id,
            "content": span.content,
            "energy": span.energy,
            "signature": span.signature,
            "created_at": span.created_at.isoformat(),
            "value": value
        })
        
    return result

@app.get("/market/stats")
async def get_market_stats():
    """Get market statistics"""
    return farm.get_market_stats()

@app.post("/god/verify")
async def verify_god_mode(request: Request):
    """Verify god mode access"""
    is_god = await verify_god_key(request)
    if is_god:
        return {"status": "verified", "message": "God mode active"}
    raise HTTPException(status_code=403, detail="Invalid god key")

# Serve the application
if __name__ == "__main__":
    uvicorn.run("diamond_api:app", host="0.0.0.0", port=8080, reload=True)