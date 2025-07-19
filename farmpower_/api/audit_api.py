"""
Audit API for LogLineOS/DiamondSpan
Provides access to the audit system
Created: 2025-07-19 05:28:32 UTC
User: danvoulez
"""
from fastapi import FastAPI, HTTPException, Depends, Query, Header
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import json
import asyncio
from datetime import datetime, timedelta

from core.audit_system import AuditSystem

# Initialize components
app = FastAPI(title="LogLineOS Audit API", version="1.0.0")
GOD_KEY = os.getenv("GOD_KEY", "user_special_key_never_share")
audit_system = AuditSystem()

# Request models
class CreateAuditEntryRequest(BaseModel):
    operation: str
    actor: str
    target_type: str
    target_id: str
    status: Optional[str] = "success"
    details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class AuditQueryRequest(BaseModel):
    operations: Optional[List[str]] = None
    actors: Optional[List[str]] = None
    target_types: Optional[List[str]] = None
    target_ids: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    limit: Optional[int] = 100
    offset: Optional[int] = 0

class ValidateAuditRequest(BaseModel):
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

# Authentication dependency
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
    
    if token != GOD_KEY:
        raise HTTPException(status_code=403, detail="Invalid credentials")
    
    return True

@app.on_event("startup")
async def startup_event():
    await audit_system.initialize()

@app.get("/")
async def root():
    return {
        "name": "LogLineOS Audit API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/entries")
async def create_audit_entry(request: CreateAuditEntryRequest, api_key: str = Depends(verify_api_key)):
    """Create a new audit entry"""
    entry = await audit_system.create_entry(
        operation=request.operation,
        actor=request.actor,
        target_type=request.target_type,
        target_id=request.target_id,
        status=request.status,
        details=request.details,
        metadata=request.metadata
    )
    
    return {"status": "success", "entry": entry.to_dict()}

@app.get("/entries/{entry_id}")
async def get_audit_entry(entry_id: str, api_key: str = Depends(verify_api_key)):
    """Get an audit entry by ID"""
    entry = await audit_system.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Audit entry not found")
    
    return {"entry": entry.to_dict()}

@app.post("/query")
async def query_audit_entries(request: AuditQueryRequest, api_key: str = Depends(verify_api_key)):
    """Query audit entries with filters"""
    entries, total = await audit_system.query_entries(
        operations=request.operations,
        actors=request.actors,
        target_types=request.target_types,
        target_ids=request.target_ids,
        statuses=request.statuses,
        start_time=request.start_time,
        end_time=request.end_time,
        limit=request.limit,
        offset=request.offset
    )
    
    return {
        "entries": [entry.to_dict() for entry in entries],
        "total": total,
        "limit": request.limit,
        "offset": request.offset
    }

@app.post("/validate")
async def validate_audit_trail(request: ValidateAuditRequest, is_god: bool = Depends(verify_god_key)):
    """Validate the integrity of the audit trail"""
    validation = await audit_system.validate_audit_trail(
        target_type=request.target_type,
        target_id=request.target_id,
        start_time=request.start_time,
        end_time=request.end_time
    )
    
    return {"validation": validation}

@app.post("/snapshot")
async def create_system_snapshot(is_god: bool = Depends(verify_god_key)):
    """Create a system snapshot for forensic purposes"""
    snapshot_id = await audit_system.create_system_snapshot()
    
    return {
        "status": "success", 
        "snapshot_id": snapshot_id,
        "created_at": datetime.now().isoformat()
    }

@app.get("/statistics")
async def get_audit_statistics(api_key: str = Depends(verify_api_key)):
    """Get audit system statistics"""
    stats = await audit_system.get_statistics()
    
    return {"statistics": stats}

@app.get("/recent")
async def get_recent_entries(
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 7 days
    limit: int = Query(50, ge=1, le=1000),
    api_key: str = Depends(verify_api_key)
):
    """Get recent audit entries"""
    start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
    
    entries, total = await audit_system.query_entries(
        start_time=start_time,
        limit=limit
    )
    
    return {
        "entries": [entry.to_dict() for entry in entries],
        "total": total,
        "hours": hours
    }

if __name__ == "__main__":
    uvicorn.run("audit_api:app", host="0.0.0.0", port=8083, reload=True)