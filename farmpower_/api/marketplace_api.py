"""
API for Diamond Span Marketplace
"""
from fastapi import FastAPI, HTTPException, Depends, Request, Header, Query
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import json
import asyncio
from datetime import datetime

from core.diamond_span import DiamondSpan
from core.scarcity_engine import ScarcityEngine
from farm.advanced_diamond_farm import EnhancedDiamondFarm
from farm.span_marketplace import SpanMarketplace

# Initialize components
app = FastAPI(title="Diamond Span Marketplace API", version="1.0.0")
GOD_KEY = os.getenv("GOD_KEY", "user_special_key_never_share")

# Create farm and marketplace instances
scarcity_engine = ScarcityEngine()
farm = EnhancedDiamondFarm(scarcity_engine, god_key=GOD_KEY)
marketplace = SpanMarketplace(farm)

# Request models
class ListingRequest(BaseModel):
    span_id: str
    price: float
    duration_hours: Optional[float] = 24.0

class AuctionRequest(BaseModel):
    span_id: str
    min_bid: float
    duration_hours: Optional[float] = 24.0

class BidRequest(BaseModel):
    amount: float

class BuyRequest(BaseModel):
    buyer: str

# Authentication dependency
async def verify_auth(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    scheme, token = authorization.split()
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    
    # In production, you would validate the token properly
    # For now, we'll just use a simple token validation
    return token

async def verify_god_key(authorization: Optional[str] = Header(None)):
    token = await verify_auth(authorization)
    if token != GOD_KEY:
        raise HTTPException(status_code=403, detail="Invalid credentials")
    return True

@app.on_event("startup")
async def startup_event():
    # Start the farm
    await farm.start_mining(num_miners=4)

@app.get("/marketplace/stats")
async def get_market_stats():
    """Get marketplace statistics"""
    return await marketplace.get_market_stats()

@app.post("/marketplace/listings")
async def create_listing(request: ListingRequest, token: str = Depends(verify_auth)):
    """Create a new listing"""
    result = await marketplace.create_listing(
        span_id=request.span_id,
        seller=token,  # Use the token as seller ID
        price=request.price,
        duration_hours=request.duration_hours
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.post("/marketplace/buy/{span_id}")
async def buy_span(span_id: str, request: BuyRequest, token: str = Depends(verify_auth)):
    """Buy a span"""
    result = await marketplace.buy_span(
        span_id=span_id,
        buyer=request.buyer
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.post("/marketplace/auctions")
async def create_auction(request: AuctionRequest, token: str = Depends(verify_auth)):
    """Create a new auction"""
    result = await marketplace.create_auction(
        span_id=request.span_id,
        seller=token,  # Use the token as seller ID
        min_bid=request.min_bid,
        duration_hours=request.duration_hours
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.post("/marketplace/auctions/{span_id}/bid")
async def place_bid(span_id: str, request: BidRequest, token: str = Depends(verify_auth)):
    """Place a bid on an auction"""
    result = await marketplace.place_bid(
        span_id=span_id,
        bidder=token,  # Use the token as bidder ID
        amount=request.amount
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.post("/marketplace/auctions/{span_id}/finalize")
async def finalize_auction(span_id: str, token: str = Depends(verify_auth)):
    """Finalize an auction"""
    result = await marketplace.finalize_auction(span_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.get("/marketplace/listings")
async def get_listings(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Get active listings"""
    return await marketplace.get_active_listings(page, page_size)

@app.get("/marketplace/auctions")
async def get_auctions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Get active auctions"""
    return await marketplace.get_active_auctions(page, page_size)

@app.get("/marketplace/price-history/{span_id}")
async def get_price_history(span_id: str):
    """Get price history for a span"""
    result = await marketplace.get_price_history(span_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.get("/marketplace/trades")
async def get_trades(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Get trade history"""
    trades = marketplace.trade_history
    
    # Sort by timestamp (newest first)
    trades.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_trades = trades[start:end]
    
    return {
        "status": "success",
        "trades": page_trades,
        "total": len(trades),
        "page": page,
        "page_size": page_size,
        "total_pages": (len(trades) + page_size - 1) // page_size
    }

@app.post("/maintenance/clean-expired")
async def clean_expired(is_god: bool = Depends(verify_god_key)):
    """Clean expired listings"""
    result = await marketplace.clean_expired_listings()
    return result

if __name__ == "__main__":
    uvicorn.run("marketplace_api:app", host="0.0.0.0", port=8081, reload=True)